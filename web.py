import json
import logging
import mmap
import os
import socket
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path


if __name__ == "__main__" and sys.prefix == sys.base_prefix:
    print("请使用 ./run_web.sh 启动，以使用项目虚拟环境。", file=sys.stderr)
    raise SystemExit(1)


import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


STREAM_FPS = 60
FRAME_INTERVAL = 1.0 / STREAM_FPS
JSON_CACHE_TTL = 0.05
SHARED_MEMORY_PATH = "/dev/shm/sp_vision_25_frame"
SHARED_MEMORY_SIZE = 2 * 1024 * 1024
DATA_PATH = "/dev/shm/sp_vision_25_data.json"
LOG_PATH = "/dev/shm/sp_vision_25_log.json"
HOST = "0.0.0.0"
PORT = 9000
PROJECT_ROOT = Path(__file__).resolve().parent

fd = None
mapfile = None
use_shared_memory = False
latest_jpg = None
latest_seq = 0
frame_cond = threading.Condition()
stop_event = threading.Event()
reader_thread = None
json_cache_lock = threading.Lock()
json_cache = {}


class CachedStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response


def get_local_ip():
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        if sock is not None:
            sock.close()


def close_shared_memory():
    global fd, mapfile, use_shared_memory

    use_shared_memory = False
    if mapfile is not None:
        try:
            mapfile.close()
        except OSError:
            pass
        mapfile = None
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
        fd = None


def init_shared_memory(verbose=True):
    global fd, mapfile, use_shared_memory

    close_shared_memory()
    try:
        fd = os.open(SHARED_MEMORY_PATH, os.O_RDONLY)
        mapfile = mmap.mmap(fd, SHARED_MEMORY_SIZE, access=mmap.ACCESS_READ)
        use_shared_memory = True
        if verbose:
            print("共享内存初始化成功")
        return True
    except OSError as error:
        if verbose:
            print(f"[ERROR] 共享内存初始化失败: {error}")
    except ValueError as error:
        if verbose:
            print(f"[ERROR] 共享内存初始化失败: {error}")

    close_shared_memory()
    return False


def read_jpeg_from_shared_memory():
    if not (use_shared_memory and mapfile is not None):
        return None

    try:
        mapfile.seek(0)
        size_bytes = mapfile.read(4)
        if len(size_bytes) != 4:
            return None

        jpg_size = struct.unpack("<I", size_bytes)[0]
        if not 0 < jpg_size <= SHARED_MEMORY_SIZE - 4:
            return None

        data = mapfile.read(jpg_size)
        if len(data) != jpg_size or not data.startswith(b"\xff\xd8"):
            return None
        return data
    except (OSError, ValueError):
        return None


def read_json_cached(path):
    now = time.monotonic()
    try:
        stat = os.stat(path)
    except OSError as error:
        raise FileNotFoundError(path) from error

    with json_cache_lock:
        cached = json_cache.get(path)
        if cached and cached["mtime_ns"] == stat.st_mtime_ns:
            cached["checked_at"] = now
            return cached["data"]
        if cached and now - cached["checked_at"] < JSON_CACHE_TTL:
            return cached["data"]

    with open(path, "r") as file:
        data = json.load(file)

    with json_cache_lock:
        json_cache[path] = {"mtime_ns": stat.st_mtime_ns, "checked_at": now, "data": data}
    return data


def clamp_int(value, default, lower, upper):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return min(max(parsed, lower), upper)


def trim_series_json(data, max_points):
    if not isinstance(data, dict) or max_points <= 0:
        return data

    time_values = data.get("time")
    if not isinstance(time_values, list) or len(time_values) <= max_points:
        return data

    start = len(time_values) - max_points
    return {
        key: value[start:] if isinstance(value, list) and len(value) == len(time_values) else value
        for key, value in data.items()
    }


def frame_reader_loop():
    global latest_jpg, latest_seq

    last_fix_attempt = 0.0
    last_fix_log = 0.0
    while not stop_event.is_set():
        jpg = read_jpeg_from_shared_memory()
        if jpg is not None:
            with frame_cond:
                latest_jpg = jpg
                latest_seq += 1
                frame_cond.notify_all()
            time.sleep(FRAME_INTERVAL)
            continue

        now = time.time()
        if now - last_fix_attempt > 5.0:
            verbose = now - last_fix_log > 30.0
            if verbose:
                print("尝试重新初始化共享内存...")
                last_fix_log = now
            init_shared_memory(verbose=verbose)
            last_fix_attempt = now
        time.sleep(0.2)


def mjpeg_stream():
    last_seq = -1
    while not stop_event.is_set():
        with frame_cond:
            frame_cond.wait_for(
                lambda: latest_seq != last_seq or stop_event.is_set(), timeout=1.0
            )
            if stop_event.is_set():
                break
            frame = latest_jpg
            last_seq = latest_seq

        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@asynccontextmanager
async def lifespan(app):
    global reader_thread

    stop_event.clear()
    init_shared_memory()
    reader_thread = threading.Thread(target=frame_reader_loop, daemon=True)
    reader_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        with frame_cond:
            frame_cond.notify_all()
        if reader_thread is not None:
            reader_thread.join(timeout=1.0)
            reader_thread = None
        close_shared_memory()


app = FastAPI(lifespan=lifespan)
app.mount(
    "/static",
    CachedStaticFiles(directory=PROJECT_ROOT / "static", check_dir=False),
    name="static",
)
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


def no_store_json(data, status=200):
    return JSONResponse(
        content=data,
        status_code=status,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/")
def index(request: Request):
    url = f"http://{get_local_ip()}:{PORT}"
    return templates.TemplateResponse(request, "index.html", {"server_url": url})


@app.get("/video")
def video_feed():
    return StreamingResponse(
        mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, max-age=0", "X-Accel-Buffering": "no"},
    )


@app.get("/data")
def get_data(max_points: str | None = None):
    try:
        points = clamp_int(max_points, 200, 10, 1000)
        return no_store_json(trim_series_json(read_json_cached(DATA_PATH), points))
    except Exception as error:
        return no_store_json({"error": str(error)}, status=500)


@app.get("/log")
def get_log():
    try:
        return no_store_json(read_json_cached(LOG_PATH))
    except Exception as error:
        return no_store_json({"error": str(error)}, status=500)


def main():
    logging.basicConfig(level=logging.INFO)
    print(f"Web 调试器已启动: http://{get_local_ip()}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, workers=1, reload=False)


if __name__ == "__main__":
    main()
