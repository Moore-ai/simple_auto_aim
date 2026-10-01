# simple_auto_aim 环境配置

## 1 项目环境

- 操作系统：Ubuntu 22.04
- 运算平台：NUC12WSKI7（i7-1260P，16GB）
- 相机型号：海康 MV-CS016-10UC
- 镜头型号：海康官方 6mm 镜头
- 下位机型号：RoboMaster 开发板 C 型（STM32F407）
- IMU 型号：C 板内置 BMI088
- 通信方式：USB2CAN 或 MicroUSB 虚拟串口

## 2 依赖安装

### 2.1 第三方 SDK

根据相机型号安装以下 SDK 之一：

- [MindVision SDK](https://mindvision.com.cn/category/software/sdk-installation-package/)
- [HikRobot SDK](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip)

安装以下库：

- [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
- [Ceres Solver](http://ceres-solver.org/installation.html)

### 2.2 系统软件包

```bash
sudo apt install -y \
    git \
    g++ \
    cmake \
    can-utils \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    nlohmann-json3-dev \
    openssh-server \
    screen
```

## 3 编译与运行

在项目根目录执行：

```bash
cmake -S . -B build
cmake --build build --parallel 1
```

运行自瞄或打符：

```bash
./build/standard configs/standard.yaml
./build/buff configs/standard.yaml --mode=0 # 小符
./build/buff configs/standard.yaml --mode=1 # 大符
```

标定可执行文件：

```bash
./build/capture -c configs/calibration.yaml -o assets/img_with_q
./build/calibrate_camera -c configs/calibration.yaml assets/img_with_q
./build/calibrate_handeye -c configs/calibration.yaml assets/img_with_q
```

## 4 开机自启

确保已安装 `screen`：

```bash
sudo apt install screen
```

创建自启文件：

```bash
mkdir -p ~/.config/autostart/
touch ~/.config/autostart/simple_auto_aim.desktop
```

写入以下内容。`Exec` 必须使用项目的绝对路径：

```ini
[Desktop Entry]
Type=Application
Exec=/home/rm/Desktop/simple_auto_aim/autostart.sh
Name=simple_auto_aim
```

授予启动脚本执行权限：

```bash
chmod +x autostart.sh
```

## 5 USB2CAN 设置（可选）

创建 udev 规则文件：

```bash
sudo touch /etc/udev/rules.d/99-can-up.rules
```

写入：

```udev
ACTION=="add", KERNEL=="can0", RUN+="/sbin/ip link set can0 up type can bitrate 1000000"
ACTION=="add", KERNEL=="can1", RUN+="/sbin/ip link set can1 up type can bitrate 1000000"
```

## 6 GPU 推理（可选）

以下示例安装 Intel GPU 运行时依赖：

```bash
mkdir -p neo
cd neo

wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-core_1.0.13463.18_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-opencl_1.0.13463.18_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu-dbgsym_1.3.25812.14_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu_1.3.25812.14_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd-dbgsym_23.09.25812.14_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd_23.09.25812.14_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/libigdgmm12_22.3.0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/ww09.sum

sha256sum -c ww09.sum
sudo dpkg -i *.deb
```

使用 GPU 异步推理（`async-infer`）时，最高显示分辨率限制为 1920×1080（24 Hz）。

## 7 串口设置

将当前用户加入 `dialout` 组：

```bash
sudo usermod -a -G dialout $USER
```

获取设备信息（将设备名替换为实际端口）：

```bash
udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
```

创建规则文件：

```bash
sudo touch /etc/udev/rules.d/99-usb-serial.rules
```

写入规则，并将示例 ID 替换为实际值：

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="1234", ATTRS{serial}=="A1234567", SYMLINK+="gimbal"
```

重新加载规则并检查软链接：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/gimbal
```

## 8 手眼标定

本机
```powershell
$env:DISPLAY = "127.0.0.1:0.0"
ssh -Y hfut@192.168.xxx.xx
```
进入远程后
```bash
cd ~/simple_auto_aim
./build/capture
```

## 9 buff_v2 R 标检测调试

独立入口为 `src/buff_v2_r_detector_debug.cpp`。它复用 `standard` 的相机、云台反馈、
`BuffConfig`、颜色选择、`FrameRuntime`、`RuneDetector` 和 Foxglove 发布机制。
调试入口运行检测与状态估计，不启动云台规划或发射线程。

```bash
cmake -S . -B build
cmake --build build --target buff_v2_r_detector_debug -j2
./build/buff_v2_r_detector_debug configs/standard.yaml --mode=1 --target-color=blue
```

`--mode=2` 使用大符模型。`--target-color=red` 检测红色，`--target-color=none`
由下位机反馈决定颜色，与主入口相同。检测参数直接读取同一份配置中的相机内参和
`buff_v2` 下的检测参数，无需维护第二份参数。配置中的 `foxglove.enable` 需要开启。

在 Foxglove 中连接 `ws://localhost:8765`，新增 Plot 面板并添加以下路径：

| 数据 | 路径 | 单位 |
| --- | --- | --- |
| 当前帧轮廓半径 | `/buff_v2/detector.candidates[:].radius` | 像素 |
| 靶心半径下限 | `/buff_v2/detector.min_radius` | 像素 |
| 靶心半径上限 | `/buff_v2/detector.max_radius` | 像素 |
| 当前帧轮廓面积 | `/buff_v2/detector.candidates[:].area` | 像素² |
| R 标面积下限 | `/buff_v2/detector.min_icon_area` | 像素² |
| R 标面积上限 | `/buff_v2/detector.max_icon_area` | 像素² |

建议分别建立半径图和面积图。用 `candidates[0].radius` 等路径可以只查看某个编号。
新增 Image 面板选择 `/image`，黄色圆圈及 `#0`、`#1` 等标注对应当前帧的候选数组下标。
检测成功的 R 标复用主链路的绿色圆圈和 `R: 得分` 标注；仅检测到 R 标、尚未建立
完整打符模型时也会显示。`/image_raw` 发布原始图像，`/image` 发布带标注的图像。
编号随每帧轮廓顺序变化，不代表跨帧跟踪 ID。

记录点位于 `RuneDetector::detect` 中计算 `radius` 和 `area` 后、形状与尺寸筛选之前，
因此被过滤的有效轮廓也可观察。`radius_pass` 和 `icon_area_pass` 分别表示尺寸阈值
是否通过，不代表最终识别成功；实际识别还会检查形状、靶心分支及 R 标骨架得分。
半径阈值属于靶心分支，R 标面积分支只在半径范围不匹配时进入。

无候选时数组为空，阈值继续发布。数值数据按处理帧发布，图像受
`foxglove.image_fps` 限制；发布队列沿用主链路的最新帧机制，客户端较慢时可能跳帧。
`standard` 默认不采集这些候选诊断数据。两个入口使用相同端口，应分别运行。
