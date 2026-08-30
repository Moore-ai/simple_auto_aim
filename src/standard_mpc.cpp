#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/aim_factory.hpp"
#include "tools/detect_factory.hpp"
#include "io/camera.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;
  tools::FoxgloveVisualizer foxglove;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);

  auto detect_armors = create_detector_result(config_path);
  auto aim_fn = create_aim_fn(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;

  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty()) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = aim_fn(target, gs.bullet_speed, std::chrono::steady_clock::now());
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc,
          plan.debug_valid ? static_cast<float>(std::hypot(plan.debug_xyza.x(), plan.debug_xyza.y()))
                            : 0.0F);

        std::this_thread::sleep_for(10ms);
      } else
        std::this_thread::sleep_for(200ms);
    }
  });

  while (!exiter.exit()) {
    if (!camera.read(img, t)) break;
    const auto gimbal_state = gimbal.state();
    if (const auto color = io::infantry_enemy_color(gimbal_state.mode)) {
      tracker.set_enemy_color(
        *color == io::InfantryEnemyColor::red ? auto_aim::red : auto_aim::blue);
    }
    auto q = gimbal.q(t);
    solver.set_R_gimbal2world(q);

    auto detections = detect_armors(img, -1);
    auto targets = tracker.track(detections, t);
    if (!targets.empty())
      target_queue.push(targets.front());
    else
      target_queue.push(std::nullopt);

    auto frame = tools::FrameSnapshot::capture(
      t, img, q, gimbal_state, gimbal.command(), std::move(detections), tracker.enemy_color(),
      tracker.debug_data());
    recorder.record(frame);
    foxglove.publish(frame);
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();

  return 0;
}
