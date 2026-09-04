#include <chrono>
#include <cmath>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>
#include <utility>

#include "io/camera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/detect_factory.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/frame_runtime.hpp"
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

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  tools::FoxgloveVisualizer foxglove(solver);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  auto detector = tools::create_detector(config_path);
  tools::FrameRuntime runtime(camera, gimbal, solver, tracker, *detector);

  using PlanRequest = std::pair<std::uint64_t, std::optional<auto_aim::Target>>;
  tools::ThreadSafeQueue<PlanRequest, true> target_queue(1);
  target_queue.push({0, std::nullopt});

  std::atomic<bool> quit = false;

  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty()) {
        const auto [target_generation, target] = target_queue.front();
        auto gs = gimbal.state();
        auto plan = planner.plan(target, gs.bullet_speed);
        foxglove.update_plan(target_generation, plan);
        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc,
          plan.debug_valid ? static_cast<float>(std::hypot(plan.debug_xyza.x(), plan.debug_xyza.y()))
                           : 0.0F);

        std::this_thread::sleep_for(10ms);
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  while (!exiter.exit()) {
    tools::ProcessedFrame processed;
    if (!runtime.next(processed)) break;

    const auto target = processed.targets.empty() ? std::nullopt :
                                                    std::optional(processed.targets.front());
    target_queue.push({processed.snapshot.target_generation, target});

    recorder.record(processed.snapshot);
    foxglove.publish(processed.snapshot);
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();

  return 0;
}
