#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>
#include <utility>

#include "io/camera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_buff_v2/buff_planner.hpp"
#include "tasks/auto_buff_v2/frame_runtime.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }"
  "{mode           |0| 0=小符，1=大符 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  const auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const int mode = cli.get<int>("mode");
  if (!cli.check() || (mode != 0 && mode != 1)) {
    cli.printErrors();
    std::cerr << "mode 必须为 0（小符）或 1（大符）\n";
    return 2;
  }

  tools::Exiter exiter;
  tools::Recorder recorder;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::Solver solver(config_path);
  tools::FoxgloveVisualizer foxglove(solver, config_path);
  auto_buff_v2::RuneModel model(config_path, mode == 1);
  auto_buff_v2::BuffPlanner planner(config_path);
  auto_buff_v2::FrameRuntime runtime(camera, gimbal, model, config_path);

  using PlanRequest = std::pair<std::uint64_t, std::optional<auto_buff_v2::RuneState>>;
  tools::ThreadSafeQueue<PlanRequest, true> target_queue(1);
  target_queue.push({0, std::nullopt});
  std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty()) {
        const auto [generation, target] = target_queue.front();
        const auto gs = gimbal.state();
        const auto plan = planner.plan(target, gs.bullet_speed, gs,
                                       std::chrono::steady_clock::now());
        auto_aim::Plan visual_plan;
        visual_plan.control = plan.control;
        visual_plan.fire = plan.fire;
        visual_plan.yaw = plan.yaw;
        visual_plan.yaw_vel = plan.yaw_vel;
        visual_plan.yaw_acc = plan.yaw_acc;
        visual_plan.pitch = plan.pitch;
        visual_plan.pitch_vel = plan.pitch_vel;
        visual_plan.pitch_acc = plan.pitch_acc;
        visual_plan.distance = plan.distance;
        foxglove.update_plan(generation, visual_plan);
        gimbal.send(plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc,
                    plan.pitch, plan.pitch_vel, plan.pitch_acc, plan.distance);
        std::this_thread::sleep_for(1ms);
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  while (!exiter.exit()) {
    auto_buff_v2::ProcessedBuffFrame processed;
    if (!runtime.next(processed)) break;
    target_queue.push({processed.snapshot.target_generation, processed.target});
    recorder.record(processed.snapshot);
    foxglove.publish(std::move(processed.snapshot));
  }
  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  return 0;
}
