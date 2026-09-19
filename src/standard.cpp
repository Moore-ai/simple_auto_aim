#include <chrono>
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>
#include <utility>
#include <variant>

#include "io/camera.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff_v2/buff_planner.hpp"
#include "tasks/auto_buff_v2/frame_runtime.hpp"
#include "tools/detect_factory.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/frame_runtime.hpp"
#include "tools/recorder.hpp"

namespace standard
{
enum class Mode
{
  small_buff = 0,
  big_buff = 1,
  auto_aim = 2
};
}  // namespace standard

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }"
  "{mode           |2| 0=小符，1=大符，2=自瞄普通目标和前哨站 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const int mode_value = cli.get<int>("mode");
  if (!cli.check() || (mode_value < 0 || mode_value > 2)) {
    cli.printErrors();
    std::cerr << "mode 必须为 0（小符）、1（大符）或 2（自瞄）\n";
    return 2;
  }
  const auto mode = static_cast<standard::Mode>(mode_value);

  tools::Exiter exiter;
  tools::Recorder recorder;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  tools::FoxgloveVisualizer foxglove(solver, config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  auto detector = tools::create_detector(config_path);
  tools::FrameRuntime runtime(camera, gimbal, solver, tracker, *detector);

  auto_buff_v2::RuneModel buff_model(config_path, mode == standard::Mode::big_buff);
  auto_buff_v2::BuffPlanner buff_planner(config_path);
  auto_buff_v2::FrameRuntime buff_runtime(camera, gimbal, buff_model, config_path);

  using AutoAimTarget = std::optional<auto_aim::Target>;
  using BuffTarget = std::optional<auto_buff_v2::RuneState>;
  using PlanTarget = std::variant<AutoAimTarget, BuffTarget>;
  using PlanRequest = std::pair<std::uint64_t, PlanTarget>;
  tools::ThreadSafeQueue<PlanRequest, true> target_queue(1);
  if (mode == standard::Mode::auto_aim)
    target_queue.push({0, AutoAimTarget{}});
  else
    target_queue.push({0, BuffTarget{}});

  std::atomic<bool> quit = false;
  auto plan_thread = std::thread([&]() {
    while (!quit) {
      if (!target_queue.empty()) {
        const auto [target_generation, target] = target_queue.front();
        const auto gs = gimbal.state();
        auto_aim::Plan plan;
        auto fire_command = io::InfantryFireCommand::none;

        if (const auto auto_aim_target = std::get_if<AutoAimTarget>(&target)) {
          plan = planner.plan(*auto_aim_target, gs.bullet_speed);
          if (plan.fire) fire_command = io::InfantryFireCommand::continuous;
        } else {
          const auto & buff_target = std::get<BuffTarget>(target);
          const auto now = std::chrono::steady_clock::now();
          const auto request =
            buff_planner.prepare(target_generation, buff_target, gs.bullet_speed, now);
          if (request) {
            plan = planner.plan(request->trajectory, request->yaw0, request->distance);
            plan.debug_xyza = {
              request->aimpoint.x(), request->aimpoint.y(), request->aimpoint.z(), request->yaw0};
            plan.fly_time = request->fly_time;
            plan.fire = buff_planner.fire_advice(*request, plan, gs, now);
          }
          if (plan.fire) fire_command = io::InfantryFireCommand::single;
        }

        foxglove.update_plan(target_generation, plan);
        gimbal.send(
          plan.control, fire_command, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch,
          plan.pitch_vel, plan.pitch_acc, plan.distance);

        std::this_thread::sleep_for(1ms);
      } else {
        std::this_thread::sleep_for(200ms);
      }
    }
  });

  while (!exiter.exit()) {
    tools::ProcessedFrame processed;
    if (mode == standard::Mode::auto_aim) {
      if (!runtime.next(processed)) break;
      AutoAimTarget target;
      if (!processed.targets.empty()) target = processed.targets.front();
      target_queue.push({processed.snapshot.target_generation, PlanTarget{std::move(target)}});
    } else {
      if (!buff_runtime.next(processed)) break;
      target_queue.push(
        {processed.snapshot.target_generation, PlanTarget{std::move(processed.buff_target)}});
    }

    recorder.record(processed.snapshot);
    foxglove.publish(std::move(processed.snapshot));
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();

  return 0;
}
