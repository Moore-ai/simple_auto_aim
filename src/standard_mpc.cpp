#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/aim_factory.hpp"
#include "tools/detect_factory.hpp"
#include "io/camera.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/web_debug.hpp"

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
  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::WebDebug web_debug;

  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);

  auto detect_armors = create_detector_result(config_path);
  auto aim_fn = create_aim_fn(config_path);

  tools::ThreadSafeQueue<std::optional<auto_aim::Target>, true> target_queue(1);
  target_queue.push(std::nullopt);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  std::atomic<bool> quit = false;

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};
  std::mutex plan_mutex;
  auto_aim::Plan latest_plan{};
  uint64_t frame_id = 0;
  const auto start_time = std::chrono::steady_clock::now();

  auto plan_thread = std::thread([&]() {
    auto t0 = std::chrono::steady_clock::now();
    uint16_t last_bullet_count = 0;

    while (!quit) {
      if (!target_queue.empty() && mode == io::GimbalMode::AUTO_AIM) {
        auto target = target_queue.front();
        auto gs = gimbal.state();
        auto plan = aim_fn(target, gs.bullet_speed, std::chrono::steady_clock::now());
        {
          std::lock_guard<std::mutex> lock(plan_mutex);
          latest_plan = plan;
        }

        gimbal.send(
          plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
          plan.pitch_acc);

        std::this_thread::sleep_for(10ms);
      } else
        std::this_thread::sleep_for(200ms);
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    camera.read(img, t);
    auto q = gimbal.q(t);
    auto gs = gimbal.state();
    recorder.record(img, q, t);
    solver.set_R_gimbal2world(q);

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto detections = detect_armors(img, -1);
      auto targets = tracker.track(detections, t);
      if (!targets.empty())
        target_queue.push(targets.front());
      else
        target_queue.push(std::nullopt);

      auto plan = auto_aim::Plan{};
      {
        std::lock_guard<std::mutex> lock(plan_mutex);
        plan = latest_plan;
      }

      tools::WebDebugContext context;
      context.frame_id = frame_id++;
      context.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
      context.latency_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
      context.mode = mode.load();
      context.gimbal = gimbal.state();
      context.armors = {detections.armors.begin(), detections.armors.end()};
      context.lightbars = {detections.lightbars.begin(), detections.lightbars.end()};
      context.plan = plan;

      if (!targets.empty()) {
        const auto & target = targets.front();
        context.target = target;
        for (const auto & xyza : target.armor_xyza_list()) {
          context.projected_target_armors.push_back(solver.reproject_armor(
            xyza.head<3>(), xyza[3], target.armor_type, target.name));
        }

        if (plan.debug_valid) {
          context.projected_aim_armor = solver.reproject_armor(
            plan.debug_xyza.head<3>(), plan.debug_xyza[3], target.armor_type, target.name);
        }

        const auto target_state = target.state();
        const auto state = target_state.vector();
        nlohmann::json tracker_state = nlohmann::json::array();
        for (Eigen::Index i = 0; i < state.size(); ++i) tracker_state.push_back(state[i]);
        const auto & tracker_debug = tracker.debug_info();
        context.tracker_json = {
          {"last_id", target.last_id},
          {"jumped", target.jumped},
          {"state", tracker_state},
          {"observation_mode", tracker_debug.observation_mode},
          {"lightbar_assist_enabled", tracker_debug.lightbar_assist_enabled},
          {"yaw_refinement_enabled", tracker_debug.yaw_refinement_enabled},
          {"last_update_source", tracker_debug.last_update_source},
          {"matched_armor_count", tracker_debug.matched_armor_count},
          {"matched_light_count", tracker_debug.matched_light_count},
          {"uvl_observation_count", tracker_debug.uvl_observation_count},
          {"diff_observation_count", tracker_debug.diff_observation_count},
          {"rejected_armor_count", tracker_debug.rejected_armor_count},
          {"rejected_light_count", tracker_debug.rejected_light_count},
          {"pnp_fallback_count", tracker_debug.pnp_fallback_count},
          {"predict_only_count", tracker_debug.predict_only_count},
          {"lightbar_assist_update_count", tracker_debug.lightbar_assist_update_count},
          {"lightbar_assist_failed_count", tracker_debug.lightbar_assist_failed_count},
          {"last_match_cost", tracker_debug.last_match_cost},
          {"last_nis", tracker_debug.last_nis}};
        context.target_velocity_norm = std::hypot(
          target_state.velocity_x(), target_state.velocity_y(), target_state.velocity_z());
        context.target_yaw_rate = target_state.yaw_rate();
        const cv::Point3f center(
          target_state.center_x(), target_state.center_y(), target_state.center_z());
        const auto velocity_points = solver.world2pixel({
          center,
          {static_cast<float>(target_state.center_x() + target_state.velocity_x()),
           static_cast<float>(target_state.center_y() + target_state.velocity_y()),
           static_cast<float>(target_state.center_z() + target_state.velocity_z())},
        });
        if (velocity_points.size() == 2)
          context.target_velocity_arrow = std::make_pair(velocity_points[0], velocity_points[1]);

        const auto yaw_radius_points = solver.world2pixel({
          center,
          {static_cast<float>(target_state.center_x() +
             target_state.radius() * std::cos(target_state.yaw())),
           static_cast<float>(target_state.center_y() +
             target_state.radius() * std::sin(target_state.yaw())),
           static_cast<float>(target_state.center_z())},
        });
        if (yaw_radius_points.size() == 2)
          context.target_yaw_rate_arrow = std::make_pair(yaw_radius_points[0], yaw_radius_points[1]);
      }

      web_debug.publish(context, img);
    }

    /// 打符
    else if (mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);

      buff_solver.solve(power_runes);

      auto_aim::Plan buff_plan;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_plan = buff_aimer.mpc_aim(target_copy, t, gs, true);
      }
      gimbal.send(
        buff_plan.control, buff_plan.fire, buff_plan.yaw, buff_plan.yaw_vel, buff_plan.yaw_acc,
        buff_plan.pitch, buff_plan.pitch_vel, buff_plan.pitch_acc);

    } else
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  }

  quit = true;
  if (plan_thread.joinable()) plan_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
