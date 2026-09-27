#include <cassert>
#include <chrono>
#include <fstream>
#include <stdexcept>

#include "tasks/auto_buff_v2/buff_planner_factory.hpp"

int main()
{
  using namespace std::chrono_literals;
  constexpr char path[] = "/tmp/buff_v2_planner_factory_test.yaml";
  std::ofstream(path) <<
    "ballistic_model: vacuum\n"
    "yaw_offset: 0\n"
    "pitch_offset: 0\n"
    "fire_thresh: 0.0035\n"
    "decision_speed: 7\n"
    "high_speed_delay_time: 0\n"
    "low_speed_delay_time: 0\n"
    "rho: 1.0\n"
    "max_iter: 10\n"
    "bullet_speed_min: 10\n"
    "bullet_speed_max: 25\n"
    "bullet_speed_default: 23.4\n"
    "max_yaw_acc: 50\n"
    "Q_yaw: [9e6, 0]\n"
    "R_yaw: [1]\n"
    "max_pitch_acc: 100\n"
    "Q_pitch: [9e6, 0]\n"
    "R_pitch: [1]\n";

  const auto start = std::chrono::steady_clock::now();
  auto_buff_v2::RuneEstimate target;
  target.center = {3, 0, 1};
  target.start_timestamp = start - 4s;
  target.timestamp = start;
  target.inactive[0] = true;
  io::GimbalState gimbal;
  auto_buff_v2::BuffConfig::Planner config;
  config.ballistic_model = "vacuum";

  config.mode = "mpc";
  auto mpc = auto_buff_v2::make_buff_planner(config, path);
  const auto mpc_plan = mpc->plan(1, target, 20, gimbal, start);
  assert(mpc_plan.control && mpc_plan.debug_valid);
  assert(!mpc_plan.fire);

  config.mode = "njust";
  auto njust = auto_buff_v2::make_buff_planner(config, path);
  const auto njust_plan = njust->plan(1, target, 20, gimbal, start);
  assert(njust_plan.control);
  assert(!njust_plan.debug_valid);
  assert(!njust_plan.fire);
  gimbal.yaw = njust_plan.yaw;
  gimbal.pitch = njust_plan.pitch;
  assert(njust->plan(1, target, 20, gimbal, start + 450ms).fire);
  assert(!njust->plan(1, std::nullopt, 20, gimbal, start + 460ms).control);
  assert(!njust->plan(2, target, 20, gimbal, start + 470ms).fire);

  config.mode = "unknown";
  bool rejected = false;
  try {
    auto_buff_v2::make_buff_planner(config, path);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}
