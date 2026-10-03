#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "tasks/auto_buff_v2/planners/buff_planner_factory.hpp"

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
  assert(std::abs(mpc_plan.fly_time - 0.176280533350353) < 1e-9);

  config.mode = "njust";
  auto njust = auto_buff_v2::make_buff_planner(config, path);
  const auto njust_plan = njust->plan(1, target, 20, gimbal, start);
  assert(njust_plan.control);
  assert(!njust_plan.debug_valid);
  assert(!njust_plan.fire);
  assert(std::abs(njust_plan.fly_time - 0.176280533350353) < 1e-9);
  gimbal.yaw = njust_plan.yaw;
  gimbal.pitch = njust_plan.pitch;
  assert(njust->plan(1, target, 20, gimbal, start + 450ms).fire);
  assert(!njust->plan(1, std::nullopt, 20, gimbal, start + 460ms).control);
  assert(!njust->plan(2, target, 20, gimbal, start + 470ms).fire);

  // Independent vacuum-ballistic fixtures with 80 ms observation age and 40 ms delay.
  // The fast target reaches the five-solve limit; the sine target converges in two solves.
  for (const auto & mode : {"mpc", "njust"}) {
    config.mode = mode;
    auto moving_planner = auto_buff_v2::make_buff_planner(config, path);
    auto moving_target = target;
    moving_target.rotation_speed = 100;
    const auto fast = moving_planner->plan(3, moving_target, 20, gimbal, start + 80ms);
    assert(fast.control);
    assert(std::abs(fast.fly_time - 0.153672682779469) < 1e-9);
    assert(std::abs(fast.debug_xyza.y() + 0.395697924416826) < 1e-8);
    assert(std::abs(fast.debug_xyza.z() - 0.422571950272403) < 1e-8);

    moving_target.start_timestamp = start - 7s;
    moving_target.sine_valid = true;
    moving_target.sine_v = 1;
    moving_target.sine_a = 0.8;
    moving_target.sine_omega = 2;
    moving_target.sine_phase = 0.2;
    const auto sine = moving_planner->plan(4, moving_target, 20, gimbal, start + 80ms);
    assert(sine.control);
    assert(std::abs(sine.fly_time - 0.175304713611082) < 1e-9);
    assert(std::abs(sine.debug_xyza.y() + 0.276559893734488) < 1e-8);
    assert(std::abs(sine.debug_xyza.z() - 1.643051028439865) < 1e-8);
  }

  config.mode = "unknown";
  bool rejected = false;
  try {
    auto_buff_v2::make_buff_planner(config, path);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}
