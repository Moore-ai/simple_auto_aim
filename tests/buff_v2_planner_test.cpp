#include <cassert>
#include <chrono>
#include <cmath>

#include "tasks/auto_buff_v2/buff_planner.hpp"
#include "tasks/auto_buff_v2/rune_trajectory.hpp"

int main()
{
  using namespace std::chrono_literals;
  const auto start = std::chrono::steady_clock::now();
  auto_buff_v2::RuneState target;
  target.center = {3, 0, 0};
  target.start_timestamp = start - 4s;
  target.timestamp = start;
  target.inactive[0] = true;
  auto_buff_v2::BuffPlanner::Config config;
  config.shoot_delay = 0;
  config.rune_idle_duration = 0.4;
  config.rune_shoot_duration = 0.2;
  config.yaw_tolerance = 0.07;
  config.pitch_tolerance = 0.04;
  auto_buff_v2::BuffPlanner planner(config);
  io::GimbalState gimbal;
  const auto idle = planner.plan(target, 20, gimbal, start);
  assert(idle.control && !idle.fire);
  gimbal.yaw = idle.yaw;
  gimbal.pitch = idle.pitch;
  const auto shoot = planner.plan(target, 20, gimbal, start + 450ms);
  assert(shoot.control && shoot.fire);
  gimbal.yaw = shoot.yaw + 0.05;
  assert(!planner.plan(target, 20, gimbal, start + 470ms).fire);
  gimbal.yaw = shoot.yaw;
  gimbal.pitch = shoot.pitch + 0.03;
  assert(!planner.plan(target, 20, gimbal, start + 480ms).fire);

  target.inactive.fill(false);
  const auto no_blade = planner.plan(target, 20, gimbal, start + 500ms);
  assert(!no_blade.fire);

  target.inactive[0] = true;
  target.start_timestamp = start - 7s;
  target.sine_valid = true;
  target.sine_v = 1;
  target.sine_a = 0.8;
  target.sine_omega = 2;
  target.sine_phase = 0.2;
  const auto accelerating = planner.plan(target, 20, gimbal, start + 510ms);
  assert(accelerating.control);
  assert(std::abs(accelerating.pitch_acc) > 0.01);

  const auto trajectory = auto_buff_v2::solve_rune_trajectory(20, 5, 0);
  assert(trajectory);
  assert(trajectory->fly_time > 0.25);
  assert(trajectory->pitch > 0);
}
