#include <cassert>
#include <chrono>

#include "tasks/auto_buff_v2/buff_planner.hpp"

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
  config.yaw_tolerance = 1;
  config.pitch_tolerance = 1;
  auto_buff_v2::BuffPlanner planner(config);
  io::GimbalState gimbal;
  const auto idle = planner.plan(target, 20, gimbal, start);
  assert(idle.control && !idle.fire);
  const auto shoot = planner.plan(target, 20, gimbal, start + 450ms);
  assert(shoot.control && shoot.fire);

  target.inactive.fill(false);
  const auto no_blade = planner.plan(target, 20, gimbal, start + 500ms);
  assert(!no_blade.fire);
}
