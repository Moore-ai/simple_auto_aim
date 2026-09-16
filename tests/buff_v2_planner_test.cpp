#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "tasks/auto_buff_v2/buff_planner.hpp"
#include "tools/ballistic_solver.hpp"

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

  const auto njust = tools::make_ballistic_solver("njust");
  const auto vacuum = tools::make_ballistic_solver("vacuum");
  const auto njust_solution = njust->solve(20, 5, 0);
  const auto vacuum_solution = vacuum->solve(20, 5, 0);
  assert(njust_solution);
  assert(vacuum_solution);
  assert(njust_solution->fly_time > vacuum_solution->fly_time);
  assert(njust_solution->pitch > vacuum_solution->pitch);

  tools::BallisticSolverConfig low_drag_config;
  low_drag_config.njust_air_resistance = 0.001;
  tools::BallisticSolverConfig high_drag_config;
  high_drag_config.njust_air_resistance = 0.006;
  const auto low_drag = tools::make_ballistic_solver("njust", low_drag_config);
  const auto high_drag = tools::make_ballistic_solver("njust", high_drag_config);
  const auto low_drag_solution = low_drag->solve(20, 5, 0);
  const auto high_drag_solution = high_drag->solve(20, 5, 0);
  assert(low_drag_solution && high_drag_solution);
  assert(high_drag_solution->fly_time > low_drag_solution->fly_time);
  assert(high_drag_solution->pitch > low_drag_solution->pitch);

  const char * njust_config_path = "/tmp/buff_v2_njust_ballistic_test.yaml";
  const char * high_drag_config_path = "/tmp/buff_v2_high_drag_ballistic_test.yaml";
  const char * vacuum_config_path = "/tmp/buff_v2_vacuum_ballistic_test.yaml";
  std::ofstream(njust_config_path) <<
    "ballistic_model: njust\nnjust_air_resistance: 0.001\n";
  std::ofstream(high_drag_config_path) <<
    "ballistic_model: njust\nnjust_air_resistance: 0.006\n";
  std::ofstream(vacuum_config_path) << "ballistic_model: vacuum\n";
  auto_buff_v2::BuffPlanner njust_planner(njust_config_path);
  auto_buff_v2::BuffPlanner high_drag_planner(high_drag_config_path);
  auto_buff_v2::BuffPlanner vacuum_planner(vacuum_config_path);
  const auto njust_plan = njust_planner.plan(target, 20, io::GimbalState{}, start);
  const auto high_drag_plan = high_drag_planner.plan(target, 20, io::GimbalState{}, start);
  const auto vacuum_plan = vacuum_planner.plan(target, 20, io::GimbalState{}, start);
  assert(njust_plan.control && high_drag_plan.control && vacuum_plan.control);
  assert(high_drag_plan.pitch < njust_plan.pitch);
  assert(njust_plan.pitch < vacuum_plan.pitch);

  bool invalid_model_rejected = false;
  try {
    static_cast<void>(tools::make_ballistic_solver("invalid"));
  } catch (const std::invalid_argument &) {
    invalid_model_rejected = true;
  }
  assert(invalid_model_rejected);
}
