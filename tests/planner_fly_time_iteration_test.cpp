#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/ballistic_solver.hpp"

namespace
{
tools::BallisticSolution trajectory_at(
  const tools::BallisticSolver & solver, auto_aim::Target target, double fly_time,
  double bullet_speed)
{
  target.predict(fly_time);

  auto min_dist = std::numeric_limits<double>::infinity();
  Eigen::Vector3d xyz;
  for (const auto & xyza : target.armor_xyza_list()) {
    const auto dist = xyza.head<3>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  return solver.solve(bullet_speed, xyz.head<2>().norm(), xyz.z()).value();
}
}  // namespace

int main()
{
  constexpr double bullet_speed = 22.0;
  tools::BallisticSolverConfig njust_config;
  njust_config.njust_air_resistance = 0.006;
  const auto ballistic_solver = tools::make_ballistic_solver("njust", njust_config);
  const auto vacuum_solver = tools::make_ballistic_solver("vacuum");
  auto_aim::Target target(10.0, 1.0, 3.0, 0.0);

  auto_aim::Planner single_pass_planner("tests/planner_fly_time_single_pass.yaml");
  const auto single_pass_plan = single_pass_planner.plan(target, bullet_speed);
  const auto initial_trajectory = trajectory_at(*ballistic_solver, target, 0.0, bullet_speed);
  assert(single_pass_plan.control);
  assert(std::abs(single_pass_plan.fly_time - initial_trajectory.fly_time) < 1e-12);

  auto_aim::Planner planner("tests/planner_fly_time_iteration.yaml");
  const auto plan = planner.plan(target, bullet_speed);
  assert(plan.control);
  const auto first_iteration =
    trajectory_at(*ballistic_solver, target, initial_trajectory.fly_time, bullet_speed);
  assert(std::abs(plan.fly_time - first_iteration.fly_time) < 1e-12);

  auto_aim::Planner threshold_planner("tests/planner_fly_time_threshold.yaml");
  const auto threshold_plan = threshold_planner.plan(target, bullet_speed);
  const auto vacuum_initial = trajectory_at(*vacuum_solver, target, 0.0, bullet_speed);
  const auto vacuum_first_iteration =
    trajectory_at(*vacuum_solver, target, vacuum_initial.fly_time, bullet_speed);
  assert(threshold_plan.control);
  assert(std::abs(threshold_plan.fly_time - vacuum_first_iteration.fly_time) < 1e-12);
  assert(threshold_plan.fly_time < plan.fly_time);

  // At this phase the elevated armor is closer in XY (9 m), but the
  // zero-height armor is closer in 3D (sqrt(101) m versus sqrt(117) m).
  for (const auto height : {6.0, -6.0}) {
    auto_aim::Target elevated_target(10.0, 1.0, 1.0, height);
    elevated_target.predict(M_PI_2);
    const auto expected = ballistic_solver->solve(bullet_speed, std::sqrt(101.0), 0.0).value();
    const auto elevated_plan = single_pass_planner.plan(elevated_target, bullet_speed);
    assert(elevated_plan.control);
    assert(std::abs(elevated_plan.fly_time - expected.fly_time) < 1e-12);

    const auto expected_iteration =
      trajectory_at(*ballistic_solver, elevated_target, expected.fly_time, bullet_speed);
    const auto elevated_iteration_plan = planner.plan(elevated_target, bullet_speed);
    assert(elevated_iteration_plan.control);
    assert(std::abs(elevated_iteration_plan.fly_time - expected_iteration.fly_time) < 1e-12);
  }
  // Climber drag comes from the auto-aim root configuration, independently of buff_v2.
  constexpr char climber_path[] = "/tmp/auto_aim_climber_ballistic_test.yaml";
  auto climber_yaml = YAML::LoadFile("tests/planner_fly_time_single_pass.yaml");
  climber_yaml["ballistic_model"] = "climber";
  climber_yaml["climber_air_resistance_k"] = 0.0;
  climber_yaml["buff_v2"]["aimer"]["ballistic"]["climber"]["air_resistance_k"] = 10;
  std::ofstream(climber_path) << climber_yaml;
  const auto vacuum_climber = auto_aim::Planner(climber_path).plan(target, bullet_speed);
  climber_yaml["climber_air_resistance_k"] = 0.02;
  std::ofstream(climber_path) << climber_yaml;
  const auto drag_climber = auto_aim::Planner(climber_path).plan(target, bullet_speed);
  assert(vacuum_climber.control && drag_climber.control);
  assert(drag_climber.fly_time > vacuum_climber.fly_time);
  return 0;
}
