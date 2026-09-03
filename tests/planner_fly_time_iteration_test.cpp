#include <cassert>
#include <cmath>
#include <limits>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/trajectory.hpp"

namespace
{
tools::Trajectory trajectory_at(auto_aim::Target target, double fly_time, double bullet_speed)
{
  target.predict(fly_time);

  auto min_dist = std::numeric_limits<double>::infinity();
  Eigen::Vector3d xyz;
  for (const auto & xyza : target.armor_xyza_list()) {
    const auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  return {bullet_speed, min_dist, xyz.z()};
}
}  // namespace

int main()
{
  constexpr double bullet_speed = 22.0;
  auto_aim::Target target(10.0, 1.0, 3.0, 0.0);

  auto_aim::Planner single_pass_planner("configs/standard.yaml");
  const auto single_pass_plan = single_pass_planner.plan(target, bullet_speed);
  const auto initial_trajectory = trajectory_at(target, 0.0, bullet_speed);
  assert(single_pass_plan.control);
  assert(std::abs(single_pass_plan.fly_time - initial_trajectory.fly_time) < 1e-12);

  auto_aim::Planner planner("tests/planner_fly_time_iteration.yaml");
  const auto plan = planner.plan(target, bullet_speed);
  assert(plan.control);
  const auto first_iteration = trajectory_at(target, initial_trajectory.fly_time, bullet_speed);
  assert(std::abs(plan.fly_time - first_iteration.fly_time) < 1e-12);

  auto_aim::Planner threshold_planner("tests/planner_fly_time_threshold.yaml");
  const auto threshold_plan = threshold_planner.plan(target, bullet_speed);
  assert(threshold_plan.control);
  assert(std::abs(threshold_plan.fly_time - first_iteration.fly_time) < 1e-12);
  return 0;
}
