#ifndef AUTO_BUFF_V2__BUFF_AIMING_HPP
#define AUTO_BUFF_V2__BUFF_AIMING_HPP

#include <optional>

#include "../buff_config.hpp"
#include "../rune_model.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/ballistic_solver.hpp"

namespace auto_buff_v2::detail
{
struct AimSolution
{
  Eigen::Vector2d angles;
  double fly_time;
  Eigen::Vector3d point;
};

Timestamp offset_time(Timestamp time, double seconds);
std::optional<AimSolution> aim_solution(
  const RuneEstimate & state, Timestamp prediction_time, double speed,
  double yaw_offset, double pitch_offset, const tools::BallisticSolver & ballistic_solver);
bool fire_advice(
  const BuffConfig::Planner & config, const Eigen::Vector3d & rune_center,
  const Eigen::Vector3d & aimpoint, const auto_aim::Plan & plan, const io::GimbalState & gimbal,
  Timestamp attack_start, Timestamp now);
}  // namespace auto_buff_v2::detail

#endif  // AUTO_BUFF_V2__BUFF_AIMING_HPP
