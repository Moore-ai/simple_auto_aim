#include "buff_aiming.hpp"

#include <algorithm>
#include <cmath>

#include "../rune_predictor.hpp"

namespace auto_buff_v2::detail
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
}  // namespace

Timestamp offset_time(Timestamp time, double seconds)
{
  return time + std::chrono::duration_cast<Timestamp::duration>(
                   std::chrono::duration<double>(seconds));
}

std::optional<AimSolution> aim_solution(
  const RuneEstimate & state, Timestamp prediction_time, double speed,
  double yaw_offset, double pitch_offset, const tools::BallisticSolver & ballistic_solver)
{
  const auto point = RunePredictor{}.aimpoint_at(state, std::max(prediction_time, state.timestamp));
  if (!point) return std::nullopt;
  const double distance = std::hypot(point->x(), point->y());
  const auto bullet = ballistic_solver.solve(speed, distance, point->z());
  if (!bullet) return std::nullopt;
  return AimSolution{
    {std::remainder(std::atan2(point->y(), point->x()) + yaw_offset, 2 * kPi),
     -bullet->pitch - pitch_offset},
    bullet->fly_time, *point};
}

bool fire_advice(
  const BuffConfig::Planner & config, const Eigen::Vector3d & rune_center,
  const Eigen::Vector3d & aimpoint, const auto_aim::Plan & plan, const io::GimbalState & gimbal,
  Timestamp attack_start, Timestamp now)
{
  const double cycle = config.rune_idle_duration + config.rune_shoot_duration;
  const double phase = std::fmod(std::chrono::duration<double>(now - attack_start).count(), cycle);
  const bool shoot_phase = phase >= config.rune_idle_duration;
  const double xy = std::hypot(aimpoint.x(), aimpoint.y());
  if (xy < 0.1) return false;
  const double ratio = std::min(xy, 5.0) / xy;
  const Eigen::Vector3d scaled = aimpoint * ratio;
  const double scaled_xy = std::hypot(scaled.x(), scaled.y());
  const double center_yaw = std::atan2(rune_center.y(), rune_center.x());
  const double target_yaw = std::atan2(aimpoint.y(), aimpoint.x());
  const double blade_scale =
    std::max(0.0, std::cos(std::remainder(target_yaw - center_yaw, 2 * kPi)));
  const double yaw_upper = std::atan2(config.yaw_tolerance * blade_scale, scaled_xy);
  const double yaw_lower = -yaw_upper;
  const double pitch_center = std::atan2(scaled.z(), scaled_xy);
  const double pitch_lower =
    std::atan2(scaled.z() - config.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
  const double pitch_upper =
    std::atan2(scaled.z() + config.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
  const double yaw_error = std::remainder(gimbal.yaw - plan.yaw, 2 * kPi);
  const double pitch_error = gimbal.pitch - plan.pitch;
  return shoot_phase && yaw_error >= yaw_lower && yaw_error <= yaw_upper &&
         pitch_error >= pitch_lower && pitch_error <= pitch_upper;
}
}  // namespace auto_buff_v2::detail
