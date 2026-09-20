#include "buff_planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "rune_predictor.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

struct AimSolution
{
  Eigen::Vector2d angles;
  double fly_time;
  Eigen::Vector3d point;
};

Timestamp offset_time(Timestamp time, double seconds)
{
  return time + std::chrono::duration_cast<Timestamp::duration>(
                   std::chrono::duration<double>(seconds));
}

std::optional<AimSolution> aim_solution(const RuneEstimate & state, Timestamp prediction_time,
                                        double speed,
                                        double yaw_offset, double pitch_offset,
                                        const tools::BallisticSolver & ballistic_solver)
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

std::optional<auto_aim::Trajectory> make_reference_trajectory(
  const RuneEstimate & state, Timestamp center_time, double speed, double yaw0,
  double yaw_offset, double pitch_offset, const tools::BallisticSolver & ballistic_solver)
{
  std::array<AimSolution, auto_aim::HORIZON + 2> samples;
  for (int i = 0; i <= auto_aim::HORIZON + 1; ++i) {
    const auto time = offset_time(
      center_time,
      (static_cast<double>(i - 1 - auto_aim::HALF_HORIZON)) * auto_aim::DT);
    const auto solution = aim_solution(
      state, time, speed, yaw_offset, pitch_offset, ballistic_solver);
    if (!solution) return std::nullopt;
    samples[i] = *solution;
  }

  auto_aim::Trajectory result;
  for (int i = 0; i < auto_aim::HORIZON; ++i) {
    const auto & before = samples[i];
    const auto & center = samples[i + 1];
    const auto & after = samples[i + 2];
    result.col(i) << std::remainder(center.angles.x() - yaw0, 2 * kPi),
      std::remainder(after.angles.x() - before.angles.x(), 2 * kPi) / (2 * auto_aim::DT),
      center.angles.y(), (after.angles.y() - before.angles.y()) / (2 * auto_aim::DT);
  }
  return result;
}
}  // namespace

BuffPlanner::BuffPlanner(Config config)
: config_(std::move(config)),
  ballistic_solver_(tools::make_ballistic_solver(config_.ballistic_model, config_.ballistic_config))
{
}

std::optional<BuffTrackingRequest> BuffPlanner::prepare(
  std::uint64_t target_generation, const std::optional<RuneEstimate> & target,
  double bullet_speed,
  Timestamp now)
{
  if (!target) {
    attack_start_.reset();
    attack_generation_.reset();
    return std::nullopt;
  }
  if (attack_generation_ != target_generation) {
    attack_start_ = now;
    attack_generation_ = target_generation;
  }
  if (bullet_speed < config_.bullet_speed_min || bullet_speed > config_.bullet_speed_max)
    bullet_speed = config_.bullet_speed_default;
  const double distance = std::hypot(target->center.x(), target->center.y());
  double fly_time = distance / bullet_speed;
  if (config_.fly_time_iteration_enabled) {
    for (int i = 0; i < config_.fly_time_iteration_max_iteration; ++i) {
      const double future = config_.shoot_delay + fly_time;
      const auto prediction_time = offset_time(now, future);
      const auto solution = aim_solution(*target, prediction_time, bullet_speed,
                                         config_.yaw_offset, config_.pitch_offset,
                                         *ballistic_solver_);
      if (!solution) return std::nullopt;
      if (std::abs(solution->fly_time - fly_time) <
          config_.fly_time_iteration_convergence_threshold) {
        break;
      }
      fly_time = solution->fly_time;
    }
  }
  const auto center_time = offset_time(now, config_.shoot_delay + fly_time);
  const auto center = aim_solution(*target, center_time, bullet_speed,
                                   config_.yaw_offset, config_.pitch_offset,
                                   *ballistic_solver_);
  if (!center) return std::nullopt;
  const auto trajectory = make_reference_trajectory(
    *target, center_time, bullet_speed, center->angles.x(), config_.yaw_offset,
    config_.pitch_offset, *ballistic_solver_);
  if (!trajectory) return std::nullopt;

  BuffTrackingRequest request;
  request.trajectory = *trajectory;
  request.yaw0 = center->angles.x();
  request.distance = distance;
  request.fly_time = fly_time;
  request.rune_center = target->center;
  request.aimpoint = center->point;
  return request;
}

bool BuffPlanner::fire_advice(const BuffTrackingRequest & request, const auto_aim::Plan & plan,
                              const io::GimbalState & gimbal, Timestamp now) const
{
  if (!plan.control || !attack_start_) return false;
  const double cycle = config_.rune_idle_duration + config_.rune_shoot_duration;
  const double phase =
    std::fmod(std::chrono::duration<double>(now - *attack_start_).count(), cycle);
  const bool shoot_phase = phase >= config_.rune_idle_duration;
  const double xy = std::hypot(request.aimpoint.x(), request.aimpoint.y());
  if (xy < 0.1) return false;
  const double ratio = std::min(xy, 5.0) / xy;
  const Eigen::Vector3d scaled = request.aimpoint * ratio;
  const double scaled_xy = std::hypot(scaled.x(), scaled.y());
  const double center_yaw = std::atan2(request.rune_center.y(), request.rune_center.x());
  const double target_yaw = std::atan2(request.aimpoint.y(), request.aimpoint.x());
  const double blade_scale = std::max(0.0, std::cos(std::remainder(
    target_yaw - center_yaw, 2 * kPi)));
  const double yaw_window = std::atan2(config_.yaw_tolerance * blade_scale, scaled_xy);
  const double pitch_center = std::atan2(scaled.z(), scaled_xy);
  const double pitch_lower =
    std::atan2(scaled.z() - config_.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
  const double pitch_upper =
    std::atan2(scaled.z() + config_.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
  const double pitch_error = gimbal.pitch - plan.pitch;
  return shoot_phase &&
         std::abs(std::remainder(plan.yaw - gimbal.yaw, 2 * kPi)) <= yaw_window &&
         pitch_error >= pitch_lower && pitch_error <= pitch_upper;
}
}  // namespace auto_buff_v2
