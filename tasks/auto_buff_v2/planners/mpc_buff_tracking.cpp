#include "mpc_buff_tracking.hpp"

#include <array>
#include <cmath>
#include <utility>

#include "buff_aiming.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

std::optional<auto_aim::Trajectory> make_reference_trajectory(
  const RuneEstimate & state, Timestamp center_time, double speed, double yaw0,
  double yaw_offset, double pitch_offset, const tools::BallisticSolver & ballistic_solver)
{
  std::array<detail::AimSolution, auto_aim::HORIZON + 2> samples;
  for (int i = 0; i <= auto_aim::HORIZON + 1; ++i) {
    const auto time = detail::offset_time(
      center_time, (static_cast<double>(i - 1 - auto_aim::HALF_HORIZON)) * auto_aim::DT);
    const auto solution = detail::aim_solution(
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

MpcBuffTracking::MpcBuffTracking(Config config)
: config_(std::move(config)),
  ballistic_solver_(tools::make_ballistic_solver(config_.ballistic_model, config_.ballistic_config))
{
}

std::optional<MpcBuffTrackingRequest> MpcBuffTracking::prepare(
  std::uint64_t target_generation, const std::optional<RuneEstimate> & target,
  double bullet_speed, Timestamp now)
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
  double fly_time = target->center.norm() / bullet_speed;
  Timestamp center_time;
  std::optional<detail::AimSolution> center;
  // RMCS fire control: always refine from the original estimate, at most five solves.
  for (int i = 0; i < 5; ++i) {
    center_time = detail::offset_time(now, config_.shoot_delay + fly_time);
    center = detail::aim_solution(
      *target, center_time, bullet_speed, config_.yaw_offset, config_.pitch_offset,
      *ballistic_solver_);
    if (!center) return std::nullopt;
    const double previous = fly_time;
    fly_time = center->fly_time;
    if (std::abs(fly_time - previous) < 0.001) break;
  }
  const auto trajectory = make_reference_trajectory(
    *target, center_time, bullet_speed, center->angles.x(), config_.yaw_offset,
    config_.pitch_offset, *ballistic_solver_);
  if (!trajectory) return std::nullopt;

  MpcBuffTrackingRequest request;
  request.trajectory = *trajectory;
  request.yaw0 = center->angles.x();
  request.distance = distance;
  request.fly_time = fly_time;
  request.rune_center = target->center;
  request.aimpoint = center->point;
  return request;
}

bool MpcBuffTracking::fire_advice(
  const MpcBuffTrackingRequest & request, const auto_aim::Plan & plan,
  const io::GimbalState & gimbal, Timestamp now) const
{
  if (!plan.control || !attack_start_) return false;
  return detail::fire_advice(
    config_, request.rune_center, request.aimpoint, plan, gimbal, *attack_start_, now);
}
}  // namespace auto_buff_v2
