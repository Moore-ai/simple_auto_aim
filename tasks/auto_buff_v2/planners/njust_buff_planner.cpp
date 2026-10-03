#include "njust_buff_planner.hpp"

#include <cmath>
#include <utility>

#include "buff_aiming.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
}  // namespace

NjustBuffPlanner::NjustBuffPlanner(BuffConfig::Planner config)
: config_(std::move(config)),
  ballistic_solver_(tools::make_ballistic_solver(config_.ballistic_model, config_.ballistic_config))
{
}

auto_aim::Plan NjustBuffPlanner::plan(
  std::uint64_t generation, const std::optional<RuneEstimate> & target, double speed,
  const io::GimbalState & gimbal, Timestamp now)
{
  auto_aim::Plan result;
  if (!target) {
    attack_start_.reset();
    attack_generation_.reset();
    return result;
  }
  if (attack_generation_ != generation) {
    attack_start_ = now;
    attack_generation_ = generation;
  }
  if (speed < config_.bullet_speed_min || speed > config_.bullet_speed_max)
    speed = config_.bullet_speed_default;
  const double distance = target->center.norm();
  double fly_time = distance / speed;
  detail::AimSolution solution;
  // RMCS fire control: always refine from the original estimate, at most five solves.
  for (int i = 0; i < 5; ++i) {
    const auto next = detail::aim_solution(
      *target, detail::offset_time(now, config_.shoot_delay + fly_time), speed,
      config_.yaw_offset, config_.pitch_offset, *ballistic_solver_);
    if (!next) return result;
    solution = *next;
    const double previous = fly_time;
    fly_time = next->fly_time;
    if (std::abs(fly_time - previous) < 0.001) break;
  }
  const double future = config_.shoot_delay + fly_time;
  const auto before = detail::aim_solution(
    *target, detail::offset_time(now, future - 0.01), speed,
    config_.yaw_offset, config_.pitch_offset, *ballistic_solver_);
  const auto after = detail::aim_solution(
    *target, detail::offset_time(now, future + 0.01), speed,
    config_.yaw_offset, config_.pitch_offset, *ballistic_solver_);
  if (!before || !after) return result;
  result.control = true;
  result.yaw = solution.angles.x();
  result.pitch = solution.angles.y();
  result.yaw_vel = std::remainder(after->angles.x() - before->angles.x(), 2 * kPi) / 0.02;
  result.pitch_vel = (after->angles.y() - before->angles.y()) / 0.02;
  result.yaw_acc = (std::remainder(after->angles.x() - solution.angles.x(), 2 * kPi) -
                   std::remainder(solution.angles.x() - before->angles.x(), 2 * kPi)) / 0.0001;
  result.pitch_acc = (after->angles.y() - 2 * solution.angles.y() + before->angles.y()) / 0.0001;
  result.distance = distance;
  result.fly_time = fly_time;
  result.debug_xyza = {solution.point.x(), solution.point.y(), solution.point.z(), result.yaw};

  result.fire = detail::fire_advice(
    config_, target->center, solution.point, result, gimbal, *attack_start_, now);
  return result;
}
}  // namespace auto_buff_v2
