#include "buff_planner_factory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "buff_planner.hpp"
#include "rune_predictor.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

Timestamp offset_time(Timestamp time, double seconds)
{
  return time + std::chrono::duration_cast<Timestamp::duration>(std::chrono::duration<double>(seconds));
}

struct AimSolution
{
  Eigen::Vector2d angles;
  double fly_time;
  Eigen::Vector3d point;
};

std::optional<AimSolution> aim_solution(const RuneEstimate & state, Timestamp time,
                                        double speed, const BuffConfig::Planner & config,
                                        const tools::BallisticSolver & ballistic_solver)
{
  const auto point = RunePredictor{}.aimpoint_at(state, std::max(time, state.timestamp));
  if (!point) return std::nullopt;
  const auto bullet = ballistic_solver.solve(speed, std::hypot(point->x(), point->y()), point->z());
  if (!bullet) return std::nullopt;
  return AimSolution{{std::remainder(std::atan2(point->y(), point->x()) + config.yaw_offset,
                                     2 * kPi),
                      -bullet->pitch - config.pitch_offset}, bullet->fly_time, *point};
}

class MpcBuffPlanner final : public BuffPlanStrategy
{
public:
  MpcBuffPlanner(const BuffConfig::Planner & config, const std::string & path)
  : planner_(path), buff_planner_(config) {}

  auto_aim::Plan plan(std::uint64_t generation, const std::optional<RuneEstimate> & target,
                      double speed, const io::GimbalState & gimbal, Timestamp now) override
  {
    const auto request = buff_planner_.prepare(generation, target, speed, now);
    if (!request) return {};
    auto result = planner_.plan(request->trajectory, request->yaw0, request->distance);
    result.debug_xyza = {request->aimpoint.x(), request->aimpoint.y(),
                         request->aimpoint.z(), request->yaw0};
    result.fly_time = request->fly_time;
    result.fire = buff_planner_.fire_advice(*request, result, gimbal, now);
    return result;
  }

private:
  auto_aim::Planner planner_;
  BuffPlanner buff_planner_;
};

class NjustBuffPlanner final : public BuffPlanStrategy
{
public:
  explicit NjustBuffPlanner(BuffConfig::Planner config)
  : config_(std::move(config)),
    ballistic_solver_(tools::make_ballistic_solver(config_.ballistic_model,
                                                    config_.ballistic_config)) {}

  auto_aim::Plan plan(std::uint64_t generation, const std::optional<RuneEstimate> & target,
                      double speed, const io::GimbalState & gimbal, Timestamp now) override
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
    AimSolution solution;
    // Match the pre-MPC aimer's five ballistic flight-time refinements.
    for (int i = 0; i < 5; ++i) {
      const auto next = aim_solution(*target, offset_time(now, config_.shoot_delay + fly_time),
                                     speed, config_, *ballistic_solver_);
      if (!next) return result;
      solution = *next;
      if (std::abs(next->fly_time - fly_time) < 0.001) break;
      fly_time = next->fly_time;
    }
    const double future = config_.shoot_delay + fly_time;
    const auto before = aim_solution(*target, offset_time(now, future - 0.01),
                                      speed, config_, *ballistic_solver_);
    const auto after = aim_solution(*target, offset_time(now, future + 0.01),
                                     speed, config_, *ballistic_solver_);
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

    const double cycle = config_.rune_idle_duration + config_.rune_shoot_duration;
    const double phase = std::fmod(std::chrono::duration<double>(now - *attack_start_).count(), cycle);
    const bool shoot_phase = phase >= config_.rune_idle_duration;
    const double xy = std::hypot(solution.point.x(), solution.point.y());
    if (xy < 0.1) return result;
    const double ratio = std::min(xy, 5.0) / xy;
    const Eigen::Vector3d scaled = solution.point * ratio;
    const double scaled_xy = std::hypot(scaled.x(), scaled.y());
    const double center_yaw = std::atan2(target->center.y(), target->center.x());
    const double target_yaw = std::atan2(solution.point.y(), solution.point.x());
    const double blade_scale = std::max(0.0, std::cos(std::remainder(target_yaw - center_yaw, 2 * kPi)));
    const double yaw_window = std::atan2(config_.yaw_tolerance * blade_scale, scaled_xy);
    const double pitch_center = std::atan2(scaled.z(), scaled_xy);
    const double pitch_lower =
      std::atan2(scaled.z() - config_.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
    const double pitch_upper =
      std::atan2(scaled.z() + config_.pitch_tolerance * blade_scale, scaled_xy) - pitch_center;
    const double pitch_error = gimbal.pitch - result.pitch;
    result.fire = shoot_phase &&
                  std::abs(std::remainder(result.yaw - gimbal.yaw, 2 * kPi)) <= yaw_window &&
                  pitch_error >= pitch_lower && pitch_error <= pitch_upper;
    return result;
  }

private:
  BuffConfig::Planner config_;
  std::unique_ptr<tools::BallisticSolver> ballistic_solver_;
  std::optional<Timestamp> attack_start_;
  std::optional<std::uint64_t> attack_generation_;
};
}  // namespace

std::unique_ptr<BuffPlanStrategy> make_buff_planner(
  const BuffConfig::Planner & config, const std::string & config_path)
{
  if (config.mode == "mpc") return std::make_unique<MpcBuffPlanner>(config, config_path);
  if (config.mode == "njust") return std::make_unique<NjustBuffPlanner>(config);
  throw std::invalid_argument("unknown buff_v2 planner mode: " + config.mode);
}
}  // namespace auto_buff_v2
