#include "buff_planner.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

BuffPlanner::Config load_config(const std::string & path)
{
  const auto yaml = YAML::LoadFile(path);
  BuffPlanner::Config result;
  const auto buff = yaml["buff_v2"];
  if (buff) {
    result.shoot_delay = buff["shoot_delay"].as<double>(result.shoot_delay);
    result.rune_idle_duration =
      buff["rune_idle_duration"].as<double>(result.rune_idle_duration);
    result.rune_shoot_duration =
      buff["rune_shoot_duration"].as<double>(result.rune_shoot_duration);
    result.yaw_tolerance = buff["yaw_tolerance"].as<double>(result.yaw_tolerance);
    result.pitch_tolerance = buff["pitch_tolerance"].as<double>(result.pitch_tolerance);
  }
  result.ballistic_model = yaml["ballistic_model"].as<std::string>(result.ballistic_model);
  result.ballistic_config.njust_air_resistance = yaml["njust_air_resistance"].as<double>(
    result.ballistic_config.njust_air_resistance);
  result.yaw_offset = yaml["yaw_offset"].as<double>(0) * kPi / 180;
  result.pitch_offset = yaml["pitch_offset"].as<double>(0) * kPi / 180;
  result.bullet_speed_min = yaml["bullet_speed_min"].as<double>(result.bullet_speed_min);
  result.bullet_speed_max = yaml["bullet_speed_max"].as<double>(result.bullet_speed_max);
  result.bullet_speed_default =
    yaml["bullet_speed_default"].as<double>(result.bullet_speed_default);
  return result;
}

std::optional<Eigen::Vector3d> future_point(RuneState state, double future_seconds)
{
  state.transition(future_seconds);
  state.timestamp += std::chrono::duration_cast<Timestamp::duration>(
    std::chrono::duration<double>(future_seconds));
  return state.aimpoint();
}

struct AimSolution
{
  Eigen::Vector2d angles;
  double fly_time;
  Eigen::Vector3d point;
};

std::optional<AimSolution> aim_solution(RuneState state, double seconds, double speed,
                                        double yaw_offset, double pitch_offset,
                                        const tools::BallisticSolver & ballistic_solver)
{
  const auto point = future_point(state, seconds);
  if (!point) return std::nullopt;
  const double distance = std::hypot(point->x(), point->y());
  const auto bullet = ballistic_solver.solve(speed, distance, point->z());
  if (!bullet) return std::nullopt;
  return AimSolution{
    {std::remainder(std::atan2(point->y(), point->x()) + yaw_offset, 2 * kPi),
     -bullet->pitch - pitch_offset},
    bullet->fly_time, *point};
}
}  // namespace

BuffPlanner::BuffPlanner(Config config)
: config_(std::move(config)),
  ballistic_solver_(tools::make_ballistic_solver(config_.ballistic_model, config_.ballistic_config))
{
}
BuffPlanner::BuffPlanner(const std::string & config_path) : BuffPlanner(load_config(config_path)) {}

BuffPlan BuffPlanner::plan(std::uint64_t target_generation, std::optional<RuneState> target,
                           double bullet_speed,
                           const io::GimbalState & gimbal, Timestamp now)
{
  BuffPlan result;
  if (!target) {
    attack_start_.reset();
    attack_generation_.reset();
    return result;
  }
  if (attack_generation_ != target_generation) {
    attack_start_ = now;
    attack_generation_ = target_generation;
  }
  if (bullet_speed < config_.bullet_speed_min || bullet_speed > config_.bullet_speed_max)
    bullet_speed = config_.bullet_speed_default;
  const double stale =
    std::max(0.0, std::chrono::duration<double>(now - target->timestamp).count());
  const double distance = target->center.norm();
  double fly_time = distance / bullet_speed;
  Eigen::Vector2d angles;
  Eigen::Vector3d attack_point;
  for (int i = 0; i < 5; ++i) {
    const double future = stale + config_.shoot_delay + fly_time;
    const auto solution = aim_solution(*target, future, bullet_speed,
                                       config_.yaw_offset, config_.pitch_offset,
                                       *ballistic_solver_);
    if (!solution) return result;
    angles = solution->angles;
    attack_point = solution->point;
    if (std::abs(solution->fly_time - fly_time) < 0.001) break;
    fly_time = solution->fly_time;
  }
  const double future = stale + config_.shoot_delay + fly_time;
  const auto before = aim_solution(*target, future - 0.01, bullet_speed,
                                   config_.yaw_offset, config_.pitch_offset,
                                   *ballistic_solver_);
  const auto after = aim_solution(*target, future + 0.01, bullet_speed,
                                  config_.yaw_offset, config_.pitch_offset,
                                  *ballistic_solver_);
  if (!before || !after) return result;
  result.control = true;
  result.yaw = angles.x();
  result.pitch = angles.y();
  result.yaw_vel = std::remainder(after->angles.x() - before->angles.x(), 2 * kPi) / 0.02;
  result.pitch_vel = (after->angles.y() - before->angles.y()) / 0.02;
  result.yaw_acc = (std::remainder(after->angles.x() - angles.x(), 2 * kPi) -
                    std::remainder(angles.x() - before->angles.x(), 2 * kPi)) / 0.0001;
  result.pitch_acc = (after->angles.y() - 2 * angles.y() + before->angles.y()) / 0.0001;
  result.distance = distance;
  const double cycle = config_.rune_idle_duration + config_.rune_shoot_duration;
  const double phase =
    std::fmod(std::chrono::duration<double>(now - *attack_start_).count(), cycle);
  const bool shoot_phase = phase >= config_.rune_idle_duration;
  const double xy = std::hypot(attack_point.x(), attack_point.y());
  if (xy < 0.1) return result;
  const double ratio = std::min(xy, 5.0) / xy;
  const Eigen::Vector3d scaled = attack_point * ratio;
  const double scaled_xy = std::hypot(scaled.x(), scaled.y());
  const double center_yaw = std::atan2(target->center.y(), target->center.x());
  const double target_yaw = std::atan2(attack_point.y(), attack_point.x());
  const double blade_scale = std::max(0.0, std::cos(std::remainder(
    target_yaw - center_yaw, 2 * kPi)));
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
}  // namespace auto_buff_v2
