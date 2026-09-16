#include "buff_planner.hpp"

#include <algorithm>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/trajectory.hpp"

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

std::optional<Eigen::Vector2d> aim_angles(RuneState state, double seconds, double speed,
                                          double yaw_offset, double pitch_offset)
{
  const auto point = future_point(state, seconds);
  if (!point) return std::nullopt;
  const double distance = std::hypot(point->x(), point->y());
  tools::Trajectory bullet(speed, distance, point->z());
  if (bullet.unsolvable) return std::nullopt;
  return Eigen::Vector2d(
    std::remainder(std::atan2(point->y(), point->x()) + yaw_offset, 2 * kPi),
    -bullet.pitch - pitch_offset);
}
}  // namespace

BuffPlanner::BuffPlanner(Config config) : config_(config) {}
BuffPlanner::BuffPlanner(const std::string & config_path) : config_(load_config(config_path)) {}

BuffPlan BuffPlanner::plan(std::optional<RuneState> target, double bullet_speed,
                           const io::GimbalState & gimbal, Timestamp now)
{
  BuffPlan result;
  if (!target) return result;
  if (bullet_speed < config_.bullet_speed_min || bullet_speed > config_.bullet_speed_max)
    bullet_speed = config_.bullet_speed_default;
  const double stale =
    std::max(0.0, std::chrono::duration<double>(now - target->timestamp).count());
  const double distance = target->center.norm();
  double fly_time = distance / bullet_speed;
  std::optional<Eigen::Vector2d> angles;
  for (int i = 0; i < 5; ++i) {
    const double future = stale + config_.shoot_delay + fly_time;
    const auto point = future_point(*target, future);
    if (!point) return result;
    tools::Trajectory bullet(
      bullet_speed, std::hypot(point->x(), point->y()), point->z());
    if (bullet.unsolvable) return result;
    angles = Eigen::Vector2d(std::remainder(std::atan2(point->y(), point->x()) +
                                              config_.yaw_offset, 2 * kPi),
                             -bullet.pitch - config_.pitch_offset);
    if (std::abs(bullet.fly_time - fly_time) < 0.001) break;
    fly_time = bullet.fly_time;
  }
  if (!angles) return result;
  const double future = stale + config_.shoot_delay + fly_time;
  const auto before = aim_angles(*target, future - 0.01, bullet_speed,
                                 config_.yaw_offset, config_.pitch_offset);
  const auto after = aim_angles(*target, future + 0.01, bullet_speed,
                                config_.yaw_offset, config_.pitch_offset);
  if (!before || !after) return result;
  result.control = true;
  result.yaw = angles->x();
  result.pitch = angles->y();
  result.yaw_vel = std::remainder(after->x() - before->x(), 2 * kPi) / 0.02;
  result.pitch_vel = (after->y() - before->y()) / 0.02;
  result.distance = distance;
  if (!attack_start_) attack_start_ = now;
  const double cycle = config_.rune_idle_duration + config_.rune_shoot_duration;
  const double phase =
    std::fmod(std::chrono::duration<double>(now - *attack_start_).count(), cycle);
  const bool shoot_phase = phase >= config_.rune_idle_duration;
  result.fire = shoot_phase &&
                std::abs(std::remainder(result.yaw - gimbal.yaw, 2 * kPi)) <
                  config_.yaw_tolerance &&
                std::abs(result.pitch - gimbal.pitch) < config_.pitch_tolerance;
  return result;
}
}  // namespace auto_buff_v2
