#include "climber_rune_aimer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "../rune_predictor.hpp"
#include "tools/trajectory.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

// Climber AirResistTrajectory model: dv/dt = -k*v - g, low arc within +/-60 degrees.
std::optional<tools::BallisticSolution> solve_trajectory(
  double speed, double distance, double height, double drag)
{
  if (distance < 0.1 || !std::isfinite(speed) || !std::isfinite(distance) ||
      !std::isfinite(height)) return std::nullopt;
  const tools::Trajectory vacuum(speed, distance, height);
  if (vacuum.unsolvable || std::abs(vacuum.pitch) > kPi / 3) return std::nullopt;
  if (drag < 1e-6) return tools::BallisticSolution{vacuum.pitch, vacuum.fly_time};
  constexpr double gravity = 9.7833;
  const auto flight_time = [&](double pitch) {
    const double ratio = drag * distance / (speed * std::cos(pitch));
    return ratio < 0.99 ? -std::log1p(-ratio) / drag : NAN;
  };
  const auto residual = [&](double pitch) {
    const double time = flight_time(pitch);
    return (speed * std::sin(pitch) + gravity / drag) *
             distance / (speed * std::cos(pitch)) - gravity * time / drag - height;
  };
  // Locate the first root to select the low arc, then bisect the height residual.
  double lower = -kPi / 3;
  for (int i = 1; i <= 128; ++i) {
    double upper = -kPi / 3 + i * (2 * kPi / 3) / 128;
    if (residual(lower) <= 0 && residual(upper) >= 0) {
      for (int j = 0; j < 60; ++j) {
        const double middle = (lower + upper) / 2;
        if (residual(middle) < 0) lower = middle;
        else upper = middle;
      }
      const double pitch = (lower + upper) / 2;
      return tools::BallisticSolution{pitch, flight_time(pitch)};
    }
    lower = upper;
  }
  return std::nullopt;
}

// RunePredictor handles forward prediction. Climber's MPC also samples before the estimate.
RuneEstimate rewind_to(const RuneEstimate & target, Timestamp time)
{
  auto state = target;
  const double dt = std::chrono::duration<double>(time - target.timestamp).count();
  if (state.sine_valid) {
    state.sine_phase += state.sine_omega * dt;
    state.sine_t += dt;
    if (std::abs(state.sine_omega) > 1e-12) {
      state.rotation_angle += state.sine_v * dt + state.sine_a / state.sine_omega *
        (std::cos(target.sine_phase) - std::cos(state.sine_phase));
    } else {
      state.rotation_angle += state.rotation_speed * dt;
    }
    state.rotation_speed = state.sine_v + state.sine_a * std::sin(state.sine_phase);
  } else {
    state.rotation_angle += state.rotation_speed * dt;
  }
  state.timestamp = time;
  return state;
}
}  // namespace

ClimberRuneAimer::Config ClimberRuneAimer::Config::load(const YAML::Node & node)
{
  Config result;
  if (node) {
    result.fire_gap_time = node["fire_gap_time"].as<double>(result.fire_gap_time);
    result.predict_time = node["predict_time"].as<double>(result.predict_time);
    result.air_resistance_k = node["air_resistance_k"].as<double>(result.air_resistance_k);
  }
  if (!std::isfinite(result.fire_gap_time) || result.fire_gap_time < 0 ||
      !std::isfinite(result.predict_time) || result.predict_time <= 0 ||
      !std::isfinite(result.air_resistance_k) || result.air_resistance_k < 0) {
    throw std::invalid_argument(
      "buff_v2.aimer.climber requires fire_gap_time >= 0, predict_time > 0 and air_resistance_k >= 0");
  }
  return result;
}

ClimberRuneAimer::ClimberRuneAimer(BuffConfig::Planner config, Config climber_config)
: config_(std::move(config)), climber_config_(climber_config)
{
}

std::optional<AimSolution> ClimberRuneAimer::aim_at(
  const RuneEstimate & target, Timestamp prediction_time, double speed) const
{
  if (speed < 10 || speed > 28) speed = 24;
  const auto state =
    prediction_time < target.timestamp ? rewind_to(target, prediction_time) : target;
  const auto point = RunePredictor{}.aimpoint_at(state, prediction_time);
  if (!point) return std::nullopt;
  const double distance = std::hypot(point->x(), point->y());
  const auto bullet =
    solve_trajectory(speed, distance, point->z(), climber_config_.air_resistance_k);
  if (!bullet) return std::nullopt;
  return AimSolution{
    {std::atan2(point->y(), point->x()) + config_.yaw_offset,
     -bullet->pitch - config_.pitch_offset},
    bullet->fly_time, *point};
}

std::optional<RuneAim> ClimberRuneAimer::aim(
  std::uint64_t generation, const std::optional<RuneEstimate> & target, double speed,
  const io::GimbalState & gimbal, Timestamp now)
{
  if (!target) {
    generation_.reset();
    return std::nullopt;
  }
  if (generation_ != generation) {
    generation_ = generation;
    last_fire_ = now;
    last_angles_.setZero();
    mistake_count_ = 0;
    switch_fanblade_ = false;
    first_in_aimer_ = true;
  }
  if (speed < 10 || speed > 28) speed = 24;
  predicted_ = target->predicted;
  const auto future = offset_time(now, climber_config_.predict_time);
  const auto first = aim_at(*target, future, speed);
  if (!first) {
    first_in_aimer_ = true;
    return std::nullopt;
  }
  const auto prediction_time = offset_time(future, first->fly_time);
  const auto second = aim_at(*target, prediction_time, speed);
  if (!second || std::abs(second->fly_time - first->fly_time) > 0.03) {
    first_in_aimer_ = true;
    return std::nullopt;
  }

  auto_aim::Plan result;
  result.yaw = second->angles.x();
  result.pitch = second->angles.y();
  result.distance = target->center.norm();
  result.fly_time = second->fly_time;
  result.debug_xyza = {second->point.x(), second->point.y(), second->point.z(), result.yaw};
  const bool changed = (second->angles - last_angles_).cwiseAbs().maxCoeff() > 5 / 57.3;
  if (mistake_count_ > 3) {
    switch_fanblade_ = true;
    mistake_count_ = 0;
    result.control = true;
    first_in_aimer_ = true;
  } else if (changed) {
    switch_fanblade_ = true;
    ++mistake_count_;
    result.control = true;
    first_in_aimer_ = true;
  } else {
    switch_fanblade_ = false;
    mistake_count_ = 0;
    result.control = true;
  }
  last_angles_ = second->angles;
  if (switch_fanblade_) last_fire_ = now;

  if (result.control) {
    if (first_in_aimer_) {
      first_in_aimer_ = false;
    } else {
      // Climber restores the original estimate between forward and backward solves.
      const double dt = climber_config_.predict_time;
      const auto previous_time = offset_time(target->timestamp, -dt);
      const auto previous_first = aim_at(*target, previous_time, speed);
      if (!previous_first) {
        first_in_aimer_ = true;
        return std::nullopt;
      }
      const auto previous = aim_at(
        *target, offset_time(previous_time, previous_first->fly_time), speed);
      if (!previous || std::abs(previous->fly_time - previous_first->fly_time) > 0.03) {
        first_in_aimer_ = true;
        return std::nullopt;
      }
      result.yaw_vel =
        std::remainder(second->angles.x() - previous->angles.x(), 2 * kPi) / (2 * dt);
      result.yaw_acc = (std::remainder(second->angles.x() - gimbal.yaw, 2 * kPi) -
                        std::remainder(gimbal.yaw - previous->angles.x(), 2 * kPi)) / (dt * dt);
      result.pitch_vel =
        std::remainder(second->angles.y() - previous->angles.y(), 2 * kPi) / (2 * dt);
      result.pitch_acc = (second->angles.y() - 2 * gimbal.pitch + previous->angles.y()) / (dt * dt);
    }
  }
  return RuneAim{result, *second, prediction_time, speed};
}

bool ClimberRuneAimer::fire_advice(
  const Eigen::Vector3d &, const Eigen::Vector3d &, const auto_aim::Plan & plan,
  const io::GimbalState &, Timestamp now)
{
  if (!plan.control || !generation_ || switch_fanblade_) return false;
  if (std::chrono::duration<double>(now - last_fire_).count() <= climber_config_.fire_gap_time)
    return false;
  last_fire_ = now;
  return !predicted_;
}
}  // namespace auto_buff_v2
