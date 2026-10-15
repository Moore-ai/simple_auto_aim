#include "sp_rune_config.hpp"

#include <stdexcept>

#include <opencv2/core.hpp>

namespace auto_buff_v2
{
SPRuneConfig SPRuneConfig::load(const YAML::Node & node)
{
  SPRuneConfig result;
  if (node) {
    result.max_lost_frames = node["max_lost_frames"].as<int>(result.max_lost_frames);
    result.small_speed_tolerance = node["small_speed_tolerance"].as<double>(
      result.small_speed_tolerance);
    result.small_yaw_acceleration_noise = node["small_yaw_acceleration_noise"].as<double>(
      result.small_yaw_acceleration_noise);
    result.big_yaw_acceleration_noise = node["big_yaw_acceleration_noise"].as<double>(
      result.big_yaw_acceleration_noise);
    result.big_rotation_angle_noise = node["big_rotation_angle_noise"].as<double>(
      result.big_rotation_angle_noise);
    result.big_rotation_speed_noise = node["big_rotation_speed_noise"].as<double>(
      result.big_rotation_speed_noise);
    result.big_phase_noise = node["big_phase_noise"].as<double>(result.big_phase_noise);
    result.observation_yaw = node["observation_yaw"].as<double>(result.observation_yaw);
    result.observation_pitch = node["observation_pitch"].as<double>(result.observation_pitch);
    result.observation_distance = node["observation_distance"].as<double>(
      result.observation_distance);
    result.observation_rotation_angle = node["observation_rotation_angle"].as<double>(
      result.observation_rotation_angle);
    result.initial_small_speed = node["initial_small_speed"].as<double>(result.initial_small_speed);
    result.initial_big_speed = node["initial_big_speed"].as<double>(result.initial_big_speed);
    result.initial_amplitude = node["initial_amplitude"].as<double>(result.initial_amplitude);
    result.initial_frequency = node["initial_frequency"].as<double>(result.initial_frequency);
    result.initial_phase = node["initial_phase"].as<double>(result.initial_phase);
    result.initial_covariance = node["initial_covariance"].as<double>(result.initial_covariance);
    result.small_speed_covariance = node["small_speed_covariance"].as<double>(
      result.small_speed_covariance);
    result.big_speed_covariance = node["big_speed_covariance"].as<double>(
      result.big_speed_covariance);
    result.big_phase_covariance = node["big_phase_covariance"].as<double>(
      result.big_phase_covariance);
    result.divergence_factor = node["divergence_factor"].as<double>(result.divergence_factor);
    result.pnp_mode = node["pnp_mode"].as<std::string>(result.pnp_mode);
    result.target_radius = node["target_radius"].as<double>(result.target_radius);
    result.target_top = node["target_top"].as<double>(result.target_top);
    result.target_bottom = node["target_bottom"].as<double>(result.target_bottom);
    result.target_left = node["target_left"].as<double>(result.target_left);
    result.target_right = node["target_right"].as<double>(result.target_right);
    result.r_center_z = node["r_center_z"].as<double>(result.r_center_z);
    result.min_depth = node["min_depth"].as<double>(result.min_depth);
    result.max_depth = node["max_depth"].as<double>(result.max_depth);
    result.small_rotation_acceleration_noise = node["small_rotation_acceleration_noise"].as<double>(
      result.small_rotation_acceleration_noise);
    result.blade_observation_yaw = node["blade_observation_yaw"].as<double>(
      result.blade_observation_yaw);
    result.blade_observation_pitch = node["blade_observation_pitch"].as<double>(
      result.blade_observation_pitch);
    result.blade_observation_distance = node["blade_observation_distance"].as<double>(
      result.blade_observation_distance);
    result.max_center_angle_residual = node["max_center_angle_residual"].as<double>(
      result.max_center_angle_residual);
    result.max_center_distance_residual = node["max_center_distance_residual"].as<double>(
      result.max_center_distance_residual);
    result.max_roll_residual = node["max_roll_residual"].as<double>(result.max_roll_residual);
    result.max_prediction_gap = node["max_prediction_gap"].as<double>(result.max_prediction_gap);
    result.max_blade_missed_frames = node["max_blade_missed_frames"].as<int>(
      result.max_blade_missed_frames);
    result.slot_match_angle = node["slot_match_angle"].as<double>(result.slot_match_angle);
    result.amplitude_min = node["amplitude_min"].as<double>(result.amplitude_min);
    result.amplitude_max = node["amplitude_max"].as<double>(result.amplitude_max);
    result.frequency_min = node["frequency_min"].as<double>(result.frequency_min);
    result.frequency_max = node["frequency_max"].as<double>(result.frequency_max);
    result.speed_baseline = node["speed_baseline"].as<double>(result.speed_baseline);
    result.big_speed_limit = node["big_speed_limit"].as<double>(result.big_speed_limit);
    result.fit_iterations = node["fit_iterations"].as<int>(result.fit_iterations);
    result.fit_history = node["fit_history"].as<int>(result.fit_history);
    result.fit_threshold = node["fit_threshold"].as<double>(result.fit_threshold);
    result.fit_max_speed = node["fit_max_speed"].as<double>(result.fit_max_speed);
    result.fit_output_limit = node["fit_output_limit"].as<double>(result.fit_output_limit);
    result.phase_switch_threshold = node["phase_switch_threshold"].as<double>(
      result.phase_switch_threshold);
    result.phase_match_tolerance = node["phase_match_tolerance"].as<double>(
      result.phase_match_tolerance);
    result.direction_vote_min_count = node["direction_vote_min_count"].as<int>(
      result.direction_vote_min_count);
    result.direction_vote_limit = node["direction_vote_limit"].as<int>(result.direction_vote_limit);
    result.direction_vote_max_delta = node["direction_vote_max_delta"].as<double>(
      result.direction_vote_max_delta);
  }
  result.validate();
  return result;
}

void SPRuneConfig::validate() const
{
  if (pnp_mode != "4" && pnp_mode != "4i" && pnp_mode != "5")
    throw std::invalid_argument("buff_v2.model.sp.pnp_mode must be 4, 4i or 5");
  if (max_lost_frames < 0 || max_blade_missed_frames < 0)
    throw std::invalid_argument("SP lost-frame limits must be nonnegative");
  const auto require_positive = [](double value, const char * name) {
    if (!(value > 0))
      throw std::invalid_argument(std::string("invalid SP ") + name);
  };
  require_positive(small_speed_tolerance, "small_speed_tolerance");
  require_positive(small_yaw_acceleration_noise, "small_yaw_acceleration_noise");
  require_positive(big_yaw_acceleration_noise, "big_yaw_acceleration_noise");
  require_positive(big_rotation_angle_noise, "big_rotation_angle_noise");
  require_positive(big_rotation_speed_noise, "big_rotation_speed_noise");
  require_positive(big_phase_noise, "big_phase_noise");
  require_positive(observation_yaw, "observation_yaw");
  require_positive(observation_pitch, "observation_pitch");
  require_positive(observation_distance, "observation_distance");
  require_positive(observation_rotation_angle, "observation_rotation_angle");
  require_positive(initial_small_speed, "initial_small_speed");
  require_positive(initial_big_speed, "initial_big_speed");
  require_positive(initial_amplitude, "initial_amplitude");
  require_positive(initial_frequency, "initial_frequency");
  require_positive(initial_covariance, "initial_covariance");
  require_positive(small_speed_covariance, "small_speed_covariance");
  require_positive(big_speed_covariance, "big_speed_covariance");
  require_positive(big_phase_covariance, "big_phase_covariance");
  require_positive(divergence_factor, "divergence_factor");
  require_positive(target_radius, "target_radius");
  require_positive(target_top, "target_top");
  require_positive(target_bottom, "target_bottom");
  require_positive(target_left, "target_left");
  require_positive(target_right, "target_right");
  require_positive(min_depth, "min_depth");
  require_positive(max_depth, "max_depth");
  require_positive(small_rotation_acceleration_noise, "small_rotation_acceleration_noise");
  require_positive(blade_observation_yaw, "blade_observation_yaw");
  require_positive(blade_observation_pitch, "blade_observation_pitch");
  require_positive(blade_observation_distance, "blade_observation_distance");
  require_positive(max_center_angle_residual, "max_center_angle_residual");
  require_positive(max_center_distance_residual, "max_center_distance_residual");
  require_positive(max_roll_residual, "max_roll_residual");
  require_positive(max_prediction_gap, "max_prediction_gap");
  require_positive(slot_match_angle, "slot_match_angle");
  require_positive(amplitude_min, "amplitude_min");
  require_positive(amplitude_max, "amplitude_max");
  require_positive(frequency_min, "frequency_min");
  require_positive(frequency_max, "frequency_max");
  require_positive(speed_baseline, "speed_baseline");
  require_positive(big_speed_limit, "big_speed_limit");
  if (fit_iterations <= 0 || fit_history < 3 || !(fit_threshold > 0) ||
      !(fit_max_speed > 0) || !(fit_output_limit > 0))
    throw std::invalid_argument("invalid SP diagnostic fit parameters");
  if (!(phase_switch_threshold > 0) ||
      !(phase_match_tolerance > 0 && phase_match_tolerance <= kSPSlotAngle / 2) ||
      !(direction_vote_max_delta > 0 && direction_vote_max_delta <= kSPSlotAngle) ||
      direction_vote_min_count <= 0 || direction_vote_limit < direction_vote_min_count)
    throw std::invalid_argument("invalid SP phase or direction-vote parameters");
  if (max_depth <= min_depth || target_radius <= target_bottom ||
    amplitude_max < amplitude_min || frequency_max < frequency_min ||
    initial_frequency < frequency_min || initial_frequency > frequency_max ||
    initial_amplitude < amplitude_min || initial_amplitude > amplitude_max ||
    divergence_factor < 1 || slot_match_angle > CV_PI / 5)
    throw std::invalid_argument("inconsistent SP geometry or motion bounds");
}

}  // namespace auto_buff_v2
