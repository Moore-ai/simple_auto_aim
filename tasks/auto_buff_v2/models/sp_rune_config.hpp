#ifndef AUTO_BUFF_V2__SP_RUNE_CONFIG_HPP
#define AUTO_BUFF_V2__SP_RUNE_CONFIG_HPP

#include <string>

#include <yaml-cpp/yaml.h>

namespace auto_buff_v2
{
inline constexpr int kSPSlotCount = 5;
inline constexpr double kSPSlotAngle = 6.28318530717958647692 / kSPSlotCount;

struct SPRuneConfig
{
  int max_lost_frames = 20;
  double small_speed_tolerance = 0.524;
  double small_yaw_acceleration_noise = 0.001;
  double big_yaw_acceleration_noise = 0.9;
  double big_rotation_angle_noise = 0.09;
  double big_rotation_speed_noise = 0.5;
  double big_phase_noise = 1.0;
  double observation_yaw = 0.05;
  double observation_pitch = 0.05;
  double observation_distance = 1.0;
  double observation_rotation_angle = 0.004;
  double initial_small_speed = 1.047;
  double initial_big_speed = 1.1775;
  double initial_amplitude = 0.9125;
  double initial_frequency = 1.942;
  double initial_phase = 0.0;
  double initial_covariance = 10.0;
  double small_speed_covariance = 0.01;
  double big_speed_covariance = 100.0;
  double big_phase_covariance = 400.0;
  double divergence_factor = 1.5;

  std::string pnp_mode = "4i";
  double target_radius = 0.700;
  double target_top = 0.115;
  double target_bottom = 0.115;
  double target_left = 0.150;
  double target_right = 0.115;
  double r_center_z = 0.0;
  double min_depth = 0.1;
  double max_depth = 20.0;
  double small_rotation_acceleration_noise = 0.005;
  double blade_observation_yaw = 0.1;
  double blade_observation_pitch = 0.1;
  double blade_observation_distance = 1.0;
  double max_center_angle_residual = 0.55;
  double max_center_distance_residual = 2.5;
  double max_roll_residual = 0.55;
  double max_prediction_gap = 1.0;
  int max_blade_missed_frames = 30;
  double slot_match_angle = 0.6283185307179586;
  double amplitude_min = 0.78;
  double amplitude_max = 1.045;
  double frequency_min = 1.884;
  double frequency_max = 2.0;
  double speed_baseline = 2.09;
  double big_speed_limit = 4.0;

  int fit_iterations = 100;
  int fit_history = 150;
  double fit_threshold = 0.5;
  double fit_max_speed = 2.1;
  double fit_output_limit = 4.0;

  double phase_switch_threshold = 0.2617993877991494;
  double phase_match_tolerance = kSPSlotAngle / 2;
  int direction_vote_min_count = 5;
  int direction_vote_limit = 50;
  double direction_vote_max_delta = 0.7853981633974483;

  void validate() const;
  static SPRuneConfig load(const YAML::Node & node);
};
}  // namespace auto_buff_v2

#endif
