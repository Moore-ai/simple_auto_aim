#ifndef AUTO_BUFF_V2__NJUST_RUNE_CONFIG_HPP
#define AUTO_BUFF_V2__NJUST_RUNE_CONFIG_HPP

#include <yaml-cpp/yaml.h>

#include "njust_energy_fitter.hpp"
#include "rune_model.hpp"

namespace auto_buff_v2
{
struct NjustRuneConfig
{
  RunePredictionMethod prediction_method = RunePredictionMethod::weighted_least_squares;  // 小符
  NjustEnergyFitter::Config small_motion_fit;
  NjustEnergyFitter::Config big_motion_fit;
  double timeout_seconds = 1.5;
  double noise_x = 1e-5;
  double noise_y = 1e-5;
  double noise_z = 1e-5;
  double noise_rotation_speed = 1;
  double noise_rotation_angle = 1e-3;
  double noise_face_yaw = 1e-5;
  double noise_observation = 20;
  double gate_threshold = 13.816;
  double init_seed_mean_error = 10;
  double init_seed_max_error = 20;
  double init_center_gate = 30;
  double init_pitch_bound = 20;
  double diverge_face_angle = 45;

  static NjustRuneConfig load(const YAML::Node & node);
};
}  // namespace auto_buff_v2

#endif
