#include "njust_rune_config.hpp"

#include <stdexcept>
#include <string>

namespace auto_buff_v2
{
NjustRuneConfig NjustRuneConfig::load(const YAML::Node & node)
{
  NjustRuneConfig result;
  if (!node) return result;
  const auto prediction_method =
    node["prediction_method"].as<std::string>("weighted_least_squares");
  if (prediction_method == "ekf") {
    result.prediction_method = RunePredictionMethod::ekf;
  } else if (prediction_method != "weighted_least_squares") {
    throw std::invalid_argument(
      "buff_v2.model.njust.prediction_method must be weighted_least_squares or ekf");
  }
  const auto motion_fit = node["motion_fit"];
  if (motion_fit) {
    const auto load_fit = [](const YAML::Node & node, NjustEnergyFitter::Config & fit) {
      if (!node) return;
      fit.window_seconds = node["window_seconds"].as<double>(fit.window_seconds);
      fit.min_fit_seconds = node["min_fit_seconds"].as<double>(fit.min_fit_seconds);
      fit.weight_half_life_seconds =
        node["weight_half_life_seconds"].as<double>(fit.weight_half_life_seconds);
    };
    load_fit(motion_fit["small"], result.small_motion_fit);
    load_fit(motion_fit["big"], result.big_motion_fit);
  }
  result.timeout_seconds = node["timeout_seconds"].as<double>(result.timeout_seconds);
  result.noise_x = node["noise_x"].as<double>(result.noise_x);
  result.noise_y = node["noise_y"].as<double>(result.noise_y);
  result.noise_z = node["noise_z"].as<double>(result.noise_z);
  result.noise_rotation_speed =
    node["noise_rotation_speed"].as<double>(result.noise_rotation_speed);
  result.noise_rotation_angle =
    node["noise_rotation_angle"].as<double>(result.noise_rotation_angle);
  result.noise_face_yaw =
    node["noise_face_yaw"].as<double>(result.noise_face_yaw);
  result.noise_observation =
    node["noise_observation"].as<double>(result.noise_observation);
  result.gate_threshold = node["gate_threshold"].as<double>(result.gate_threshold);
  result.init_seed_mean_error =
    node["init_seed_mean_error"].as<double>(result.init_seed_mean_error);
  result.init_seed_max_error =
    node["init_seed_max_error"].as<double>(result.init_seed_max_error);
  result.init_center_gate =
    node["init_center_gate"].as<double>(result.init_center_gate);
  result.init_pitch_bound =
    node["init_pitch_bound"].as<double>(result.init_pitch_bound);
  result.diverge_face_angle =
    node["diverge_face_angle"].as<double>(result.diverge_face_angle);
  return result;
}

}  // namespace auto_buff_v2
