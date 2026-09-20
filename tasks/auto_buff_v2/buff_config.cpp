#include "buff_config.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "tools/camera2gimbal_extrinsic.hpp"

namespace auto_buff_v2
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
}

BuffConfig BuffConfig::load(const std::string & path)
{
  const auto yaml = YAML::LoadFile(path);
  BuffConfig result;
  const auto intrinsics = yaml["camera_matrix"].as<std::vector<double>>();
  result.camera.camera_matrix = (cv::Mat_<double>(3, 3) <<
    intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4], intrinsics[5],
    intrinsics[6], intrinsics[7], intrinsics[8]);
  result.detector.fx = intrinsics[0];
  result.detector.fy = intrinsics[4];

  const auto distortion = yaml["distort_coeffs"].as<std::vector<double>>();
  result.camera.distort_coeffs = cv::Mat(1, 5, CV_64F);
  for (int i = 0; i < 5; ++i) result.camera.distort_coeffs.at<double>(0, i) = distortion[i];
  const auto extrinsic = tools::load_camera2gimbal_extrinsic(yaml);
  result.camera.R_camera2gimbal = extrinsic.rotation;
  result.camera.t_camera2gimbal = extrinsic.translation;
  const auto body = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  result.camera.R_gimbal2imubody = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(body.data());

  const auto buff = yaml["buff_v2"];
  if (buff) {
    result.detector.min_distance = buff["min_distance"].as<double>(result.detector.min_distance);
    result.detector.max_distance = buff["max_distance"].as<double>(result.detector.max_distance);
    result.detector.active_threshold =
      buff["active_threshold"].as<double>(result.detector.active_threshold);
    result.detector.match_threshold =
      buff["match_threshold"].as<double>(result.detector.match_threshold);
    result.model.timeout_seconds = buff["timeout_seconds"].as<double>(result.model.timeout_seconds);
    result.model.noise_x = buff["noise_x"].as<double>(result.model.noise_x);
    result.model.noise_y = buff["noise_y"].as<double>(result.model.noise_y);
    result.model.noise_z = buff["noise_z"].as<double>(result.model.noise_z);
    result.model.noise_rotation_speed =
      buff["noise_rotation_speed"].as<double>(result.model.noise_rotation_speed);
    result.model.noise_rotation_angle =
      buff["noise_rotation_angle"].as<double>(result.model.noise_rotation_angle);
    result.model.noise_face_yaw =
      buff["noise_face_yaw"].as<double>(result.model.noise_face_yaw);
    result.model.noise_observation =
      buff["noise_observation"].as<double>(result.model.noise_observation);
    result.model.gate_threshold = buff["gate_threshold"].as<double>(result.model.gate_threshold);
    result.model.init_seed_mean_error =
      buff["init_seed_mean_error"].as<double>(result.model.init_seed_mean_error);
    result.model.init_seed_max_error =
      buff["init_seed_max_error"].as<double>(result.model.init_seed_max_error);
    result.model.init_center_gate =
      buff["init_center_gate"].as<double>(result.model.init_center_gate);
    result.model.init_pitch_bound =
      buff["init_pitch_bound"].as<double>(result.model.init_pitch_bound);
    result.model.diverge_face_angle =
      buff["diverge_face_angle"].as<double>(result.model.diverge_face_angle);
    result.planner.shoot_delay = buff["shoot_delay"].as<double>(result.planner.shoot_delay);
    result.planner.rune_idle_duration =
      buff["rune_idle_duration"].as<double>(result.planner.rune_idle_duration);
    result.planner.rune_shoot_duration =
      buff["rune_shoot_duration"].as<double>(result.planner.rune_shoot_duration);
    result.planner.yaw_tolerance =
      buff["yaw_tolerance"].as<double>(result.planner.yaw_tolerance);
    result.planner.pitch_tolerance =
      buff["pitch_tolerance"].as<double>(result.planner.pitch_tolerance);
  }
  result.planner.ballistic_model =
    yaml["ballistic_model"].as<std::string>(result.planner.ballistic_model);
  result.planner.ballistic_config.njust_air_resistance = yaml["njust_air_resistance"].as<double>(
    result.planner.ballistic_config.njust_air_resistance);
  result.planner.yaw_offset = yaml["yaw_offset"].as<double>(0) * kPi / 180;
  result.planner.pitch_offset = yaml["pitch_offset"].as<double>(0) * kPi / 180;
  result.planner.bullet_speed_min =
    yaml["bullet_speed_min"].as<double>(result.planner.bullet_speed_min);
  result.planner.bullet_speed_max =
    yaml["bullet_speed_max"].as<double>(result.planner.bullet_speed_max);
  result.planner.bullet_speed_default =
    yaml["bullet_speed_default"].as<double>(result.planner.bullet_speed_default);
  if (const auto iteration_yaml = yaml["fly_time_iteration"]; iteration_yaml) {
    result.planner.fly_time_iteration_enabled =
      iteration_yaml["enable"].as<bool>(result.planner.fly_time_iteration_enabled);
    result.planner.fly_time_iteration_max_iteration = std::max(
      iteration_yaml["max_iteration"].as<int>(result.planner.fly_time_iteration_max_iteration), 1);
    result.planner.fly_time_iteration_convergence_threshold = std::max(
      iteration_yaml["convergence_threshold"].as<double>(
        result.planner.fly_time_iteration_convergence_threshold), 0.0);
  }
  return result;
}
}  // namespace auto_buff_v2
