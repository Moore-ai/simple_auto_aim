#include "buff_config.hpp"

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

  const auto buff = yaml["buff_v2"];
  if (buff) {
    const auto detector = buff["detector"];
    if (detector) {
      result.detector.type = detector["type"].as<std::string>(result.detector.type);
      result.detector.parameters = detector;
    }
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
    result.planner.mode = buff["planner_mode"].as<std::string>(result.planner.mode);
    const auto aimer = buff["aimer"];
    if (aimer) {
      result.planner.aimer = aimer["type"].as<std::string>(result.planner.aimer);
      result.planner.aimer_parameters = aimer;
      const auto ballistic = aimer["ballistic"];
      if (ballistic) {
        result.planner.ballistic.type =
          ballistic["type"].as<std::string>(result.planner.ballistic.type);
        const auto njust = ballistic["njust"];
        if (njust)
          result.planner.ballistic.njust_air_resistance =
            njust["air_resistance"].as<double>(result.planner.ballistic.njust_air_resistance);
        const auto climber = ballistic["climber"];
        if (climber)
          result.planner.ballistic.climber_air_resistance_k = climber["air_resistance_k"].as<double>(
            result.planner.ballistic.climber_air_resistance_k);
      }
    }
  }
  result.planner.yaw_offset = yaml["yaw_offset"].as<double>(0) * kPi / 180;
  result.planner.pitch_offset = yaml["pitch_offset"].as<double>(0) * kPi / 180;
  result.planner.bullet_speed_min =
    yaml["bullet_speed_min"].as<double>(result.planner.bullet_speed_min);
  result.planner.bullet_speed_max =
    yaml["bullet_speed_max"].as<double>(result.planner.bullet_speed_max);
  result.planner.bullet_speed_default =
    yaml["bullet_speed_default"].as<double>(result.planner.bullet_speed_default);
  return result;
}
}  // namespace auto_buff_v2
