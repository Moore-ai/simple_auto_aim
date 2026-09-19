#ifndef AUTO_BUFF_V2__BUFF_CONFIG_HPP
#define AUTO_BUFF_V2__BUFF_CONFIG_HPP

#include <string>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "tools/ballistic_solver.hpp"

namespace auto_buff_v2
{
struct BuffConfig
{
  struct Camera
  {
    cv::Mat camera_matrix;
    cv::Mat distort_coeffs;
    Eigen::Matrix3d R_camera2gimbal = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d R_gimbal2imubody = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_camera2gimbal = Eigen::Vector3d::Zero();
  } camera;

  struct Detector
  {
    double fx = 0;
    double fy = 0;
    double min_distance = 1.7;
    double max_distance = 5.0;
    double active_threshold = 0.2;
    double match_threshold = 0.5;
  } detector;

  struct Model
  {
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
  } model;

  struct Planner
  {
    double shoot_delay = 0.04;
    double rune_idle_duration = 0.4;
    double rune_shoot_duration = 0.2;
    double yaw_tolerance = 0.07;
    double pitch_tolerance = 0.04;
    double yaw_offset = 0;
    double pitch_offset = 0;
    double bullet_speed_min = 10;
    double bullet_speed_max = 25;
    double bullet_speed_default = 23.4;
    std::string ballistic_model = "njust";
    tools::BallisticSolverConfig ballistic_config;
  } planner;

  static BuffConfig load(const std::string & path);
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__BUFF_CONFIG_HPP
