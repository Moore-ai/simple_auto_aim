#include <cassert>
#include <cmath>
#include <fstream>

#include "tasks/auto_buff_v2/buff_config.hpp"

int main()
{
  constexpr char path[] = "/tmp/buff_v2_config_test.yaml";
  std::ofstream(path) <<
    "camera_matrix: [1000, 0, 640, 0, 900, 360, 0, 0, 1]\n"
    "distort_coeffs: [0.1, -0.2, 0, 0, 0]\n"
    "camera2gimbal_mode: xyz_ypr\n"
    "camera2gimbal_xyz: [0.1, 0.2, 0.3]\n"
    "camera2gimbal_ypr: [0, 0, 0]\n"
    "R_gimbal2imubody: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
    "ballistic_model: vacuum\n"
    "njust_air_resistance: 0.006\n"
    "yaw_offset: 90\n"
    "pitch_offset: -45\n"
    "bullet_speed_min: 11\n"
    "bullet_speed_max: 22\n"
    "bullet_speed_default: 20\n"
    "fly_time_iteration:\n"
    "  enable: false\n"
    "  max_iteration: 7\n"
    "  convergence_threshold: 0.002\n"
    "buff_v2:\n"
    "  min_distance: 2.1\n"
    "  max_distance: 4.8\n"
    "  active_threshold: 0.3\n"
    "  match_threshold: 0.6\n"
    "  shoot_delay: 0.07\n"
    "  rune_idle_duration: 0.5\n"
    "  rune_shoot_duration: 0.3\n"
    "  yaw_tolerance: 0.08\n"
    "  pitch_tolerance: 0.05\n"
    "  timeout_seconds: 1.2\n"
    "  noise_x: 0.02\n"
    "  noise_y: 0.03\n"
    "  noise_z: 0.04\n"
    "  noise_rotation_speed: 1.1\n"
    "  noise_rotation_angle: 0.002\n"
    "  noise_face_yaw: 0.005\n"
    "  noise_observation: 12\n"
    "  gate_threshold: 10\n"
    "  init_seed_mean_error: 9\n"
    "  init_seed_max_error: 19\n"
    "  init_center_gate: 29\n"
    "  init_pitch_bound: 18\n"
    "  diverge_face_angle: 40\n";

  const auto config = auto_buff_v2::BuffConfig::load(path);

  assert(config.camera.camera_matrix.at<double>(0, 0) == 1000);
  assert(config.camera.distort_coeffs.at<double>(0, 1) == -0.2);
  assert(config.detector.fx == 1000 && config.detector.fy == 900);
  assert(config.detector.min_distance == 2.1 && config.detector.match_threshold == 0.6);
  assert(config.model.timeout_seconds == 1.2 && config.model.noise_rotation_angle == 0.002);
  assert(config.model.diverge_face_angle == 40);
  assert(config.planner.shoot_delay == 0.07 && config.planner.rune_shoot_duration == 0.3);
  assert(config.planner.ballistic_model == "vacuum");
  assert(config.planner.ballistic_config.njust_air_resistance == 0.006);
  assert(config.planner.bullet_speed_min == 11 && config.planner.bullet_speed_default == 20);
  assert(!config.planner.fly_time_iteration_enabled);
  assert(config.planner.fly_time_iteration_max_iteration == 7);
  assert(config.planner.fly_time_iteration_convergence_threshold == 0.002);
  assert(std::abs(config.planner.yaw_offset - std::acos(-1) / 2) < 1e-12);
  assert(std::abs(config.planner.pitch_offset + std::acos(-1) / 4) < 1e-12);
  assert((config.camera.t_camera2gimbal - Eigen::Vector3d(0.1, 0.2, 0.3)).norm() < 1e-12);
}
