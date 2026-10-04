#include <cassert>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tasks/auto_buff_v2/aimers/rune_aimer_factory.hpp"
#include "tasks/auto_buff_v2/planners/mpc_buff_tracking.hpp"
#include "tasks/auto_buff_v2/detectors/rune_detector_factory.hpp"

int main()
{
  constexpr char path[] = "/tmp/buff_v2_config_test.yaml";
  std::ofstream(path) <<
    "camera_matrix: [1000, 0, 640, 0, 900, 360, 0, 0, 1]\n"
    "distort_coeffs: [0.1, -0.2, 0, 0, 0]\n"
    "camera2gimbal_mode: xyz_ypr\n"
    "camera2gimbal_xyz: [0.1, 0.2, 0.3]\n"
    "camera2gimbal_ypr: [0, 0, 0]\n"
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
    "  planner_mode: njust\n"
    "  detector:\n"
    "    type: njust\n"
    "    njust:\n"
    "      min_distance: 2.1\n"
    "      max_distance: 4.8\n"
    "      active_threshold: 0.3\n"
    "      match_threshold: 0.6\n"
    "      max_perspective: 55\n"
    "      shoot_delay: 0.9\n"
    "  aimer:\n"
    "    type: njust\n"
    "    njust:\n"
    "      shoot_delay: 0.07\n"
    "      max_distance: 9.0\n"
    "      rune_idle_duration: 0.5\n"
    "      rune_shoot_duration: 0.3\n"
    "      yaw_tolerance: 0.08\n"
    "      pitch_tolerance: 0.05\n"
    // Conflicting detector and root values must not override aimer parameters.
    "  shoot_delay: 0.9\n"
    "  rune_idle_duration: 0.9\n"
    "  rune_shoot_duration: 0.9\n"
    "  yaw_tolerance: 0.001\n"
    "  pitch_tolerance: 0.001\n"
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
  assert(config.detector.type == "njust");
  const auto njust = auto_buff_v2::NjustRuneDetector::Config::load(
    config.detector.parameters["njust"]);
  assert(njust.max_distance == 4.8);
  assert(njust.active_threshold == 0.3);
  assert(njust.max_perspective == 55);
  assert(njust.min_distance == 2.1 && njust.match_threshold == 0.6);
  assert(config.model.timeout_seconds == 1.2 && config.model.noise_rotation_angle == 0.002);
  assert(config.model.diverge_face_angle == 40);
  assert(config.planner.mode == "njust");
  assert(config.planner.aimer == "njust");
  assert(config.planner.ballistic_model == "vacuum");
  assert(config.planner.ballistic_config.njust_air_resistance == 0.006);
  assert(config.planner.bullet_speed_min == 11 && config.planner.bullet_speed_default == 20);
  // Legacy iteration settings, including enable=false, cannot disable buff iteration.
  auto_buff_v2::RuneEstimate target;
  target.timestamp = std::chrono::steady_clock::now();
  target.start_timestamp = target.timestamp - std::chrono::seconds(4);
  target.center = {3, 0, 1};
  target.inactive[0] = true;
  auto aimer = auto_buff_v2::make_rune_aimer(config.planner);
  const auto aimed = aimer->aim(1, target, 20, {}, target.timestamp);
  assert(aimed);
  const double prediction_delay =
    std::chrono::duration<double>(aimed->prediction_time - target.timestamp).count();
  assert(std::abs(prediction_delay - (0.07 + aimed->solution.fly_time)) < 0.001);
  io::GimbalState gimbal;
  gimbal.yaw = aimed->plan.yaw;
  gimbal.pitch = aimed->plan.pitch;
  assert(!aimer->fire_advice(
    target.center, aimed->solution.point, aimed->plan, gimbal,
    target.timestamp + std::chrono::milliseconds(450)));
  assert(aimer->fire_advice(
    target.center, aimed->solution.point, aimed->plan, gimbal,
    target.timestamp + std::chrono::milliseconds(550)));
  assert(!aimer->fire_advice(
    target.center, aimed->solution.point, aimed->plan, gimbal,
    target.timestamp + std::chrono::milliseconds(850)));
  gimbal.yaw += 0.025;
  assert(aimer->fire_advice(
    target.center, aimed->solution.point, aimed->plan, gimbal,
    target.timestamp + std::chrono::milliseconds(550)));
  gimbal.yaw = aimed->plan.yaw;
  gimbal.pitch += 0.0115;
  assert(aimer->fire_advice(
    target.center, aimed->solution.point, aimed->plan, gimbal,
    target.timestamp + std::chrono::milliseconds(550)));
  auto_buff_v2::MpcBuffTracking planner(config.planner);
  const auto request = planner.prepare(1, target, 20, target.timestamp);
  assert(request);
  assert(std::abs(request->fly_time - 0.176280533350353) < 1e-9);
  assert(std::abs(request->yaw0 - std::acos(-1) / 2) < 1e-12);
  assert(std::abs(config.planner.yaw_offset - std::acos(-1) / 2) < 1e-12);
  assert(std::abs(config.planner.pitch_offset + std::acos(-1) / 4) < 1e-12);
  assert((config.camera.t_camera2gimbal - Eigen::Vector3d(0.1, 0.2, 0.3)).norm() < 1e-12);

  auto yaml = YAML::LoadFile(path);
  yaml["buff_v2"]["aimer"]["type"] = "climber";
  yaml["buff_v2"]["aimer"]["climber"]["fire_gap_time"] = 0.3;
  yaml["buff_v2"]["aimer"]["climber"]["predict_time"] = 0.08;
  std::ofstream(path) << yaml;
  const auto climber_config = auto_buff_v2::BuffConfig::load(path);
  assert(climber_config.planner.aimer == "climber");
  assert(climber_config.detector.type == "njust");
  const auto detector = auto_buff_v2::make_rune_detector(climber_config.detector);
  assert(dynamic_cast<auto_buff_v2::NjustRuneDetector *>(detector.get()));
  const auto climber = auto_buff_v2::ClimberRuneAimer::Config::load(
    climber_config.planner.aimer_parameters["climber"]);
  assert(climber.fire_gap_time == 0.3 && climber.predict_time == 0.08);
  auto climber_aimer = auto_buff_v2::make_rune_aimer(climber_config.planner);
  assert(climber_aimer->aim(1, target, 20, {}, target.timestamp));
  yaml["buff_v2"]["detector"]["type"] = "unknown";
  std::ofstream(path) << yaml;
  bool rejected = false;
  try {
    auto_buff_v2::make_rune_detector(auto_buff_v2::BuffConfig::load(path).detector);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

}
