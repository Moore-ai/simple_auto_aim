#include <cassert>
#include <cmath>
#include <fstream>
#include <list>
#include <sstream>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/lightbar_detector.hpp"
#include "tasks/auto_aim/reprojection.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"

int main()
{
  auto_aim::ReprojectionObservationConfig noise_config;
  noise_config.use_adaptive_noise = true;
  const auto noise = auto_aim::uvl_noise_variances(40.0, noise_config);
  assert(std::abs(noise[0] - 0.005) < 1e-12);
  assert(std::abs(noise[1] - 32.0) < 1e-12);
  assert(std::abs(noise[2] - 32.0) < 1e-12);
  assert(std::abs(noise[3] - 200.0) < 1e-12);

  const auto observation = auto_aim::uvl_from_endpoints({100.0F, 40.0F}, {100.0F, 80.0F});
  assert(std::abs(observation[0]) < 1e-12);
  assert(std::abs(observation[1] - 100.0) < 1e-12);
  assert(std::abs(observation[2] - 60.0) < 1e-12);
  assert(std::abs(observation[3] - 40.0) < 1e-12);

  const auto wrapped = auto_aim::uvl_residual(
    Eigen::Vector4d{3.13, 0.0, 0.0, 0.0}, Eigen::Vector4d{-3.13, 0.0, 0.0, 0.0});
  assert(std::abs(wrapped[0]) < 0.03);

  auto_aim::DetectionResult result;
  result.lightbars.emplace_back();
  assert(result.armors.empty());
  assert(result.lightbars.size() == 1);

  std::vector<Eigen::Vector4d> observations(3, Eigen::Vector4d::Ones());
  std::vector<Eigen::MatrixXd> jacobians(3, Eigen::MatrixXd::Zero(4, 11));
  const auto batch = auto_aim::make_uvl_batch(observations, jacobians, 4.0, 9.0);
  assert(batch.z.size() == 12);
  assert(batch.H.rows() == 12 && batch.H.cols() == 11);
  assert(batch.R.rows() == 12 && batch.R.cols() == 12);

  cv::Mat lightbar_image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::rectangle(lightbar_image, {180, 180, 10, 60}, cv::Scalar(255, 255, 0), cv::FILLED);
  cv::rectangle(lightbar_image, {260, 180, 10, 60}, cv::Scalar(255, 255, 0), cv::FILLED);
  auto lightbar_detector = auto_aim::LightbarDetector("configs/demo.yaml");
  const auto detected_lightbars = lightbar_detector.detect(lightbar_image);
  assert(detected_lightbars.size() == 2);
  assert(detected_lightbars.front().color == auto_aim::Color::blue);

  auto_aim::Solver solver("configs/demo.yaml");
  const auto projection = solver.project_armor_with_jacobian(
    {2.0, 0.0, 0.0}, 0.2, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
  assert(projection.valid);
  assert(projection.points.size() == 4);
  assert(projection.point_jacobian.size() == 4);
  assert(std::isfinite(projection.light_depth_diff));
  const auto reference_projection = solver.reproject_armor(
    {2.0, 0.0, 0.0}, 0.2, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
  assert(cv::norm(projection.points[0] - reference_projection[0]) < 1e-3);

  constexpr double epsilon = 1e-3;
  for (int parameter = 0; parameter < 4; ++parameter) {
    Eigen::Vector3d center{2.0, 0.0, 0.0};
    double yaw = 0.2;
    if (parameter < 3) {
      center[parameter] += epsilon;
    } else {
      yaw += epsilon;
    }
    const auto plus = solver.reproject_armor(
      center, yaw, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
    center = {2.0, 0.0, 0.0};
    yaw = 0.2;
    if (parameter < 3) {
      center[parameter] -= epsilon;
    } else {
      yaw -= epsilon;
    }
    const auto minus = solver.reproject_armor(
      center, yaw, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
    const auto numeric_x = (plus[0].x - minus[0].x) / (2.0 * epsilon);
    const auto numeric_y = (plus[0].y - minus[0].y) / (2.0 * epsilon);
    assert(std::abs(projection.point_jacobian[0](0, parameter) - numeric_x) < 1.0);
    assert(std::abs(projection.point_jacobian[0](1, parameter) - numeric_y) < 1.0);
  }
  for (int parameter = 0; parameter < 4; ++parameter) {
    Eigen::Vector3d center{2.0, 0.0, 0.0};
    double yaw = 0.2;
    if (parameter < 3) {
      center[parameter] += epsilon;
    } else {
      yaw += epsilon;
    }
    const auto plus = solver.project_armor_with_jacobian(
      center, yaw, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
    center = {2.0, 0.0, 0.0};
    yaw = 0.2;
    if (parameter < 3) {
      center[parameter] -= epsilon;
    } else {
      yaw -= epsilon;
    }
    const auto minus = solver.project_armor_with_jacobian(
      center, yaw, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
    assert(std::abs(
      projection.light_depth_diff_jacobian[parameter] -
      (plus.light_depth_diff - minus.light_depth_diff) / (2.0 * epsilon)) < 1e-6);
  }

  auto_aim::Tracker tracker("configs/demo.yaml", solver);
  assert(!tracker.reprojection_enabled());

  std::ifstream config_input("configs/demo.yaml");
  std::stringstream config_buffer;
  config_buffer << config_input.rdbuf();
  auto config_text = config_buffer.str();
  const auto mode_pos = config_text.find("observation_mode: \"pnp\"");
  assert(mode_pos != std::string::npos);
  config_text.replace(mode_pos, std::string("observation_mode: \"pnp\"").size(),
                      "observation_mode: \"reprojection\"");
  std::ofstream config_output("/tmp/reprojection_observation_test.yaml");
  config_output << config_text;
  config_output.close();
  auto reprojection_solver = auto_aim::Solver("/tmp/reprojection_observation_test.yaml");
  auto reprojection_tracker =
    auto_aim::Tracker("/tmp/reprojection_observation_test.yaml", reprojection_solver);
  assert(reprojection_tracker.reprojection_enabled());

  auto inekf_config = config_text;
  const auto filter_pos = inekf_config.find("filter_method: \"ekf\"");
  assert(filter_pos != std::string::npos);
  inekf_config.replace(filter_pos, std::string("filter_method: \"ekf\"").size(),
                        "filter_method: \"inekf\"");
  std::ofstream inekf_output("/tmp/reprojection_inekf_test.yaml");
  inekf_output << inekf_config;
  inekf_output.close();
  auto inekf_solver = auto_aim::Solver("/tmp/reprojection_inekf_test.yaml");
  auto inekf_tracker = auto_aim::Tracker("/tmp/reprojection_inekf_test.yaml", inekf_solver);
  assert(!inekf_tracker.reprojection_enabled());
  assert(inekf_tracker.debug_info().observation_mode == "pnp");

  const Eigen::Vector3d armor_center{2.0, 0.0, 0.0};
  const auto armor_points = reprojection_solver.reproject_armor(
    armor_center, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
  auto armor = auto_aim::Armor(0, 0.9F, cv::Rect(700, 480, 100, 100), armor_points);
  reprojection_solver.solve(armor);
  auto target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 0.2, 4, Eigen::VectorXd::Ones(11));
  auto left = auto_aim::Lightbar();
  left.color = auto_aim::Color::blue;
  left.top = armor_points[0];
  left.bottom = armor_points[3];
  auto right = auto_aim::Lightbar();
  right.color = auto_aim::Color::blue;
  right.top = armor_points[1];
  right.bottom = armor_points[2];
  const auto updated = target.update_reprojection(
    {{0, false, left}, {0, true, right}}, reprojection_solver, {});
  assert(updated);
  assert(target.ekf_x().size() == 11);

  auto_aim::ReprojectionObservationConfig diff_config;
  auto reprojection_target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 0.2, 4, Eigen::VectorXd::Ones(11),
    {}, auto_aim::FilterMethod::EKF, true, diff_config);
  const auto updated_with_diff = reprojection_target.update_reprojection(
    {{0, false, left}, {0, true, right}}, {{0, armor}}, reprojection_solver, diff_config);
  assert(updated_with_diff);
  assert(std::isfinite(reprojection_target.filter().last_nis));

  auto tracker_frame_1 = auto_aim::DetectionResult{{armor}, {left, right}};
  const auto tracker_time = std::chrono::steady_clock::now();
  const auto initialized_targets = reprojection_tracker.track(tracker_frame_1, tracker_time);
  assert(!initialized_targets.empty());
  auto full_update_frame = auto_aim::DetectionResult{{armor}, {left, right}};
  const auto full_update_targets = reprojection_tracker.track(
    full_update_frame, tracker_time + std::chrono::milliseconds(10));
  assert(!full_update_targets.empty());
  assert(reprojection_tracker.debug_info().matched_armor_count == 1);
  assert(reprojection_tracker.debug_info().uvl_observation_count >= 2);

  auto legacy_config_text = config_text;
  for (const auto * key : {
         "r_sigma_px_by_length_ratio", "r_sigma_length_by_length_ratio", "r_sigma_angle",
         "r_sigma_armor_lights_depth_diff", "armor_match_gate", "armor_match_gate_not_all_init",
         "armor_match_w_center_err", "armor_match_w_angle_err", "armor_match_w_side_length_err",
         "light_match_length_ratio_gate", "light_match_angle_gate",
         "light_match_pos_gate_by_length_ratio", "radius_min", "radius_max"}) {
    const auto line_start = legacy_config_text.find(std::string("  ") + key + ":");
    if (line_start != std::string::npos) {
      const auto line_end = legacy_config_text.find('\n', line_start);
      legacy_config_text.erase(line_start, line_end - line_start + 1);
    }
  }
  std::ofstream legacy_config_output("/tmp/reprojection_legacy.yaml");
  legacy_config_output << legacy_config_text;
  legacy_config_output.close();
  auto legacy_solver = auto_aim::Solver("/tmp/reprojection_legacy.yaml");
  auto legacy_tracker = auto_aim::Tracker("/tmp/reprojection_legacy.yaml", legacy_solver);
  auto legacy_init = legacy_tracker.track(tracker_frame_1, tracker_time);
  assert(!legacy_init.empty());
  auto legacy_update_frame = auto_aim::DetectionResult{{armor}, {left, right}};
  auto legacy_updated = legacy_tracker.track(
    legacy_update_frame, tracker_time + std::chrono::milliseconds(10));
  assert(!legacy_updated.empty());
  assert(legacy_tracker.debug_info().matched_armor_count == 1);
  assert(legacy_tracker.debug_info().uvl_observation_count >= 2);

  auto constrained_target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 2.0, 4, Eigen::VectorXd::Ones(11),
    {}, auto_aim::FilterMethod::EKF, true, diff_config);
  constrained_target.predict(0.0);
  assert(constrained_target.ekf_x()[8] <= diff_config.radius_max);
  auto pnp_target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 2.0, 4, Eigen::VectorXd::Ones(11));
  pnp_target.predict(0.0);
  assert(pnp_target.ekf_x()[8] > diff_config.radius_max);

  const auto predicted_points = reprojection_solver.reproject_armor(
    initialized_targets.front().armor_xyza_list().front().head<3>(),
    initialized_targets.front().armor_xyza_list().front()[3], auto_aim::ArmorType::small,
    auto_aim::ArmorName::sentry);
  auto predicted_left = left;
  predicted_left.top = predicted_points[0];
  predicted_left.bottom = predicted_points[3];
  auto light_only_frame = auto_aim::DetectionResult{{}, {predicted_left}};
  const auto after_light_only_targets =
    reprojection_tracker.track(light_only_frame, tracker_time + std::chrono::milliseconds(10));
  assert(after_light_only_targets.empty());
  return 0;
}
