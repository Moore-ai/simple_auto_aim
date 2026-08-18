#include <cassert>
#include <cmath>
#include <fstream>
#include <list>
#include <sstream>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detection_result.hpp"
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
  assert(result.has_independent_lightbars());
  assert(!result.independent_lightbars_collected());
  result.lightbars_collected = true;
  assert(result.independent_lightbars_collected());

  assert(!auto_aim::DetectionOptions::from_yaml(
    YAML::Load("observation_mode: pnp\nfilter_method: ekf\n"))
             .collect_lightbars);
  assert(auto_aim::DetectionOptions::from_yaml(
           YAML::Load("observation_mode: pnp\nfilter_method: ekf\npnp:\n  "
                      "enable_lightbar_assist: true\n"))
           .collect_lightbars);
  assert(auto_aim::DetectionOptions::from_yaml(
           YAML::Load("observation_mode: reprojection\nfilter_method: ekf\n"))
           .collect_lightbars);
  assert(!auto_aim::DetectionOptions::from_yaml(
           YAML::Load("observation_mode: reprojection\nfilter_method: inekf\n"))
             .collect_lightbars);

  std::vector<Eigen::Vector4d> observations(3, Eigen::Vector4d::Ones());
  std::vector<Eigen::MatrixXd> jacobians(
    3, Eigen::MatrixXd::Zero(4, auto_aim::TargetState::dimension));
  const auto batch = auto_aim::make_uvl_batch(observations, jacobians, 4.0, 9.0);
  assert(batch.z.size() == 12);
  assert(batch.H.rows() == 12 && batch.H.cols() == auto_aim::TargetState::dimension);
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

  auto_aim::ObservationGeometry geometry(solver);
  const auto center_jacobian =
    Eigen::Matrix<double, 3, auto_aim::TargetState::dimension>::Zero();
  const auto predicted_left_geometry = geometry.project_lightbar(
    {2.0, 0.0, 0.0}, 0.2, false, auto_aim::ArmorType::small,
    auto_aim::ArmorName::sentry, center_jacobian);
  assert(predicted_left_geometry.valid);
  assert(predicted_left_geometry.uvl[3] > 0.0);
  auto measured_lightbar = auto_aim::Lightbar();
  measured_lightbar.top = predicted_left_geometry.top;
  measured_lightbar.bottom = predicted_left_geometry.bottom;
  assert(geometry.lightbar_match_cost(
    measured_lightbar, predicted_left_geometry, auto_aim::ReprojectionObservationConfig{}));
  measured_lightbar.top.x += 1000.0F;
  assert(!geometry.lightbar_match_cost(
    measured_lightbar, predicted_left_geometry, auto_aim::ReprojectionObservationConfig{}));

  assert(geometry.armor_match_cost(
    reference_projection, {2.0, 0.0, 0.0}, 0.2, auto_aim::ArmorType::small,
    auto_aim::ArmorName::sentry, auto_aim::ReprojectionObservationConfig{}, true));
  auto bad_armor_points = reference_projection;
  bad_armor_points.front().x += 1000.0F;
  assert(!geometry.armor_match_cost(
    bad_armor_points, {2.0, 0.0, 0.0}, 0.2, auto_aim::ArmorType::small,
    auto_aim::ArmorName::sentry, auto_aim::ReprojectionObservationConfig{}, true));

  const auto depth_difference = geometry.project_light_depth_difference(
    {2.0, 0.0, 0.0}, 0.2, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry,
    center_jacobian);
  assert(depth_difference.valid);
  assert(std::abs(depth_difference.value - projection.light_depth_diff) < 1e-12);

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
  assert(!tracker.debug_info().lightbar_assist_enabled);

  std::ifstream config_input("configs/demo.yaml");
  std::stringstream config_buffer;
  config_buffer << config_input.rdbuf();
  const auto pnp_config_text = config_buffer.str();
  auto config_text = pnp_config_text;
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
  assert(!reprojection_tracker.debug_info().lightbar_assist_enabled);

  auto pnp_without_node = pnp_config_text;
  const auto pnp_node_pos = pnp_without_node.find("\npnp:\n");
  assert(pnp_node_pos != std::string::npos);
  const auto pnp_node_end = pnp_without_node.find("\n\n", pnp_node_pos);
  assert(pnp_node_end != std::string::npos);
  pnp_without_node.erase(pnp_node_pos, pnp_node_end - pnp_node_pos);
  std::ofstream pnp_without_node_output("/tmp/pnp_without_node.yaml");
  pnp_without_node_output << pnp_without_node;
  pnp_without_node_output.close();
  auto pnp_without_node_solver = auto_aim::Solver("/tmp/pnp_without_node.yaml");
  auto pnp_without_node_tracker =
    auto_aim::Tracker("/tmp/pnp_without_node.yaml", pnp_without_node_solver);
  assert(!pnp_without_node_tracker.debug_info().lightbar_assist_enabled);

  auto pnp_assist_config = pnp_config_text;
  const auto assist_switch_pos = pnp_assist_config.find("enable_lightbar_assist: false");
  assert(assist_switch_pos != std::string::npos);
  pnp_assist_config.replace(
    assist_switch_pos, std::string("enable_lightbar_assist: false").size(),
    "enable_lightbar_assist: true");
  std::ofstream pnp_assist_output("/tmp/pnp_lightbar_assist.yaml");
  pnp_assist_output << pnp_assist_config;
  pnp_assist_output.close();
  auto pnp_assist_solver = auto_aim::Solver("/tmp/pnp_lightbar_assist.yaml");
  auto pnp_assist_tracker =
    auto_aim::Tracker("/tmp/pnp_lightbar_assist.yaml", pnp_assist_solver);
  assert(pnp_assist_tracker.debug_info().lightbar_assist_enabled);

  auto pnp_assist_reprojection_config = pnp_assist_config;
  const auto pnp_mode_pos = pnp_assist_reprojection_config.find("observation_mode: \"pnp\"");
  assert(pnp_mode_pos != std::string::npos);
  pnp_assist_reprojection_config.replace(
    pnp_mode_pos, std::string("observation_mode: \"pnp\"").size(),
    "observation_mode: \"reprojection\"");
  std::ofstream pnp_assist_reprojection_output("/tmp/pnp_assist_reprojection.yaml");
  pnp_assist_reprojection_output << pnp_assist_reprojection_config;
  pnp_assist_reprojection_output.close();
  auto pnp_assist_reprojection_solver =
    auto_aim::Solver("/tmp/pnp_assist_reprojection.yaml");
  auto pnp_assist_reprojection_tracker = auto_aim::Tracker(
    "/tmp/pnp_assist_reprojection.yaml", pnp_assist_reprojection_solver);
  assert(pnp_assist_reprojection_tracker.reprojection_enabled());
  assert(!pnp_assist_reprojection_tracker.debug_info().lightbar_assist_enabled);

  auto inekf_config = pnp_assist_config;
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
  assert(!inekf_tracker.debug_info().lightbar_assist_enabled);

  auto ukf_config = pnp_assist_config;
  const auto ukf_filter_pos = ukf_config.find("filter_method: \"ekf\"");
  assert(ukf_filter_pos != std::string::npos);
  ukf_config.replace(ukf_filter_pos, std::string("filter_method: \"ekf\"").size(),
                     "filter_method: \"ukf\"");
  std::ofstream ukf_output("/tmp/reprojection_ukf_test.yaml");
  ukf_output << ukf_config;
  ukf_output.close();
  auto ukf_solver = auto_aim::Solver("/tmp/reprojection_ukf_test.yaml");
  auto ukf_tracker = auto_aim::Tracker("/tmp/reprojection_ukf_test.yaml", ukf_solver);
  assert(!ukf_tracker.reprojection_enabled());
  assert(ukf_tracker.debug_info().observation_mode == "pnp");
  assert(!ukf_tracker.debug_info().lightbar_assist_enabled);

  const Eigen::Vector3d armor_center{2.0, 0.0, 0.0};
  const auto armor_points = reprojection_solver.reproject_armor(
    armor_center, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
  auto armor = auto_aim::Armor(0, 0.9F, cv::Rect(700, 480, 100, 100), armor_points);
  reprojection_solver.solve(armor);
  auto target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 0.2, 4,
    Eigen::VectorXd::Ones(auto_aim::TargetState::dimension));
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
  assert(target.state_vector().size() == auto_aim::TargetState::dimension);

  auto_aim::ReprojectionObservationConfig diff_config;
  auto reprojection_target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 0.2, 4,
    Eigen::VectorXd::Ones(auto_aim::TargetState::dimension),
    {}, auto_aim::FilterMethod::EKF, true, diff_config);
  const auto updated_with_diff = reprojection_target.update_reprojection(
    {{0, false, left}, {0, true, right}}, {{0, armor}}, reprojection_solver, diff_config);
  assert(updated_with_diff);
  assert(std::isfinite(reprojection_target.diagnostics().nis));

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

  auto pnp_assist_init_frame = auto_aim::DetectionResult{{armor}, {left, right}};
  const auto pnp_assist_init_targets = pnp_assist_tracker.track(
    pnp_assist_init_frame, tracker_time);
  assert(!pnp_assist_init_targets.empty());
  auto pnp_assist_lightbars = std::list<auto_aim::Lightbar>{left, right};
  const auto predicted_armors = pnp_assist_init_targets.front().armor_xyza_list();
  for (int id = 1; id < 4; ++id) {
    const auto points = pnp_assist_solver.reproject_armor(
      predicted_armors[id].head<3>(), predicted_armors[id][3],
      auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
    auto independent = auto_aim::Lightbar();
    independent.color = auto_aim::Color::blue;
    independent.top = points[0];
    independent.bottom = points[3];
    pnp_assist_lightbars.push_back(independent);
  }
  auto pnp_assist_frame = auto_aim::DetectionResult{{armor}, pnp_assist_lightbars};
  const auto pnp_assist_targets = pnp_assist_tracker.track(
    pnp_assist_frame, tracker_time + std::chrono::milliseconds(10));
  assert(!pnp_assist_targets.empty());
  assert(pnp_assist_tracker.debug_info().matched_armor_count == 1);
  assert(pnp_assist_tracker.debug_info().matched_light_count > 0);
  assert(pnp_assist_tracker.debug_info().uvl_observation_count ==
    pnp_assist_tracker.debug_info().matched_light_count);
  assert(pnp_assist_tracker.debug_info().last_update_source == "pnp_lightbar_assist");

  const auto pnp_disabled_init_targets = tracker.track(pnp_assist_init_frame, tracker_time);
  assert(!pnp_disabled_init_targets.empty());
  auto pnp_disabled_frame = auto_aim::DetectionResult{{armor}, pnp_assist_lightbars};
  const auto pnp_disabled_targets = tracker.track(
    pnp_disabled_frame, tracker_time + std::chrono::milliseconds(10));
  assert(!pnp_disabled_targets.empty());
  assert(tracker.debug_info().matched_light_count == 0);
  assert(tracker.debug_info().uvl_observation_count == 0);
  assert(tracker.debug_info().last_update_source == "pnp");

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
    armor, std::chrono::steady_clock::now(), 2.0, 4,
    Eigen::VectorXd::Ones(auto_aim::TargetState::dimension),
    {}, auto_aim::FilterMethod::EKF, true, diff_config);
  constrained_target.predict(0.0);
  assert(constrained_target.state().radius() <= diff_config.radius_max);
  auto pnp_target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 2.0, 4,
    Eigen::VectorXd::Ones(auto_aim::TargetState::dimension));
  pnp_target.predict(0.0);
  assert(pnp_target.state().radius() > diff_config.radius_max);

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
