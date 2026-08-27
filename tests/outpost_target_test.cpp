#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/observation_path.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/outpost_target_v2.hpp"
#include "tasks/auto_aim/outpost_state_v2.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/math_tools.hpp"

namespace
{

auto_aim::Armor make_outpost_armor(
  auto_aim::Solver & solver, const Eigen::Vector3d & center, double yaw)
{
  const auto points = solver.reproject_armor(
    center, yaw, auto_aim::ArmorType::small, auto_aim::ArmorName::outpost);
  return auto_aim::Armor(19, 0.99F, cv::Rect(640, 480, 100, 50), points);
}

auto_aim::Armor make_outpost_measurement(
  const auto_aim::Armor & prototype, const Eigen::Vector3d & rotation_center, double yaw,
  double height = 0.0)
{
  auto armor = prototype;
  armor.xyz_in_world = {
    rotation_center.x() - auto_aim::OUTPOST_RADIUS * std::cos(yaw),
    rotation_center.y() - auto_aim::OUTPOST_RADIUS * std::sin(yaw),
    rotation_center.z() + height};
  armor.ypr_in_world[0] = yaw;
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  return armor;
}

}  // namespace

int main()
{
  auto_aim::Solver solver("configs/demo.yaml");
  auto armor = make_outpost_armor(solver, {2.0, 0.0, 0.3}, 0.0);
  assert(armor.points.size() == 4);
  assert(solver.solve(armor));
  const auto target = auto_aim::Target(
    armor, std::chrono::steady_clock::now(), 0.4, 3,
    Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  const auto state = target.state();
  const auto predictions = target.armor_xyza_list();
  assert(predictions.size() == 3);
  const auto radius =
    (predictions.front().head<2>() - Eigen::Vector2d{state.center_x(), state.center_y()}).norm();
  assert(std::abs(radius - 0.275) < 1e-9);

  const auto poses = target.armor_pose_list();
  assert(poses.size() == 3);
  for (const auto & pose : poses) {
    assert(std::abs(pose.pitch + CV_PI / 12.0) < 1e-12);
    assert((pose.center - predictions[&pose - poses.data()].head<3>()).norm() < 1e-12);
    const auto sin_yaw = std::sin(pose.yaw);
    const auto cos_yaw = std::cos(pose.yaw);
    const auto sin_pitch = std::sin(pose.pitch);
    const auto cos_pitch = std::cos(pose.pitch);
    Eigen::Matrix3d expected_rotation;
    expected_rotation << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
      sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch,
      -sin_pitch, 0.0, cos_pitch;
    assert(pose.rotation().isApprox(expected_rotation, 1e-12));
  }

  const auto named_projection = solver.reproject_armor(
    poses.front().center, poses.front().yaw, auto_aim::ArmorType::small,
    auto_aim::ArmorName::outpost);
  const auto explicit_projection = solver.reproject_armor(
    poses.front().center, poses.front().yaw, poses.front().pitch,
    auto_aim::ArmorType::small);
  const auto vertical_projection = solver.reproject_armor(
    poses.front().center, poses.front().yaw, 0.0, auto_aim::ArmorType::small);
  assert(named_projection.size() == 4);
  assert(explicit_projection.size() == 4);
  assert(vertical_projection.size() == 4);
  double named_error = 0.0;
  double vertical_error = 0.0;
  for (std::size_t i = 0; i < named_projection.size(); ++i) {
    named_error += cv::norm(named_projection[i] - explicit_projection[i]);
    vertical_error += cv::norm(vertical_projection[i] - explicit_projection[i]);
  }
  assert(named_error < 1e-6);
  assert(vertical_error > 1.0);
  const Eigen::Vector3d visibility_center{2.0, 0.0, 1.0};
  const auto named_visibility =
    solver.armor_visibility_score(visibility_center, 0.0, auto_aim::ArmorName::outpost);
  const auto explicit_visibility = solver.armor_visibility_score(
    visibility_center, 0.0, auto_aim::OUTPOST_MOUNT_PITCH);
  const auto vertical_visibility = solver.armor_visibility_score(visibility_center, 0.0, 0.0);
  assert(std::abs(named_visibility - explicit_visibility) < 1e-12);
  assert(std::abs(vertical_visibility - explicit_visibility) > 1e-3);

  const Eigen::Vector3d true_rotation_center{2.0 + auto_aim::OUTPOST_RADIUS, 0.0, 0.3};
  const Eigen::Vector3d perturbed_rotation_center =
    true_rotation_center + Eigen::Vector3d{0.2, -0.1, 0.1};
  Eigen::VectorXd synthetic_p0{
    {1.0, 64.0, 1.0, 64.0, 1.0, 81.0, 0.4, 0.0}};
  const auto synthetic_t0 = std::chrono::steady_clock::now();
  auto synthetic_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, perturbed_rotation_center, 0.2), synthetic_t0,
    synthetic_p0);
  const auto synthetic_before = synthetic_target.outpost_state().value();
  const auto error_before =
    (Eigen::Vector3d{
       synthetic_before.center_x(), synthetic_before.center_y(), synthetic_before.center_z()} -
     true_rotation_center).norm() +
    std::abs(tools::limit_rad(synthetic_before.yaw()));
  auto_aim::ObservationPath synthetic_path(solver);
  auto_aim::ObservationPathConfig synthetic_config;
  synthetic_config.mode = auto_aim::ObservationMode::PNP;
  synthetic_path.configure(synthetic_config, auto_aim::Color::red);
  auto synthetic_detection = auto_aim::DetectionResult{
    {make_outpost_armor(solver, {2.0, 0.0, 0.3}, 0.0)}, {}};
  assert(synthetic_path.update(
    synthetic_target, synthetic_detection, synthetic_t0 + std::chrono::milliseconds(20)));
  const auto synthetic_after = synthetic_target.outpost_state().value();
  const auto error_after =
    (Eigen::Vector3d{
       synthetic_after.center_x(), synthetic_after.center_y(), synthetic_after.center_z()} -
     true_rotation_center).norm() +
    std::abs(tools::limit_rad(synthetic_after.yaw()));
  assert(error_after < error_before);

  const auto raw_state = target.state_vector();
  assert(raw_state.size() == 8);
  assert(std::abs(raw_state[7]) < 1e-12);
  const auto outpost_state = target.outpost_state();
  assert(outpost_state.has_value());
  assert(std::abs(predictions[0].z() - state.center_z() + 0.102) < 1e-12);
  assert(std::abs(predictions[1].z() - state.center_z()) < 1e-12);
  assert(std::abs(predictions[2].z() - state.center_z() - 0.102) < 1e-12);

  const auto t0 = std::chrono::steady_clock::now();
  auto direction_target = auto_aim::Target(
    armor, t0, 0.4, 3, Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  const auto initial = direction_target.outpost_state().value();
  const Eigen::Vector3d rotation_center{initial.center_x(), initial.center_y(), initial.center_z()};
  for (int frame = 1; frame <= 3; ++frame) {
    const auto timestamp = t0 + std::chrono::milliseconds(20 * frame);
    direction_target.predict(timestamp);
    auto measurement = make_outpost_measurement(armor, rotation_center, 0.04 * frame);
    direction_target.update(measurement);
    const auto rate = direction_target.outpost_state()->yaw_rate();
    if (frame < 3)
      assert(std::abs(rate) < 1e-12);
    else
      assert(std::abs(rate - auto_aim::OUTPOST_ANGULAR_SPEED) < 1e-12);
    assert(direction_target.last_id == 0);
  }

  auto predicted_target = direction_target;
  const auto state_before_prediction = predicted_target.outpost_state().value();
  constexpr double prediction_dt = 0.1;
  predicted_target.predict(prediction_dt);
  const auto state_after_prediction = predicted_target.outpost_state().value();
  assert(std::abs(
    state_after_prediction.center_x() - state_before_prediction.center_x() -
    state_before_prediction.velocity_x() * prediction_dt) < 1e-12);
  assert(std::abs(
    state_after_prediction.center_y() - state_before_prediction.center_y() -
    state_before_prediction.velocity_y() * prediction_dt) < 1e-12);
  assert(std::abs(
    state_after_prediction.center_z() - state_before_prediction.center_z() -
    state_before_prediction.velocity_z() * prediction_dt) < 1e-12);
  assert(std::abs(tools::limit_rad(
    state_after_prediction.yaw() - state_before_prediction.yaw() -
    auto_aim::OUTPOST_ANGULAR_SPEED * prediction_dt)) < 1e-12);

  for (int frame = 4; frame <= 9; ++frame) {
    direction_target.predict(t0 + std::chrono::milliseconds(20 * frame));
    const auto reverse_yaw = 0.12 - 0.04 * (frame - 3);
    direction_target.update(make_outpost_measurement(armor, rotation_center, reverse_yaw));
    const auto expected_rate = frame < 9 ? auto_aim::OUTPOST_ANGULAR_SPEED :
                                          -auto_aim::OUTPOST_ANGULAR_SPEED;
    assert(std::abs(direction_target.outpost_state()->yaw_rate() - expected_rate) < 1e-12);
    assert(direction_target.last_id == 0);
  }

  auto negative_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, rotation_center, 0.0), t0,
    Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  for (int frame = 1; frame <= 3; ++frame) {
    negative_target.predict(t0 + std::chrono::milliseconds(20 * frame));
    negative_target.update(
      make_outpost_measurement(armor, rotation_center, -0.04 * frame));
  }
  assert(std::abs(
    negative_target.outpost_state()->yaw_rate() + auto_aim::OUTPOST_ANGULAR_SPEED) < 1e-12);

  auto switch_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, rotation_center, 0.0), t0,
    Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  for (int frame = 1; frame <= 2; ++frame) {
    switch_target.predict(t0 + std::chrono::milliseconds(20 * frame));
    switch_target.update(make_outpost_measurement(armor, rotation_center, 0.04 * frame));
  }
  switch_target.predict(t0 + std::chrono::milliseconds(60));
  switch_target.update(make_outpost_measurement(
    armor, rotation_center, 2.0 * CV_PI / 3.0 + 0.12));
  assert(switch_target.last_id == 1);
  assert(std::abs(switch_target.outpost_state()->yaw_rate()) < 1e-12);

  auto boundary_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, {2.0, 0.0, 0.3}, 3.13), t0,
    Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  boundary_target.predict(t0 + std::chrono::milliseconds(20));
  boundary_target.update(make_outpost_measurement(armor, {2.0, 0.0, 0.3}, -3.00));
  assert(boundary_target.outpost_state()->yaw() > -CV_PI);
  assert(boundary_target.outpost_state()->yaw() <= CV_PI);

  const Eigen::Vector3d wrap_center{2.0, 0.0, 0.3};
  auto wrap_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, wrap_center, 3.10), t0,
    Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension));
  const std::array<double, 3> wrapped_yaws{3.13, -3.12, -3.08};
  for (std::size_t frame = 0; frame < wrapped_yaws.size(); ++frame) {
    wrap_target.predict(t0 + std::chrono::milliseconds(20 * (frame + 1)));
    wrap_target.update(make_outpost_measurement(
      armor, wrap_center, wrapped_yaws[frame]));
  }
  assert(std::abs(
    wrap_target.outpost_state()->yaw_rate() - auto_aim::OUTPOST_ANGULAR_SPEED) < 1e-12);

  Eigen::VectorXd outpost_p0{
    {1.0, 64.0, 1.0, 64.0, 1.0, 81.0, 0.4, 0.0}};
  auto height_target = auto_aim::Target(armor, t0, 0.4, 3, outpost_p0);
  const auto height_initial = height_target.outpost_state().value();
  const Eigen::Vector3d height_center{
    height_initial.center_x(), height_initial.center_y(), height_initial.center_z()};
  const std::array<double, 3> expected_heights{-0.102, 0.0, 0.102};
  for (int frame = 1; frame <= 30; ++frame) {
    height_target.predict(t0 + std::chrono::milliseconds(20 * frame));
    for (int id = 0; id < 3; ++id) {
      const auto yaw = id * 2.0 * CV_PI / 3.0;
      auto measurement =
        make_outpost_measurement(armor, height_center, yaw, expected_heights[id]);
      height_target.update(measurement);
      assert(height_target.last_id == id);
    }
  }
  const auto height_predictions = height_target.armor_xyza_list();
  assert(std::abs(height_predictions[0].z() - height_center.z() + 0.102) < 1e-12);
  assert(std::abs(height_predictions[1].z() - height_center.z()) < 1e-12);
  assert(std::abs(height_predictions[2].z() - height_center.z() - 0.102) < 1e-12);

  // The first visible armor need not be the physical low armor.  Once all three phase
  // observations are available, their fixed heights must establish the correct phase.
  const Eigen::Vector3d phase_center{2.0, 0.0, 0.3};
  auto phase_target = auto_aim::Target::make_outpost(
    make_outpost_measurement(armor, phase_center, 0.0, 0.102), t0, outpost_p0);
  assert(phase_target.update_outpost({
    make_outpost_measurement(armor, phase_center, 0.0, 0.102),
    make_outpost_measurement(armor, phase_center, 2.0 * CV_PI / 3.0, -0.102),
    make_outpost_measurement(armor, phase_center, 4.0 * CV_PI / 3.0, 0.0)}));
  const auto phase_state = phase_target.outpost_state().value();
  assert(std::abs(phase_state.center_z() - phase_center.z()) < 0.02);
  const auto phase_predictions = phase_target.armor_xyza_list();
  assert(std::abs(phase_predictions[0].z() - (phase_center.z() + 0.102)) < 0.02);
  assert(std::abs(phase_predictions[1].z() - (phase_center.z() - 0.102)) < 0.02);
  assert(std::abs(phase_predictions[2].z() - phase_center.z()) < 0.02);

  auto reprojection_target = auto_aim::Target(armor, t0, 0.4, 3, outpost_p0);
  auto_aim::ObservationPath observation_path(solver);
  auto_aim::ObservationPathConfig observation_config;
  observation_config.mode = auto_aim::ObservationMode::REPROJECTION;
  observation_path.configure(observation_config, auto_aim::Color::red);
  auto detections = auto_aim::DetectionResult{{armor}, {}};
  assert(observation_path.update(
    reprojection_target, detections, t0 + std::chrono::milliseconds(20)));
  assert(observation_path.debug_info().last_update_source == "pnp");

  auto forced_pnp_target = auto_aim::Target(
    armor, t0, 0.4, 3, outpost_p0, {}, auto_aim::FilterMethod::UKF, true);
  forced_pnp_target.predict(t0 + std::chrono::milliseconds(20));
  const auto forced_pnp_state = forced_pnp_target.outpost_state().value();
  assert(forced_pnp_target.state_vector().size() == auto_aim::OutpostState::dimension);

  const auto factory_target = auto_aim::Target::make_outpost(armor, t0, outpost_p0);
  assert(factory_target.outpost_state().has_value());
  assert(factory_target.armor_pose_list().size() == 3);

  auto unlocked_target = auto_aim::Target::make_outpost(armor, t0, outpost_p0);
  const auto unlocked_initial = unlocked_target.outpost_state().value();
  const Eigen::Vector3d unlocked_center{
    unlocked_initial.center_x(), unlocked_initial.center_y(), unlocked_initial.center_z()};
  for (int frame = 1; frame <= 11; ++frame) {
    unlocked_target.predict(t0 + std::chrono::milliseconds(20 * frame));
    auto measurement = make_outpost_measurement(armor, unlocked_center, 0.0);
    unlocked_target.update(measurement);
  }
  assert(!unlocked_target.convergened());

  // A V2 model keeps the RPS-style phase geometry and its two height offsets.
  auto_aim::OutpostTargetV2Config v2_config;
  auto_aim::OutpostTargetV2 v2_target(
    std::vector<auto_aim::Armor>{make_outpost_measurement(armor, {2.0, 0.0, 0.3}, 0.0)},
    Eigen::VectorXd::Ones(10), v2_config);
  const auto v2_state = v2_target.state_vector();
  assert(v2_state.size() == 10);
  const auto v2_predictions = v2_target.armor_pose_list();
  assert(v2_predictions.size() == 3);
  assert(std::abs(v2_predictions[0].center.z() - 0.3) < 1e-12);
  assert(std::abs(v2_predictions[1].center.z() - 0.3) < 1e-12);
  assert(std::abs(v2_predictions[2].center.z() - 0.3) < 1e-12);

  // V2 estimates height offsets in phase order instead of using the current model's fixed IDs.
  v2_config.frontal_angle_gate = CV_PI;
  v2_config.position_match_gate = 0.5;
  auto_aim::OutpostTargetV2 height_v2_target(
    std::vector<auto_aim::Armor>{make_outpost_measurement(armor, {2.0, 0.0, 0.3}, 0.0)},
    Eigen::VectorXd::Ones(10), v2_config);
  for (int repeat = 0; repeat < 20; ++repeat) {
    for (int id = 0; id < 3; ++id) {
      const double yaw = id * 2.0 * CV_PI / 3.0;
      const double height = id == 1 ? 0.08 : (id == 2 ? -0.06 : 0.0);
      assert(height_v2_target.update(
        {make_outpost_measurement(armor, {2.0, 0.0, 0.3}, yaw, height)}).updated);
    }
  }
  const auto height_v2_predictions = height_v2_target.armor_pose_list();
  assert(std::abs(height_v2_predictions[1].center.z() - 0.38) < 0.02);
  assert(std::abs(height_v2_predictions[2].center.z() - 0.24) < 0.02);

  // The V2 direction detector locks the known positive yaw rate after its configured samples.
  v2_config.direction_sample_count = 3;
  v2_config.direction_compare_interval = 1;
  v2_config.frontal_angle_gate = CV_PI;
  auto_aim::OutpostTargetV2 direction_v2_target(
    std::vector<auto_aim::Armor>{make_outpost_measurement(armor, {2.0, 0.0, 0.3}, 0.0)},
    Eigen::VectorXd::Ones(10), v2_config);
  for (int frame = 1; frame <= 3; ++frame) {
    const auto yaw = 0.02 * frame;
    assert(direction_v2_target.update(
      {make_outpost_measurement(armor, {2.0, 0.0, 0.3}, yaw)}).updated);
  }
  assert(std::abs(
    direction_v2_target.state_vector()[7] - v2_config.yaw_rate_magnitude) < 1e-12);

  Eigen::VectorXd named_state_values(10);
  named_state_values << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8, 0.09, -0.06;
  const auto named_state = auto_aim::OutpostStateV2(named_state_values);
  assert(named_state.center_x() == 1.0);
  assert(named_state.velocity_x() == 2.0);
  assert(named_state.center_y() == 3.0);
  assert(named_state.velocity_y() == 4.0);
  assert(named_state.center_z() == 5.0);
  assert(named_state.velocity_z() == 6.0);
  assert(named_state.yaw() == 0.7);
  assert(named_state.yaw_rate() == 0.8);
  assert(named_state.height_offset_1() == 0.09);
  assert(named_state.height_offset_2() == -0.06);

  // YAML selection constructs V2 through the real Tracker path.
  std::ifstream demo_input("configs/demo.yaml");
  std::stringstream demo_text;
  demo_text << demo_input.rdbuf();
  auto v2_yaml = demo_text.str();
  const auto model_position = v2_yaml.find("outpost_model: current");
  assert(model_position != std::string::npos);
  v2_yaml.replace(model_position, std::string("outpost_model: current").size(), "outpost_model: v2");
  std::ofstream v2_output("/tmp/outpost_v2_test.yaml");
  v2_output << v2_yaml;
  v2_output.close();
  auto_aim::Solver v2_solver("/tmp/outpost_v2_test.yaml");
  auto_aim::Tracker v2_tracker("/tmp/outpost_v2_test.yaml", v2_solver);
  v2_tracker.set_enemy_color(auto_aim::Color::red);
  auto v2_detections = auto_aim::DetectionResult{{armor}, {}};
  const auto v2_targets = v2_tracker.track(v2_detections, t0 + std::chrono::milliseconds(10));
  assert(!v2_targets.empty());
  assert(v2_targets.front().state_vector().size() == 10);
  assert(!v2_targets.front().outpost_state().has_value());
  const auto v2_target_state = v2_targets.front().outpost_state_v2();
  assert(v2_target_state.has_value());
  assert(std::abs(v2_target_state->center_x() - v2_targets.front().state_vector()[0]) < 1e-12);

  // The six-point optimizer must also run on the frame that initializes an outpost target.
  const Eigen::Vector3d optimizer_center{2.0, 0.0, 0.3};
  const auto clean_points = solver.reproject_armor(
    optimizer_center, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::outpost);
  auto perturbed_points = clean_points;
  const auto image_center =
    (clean_points[0] + clean_points[1] + clean_points[2] + clean_points[3]) * 0.25F;
  for (auto & point : perturbed_points) point = image_center + (point - image_center) * 0.97F;
  auto optimizer_armor =
    auto_aim::Armor(19, 0.99F, cv::Rect(640, 480, 100, 50), perturbed_points);
  auto expected_optimized = optimizer_armor;
  assert(solver.solve(expected_optimized));
  const auto four_point_position = expected_optimized.xyz_in_world;
  auto neighbor_lightbar = auto_aim::Lightbar();
  neighbor_lightbar.color = auto_aim::Color::red;
  neighbor_lightbar.top = {896.3336F, -827.9412F};
  neighbor_lightbar.bottom = {883.1733F, -777.8211F};
  assert(solver.optimize_outpost_distance(expected_optimized, {neighbor_lightbar}));
  assert((expected_optimized.xyz_in_world - four_point_position).norm() > 1e-6);

  auto optimizer_yaml = demo_text.str();
  const auto optimizer_switch_position =
    optimizer_yaml.find("enable_outpost_distance_optimizer: false");
  assert(optimizer_switch_position != std::string::npos);
  optimizer_yaml.replace(
    optimizer_switch_position, std::string("enable_outpost_distance_optimizer: false").size(),
    "enable_outpost_distance_optimizer: true");
  std::ofstream optimizer_output("/tmp/outpost_optimizer_init_test.yaml");
  optimizer_output << optimizer_yaml;
  optimizer_output.close();
  auto_aim::Solver optimizer_solver("/tmp/outpost_optimizer_init_test.yaml");
  auto_aim::Tracker optimizer_tracker("/tmp/outpost_optimizer_init_test.yaml", optimizer_solver);
  optimizer_tracker.set_enemy_color(auto_aim::Color::red);
  auto optimizer_detections =
    auto_aim::DetectionResult{{optimizer_armor}, {neighbor_lightbar}};
  const auto optimizer_targets = optimizer_tracker.track(
    optimizer_detections, std::chrono::steady_clock::now());
  assert(!optimizer_targets.empty());
  assert((optimizer_detections.armors.front().xyz_in_world -
          expected_optimized.xyz_in_world).norm() < 1e-6);
}
