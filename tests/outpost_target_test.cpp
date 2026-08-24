#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/observation_path.hpp"
#include "tasks/auto_aim/target.hpp"
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
}
