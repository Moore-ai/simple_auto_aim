#include "outpost_target_v2.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"

namespace auto_aim
{

OutpostTargetV2::OutpostTargetV2(
  const std::vector<Armor> & armors, const Eigen::VectorXd & covariance_diagonal,
  const OutpostTargetV2Config & config)
: config_(config)
{
  const auto selected = std::min_element(
    armors.begin(), armors.end(), [](const Armor & lhs, const Armor & rhs) {
      return lhs.ypr_in_world[0] < rhs.ypr_in_world[0];
    });
  OutpostStateV2 initial_state(Eigen::VectorXd::Zero(dimension));
  if (selected != armors.end()) {
    initial_state.set_center_x(
      selected->xyz_in_world.x() + config_.radius * std::cos(selected->ypr_in_world[0]));
    initial_state.set_center_y(
      selected->xyz_in_world.y() + config_.radius * std::sin(selected->ypr_in_world[0]));
    initial_state.set_center_z(selected->xyz_in_world.z());
    initial_state.set_yaw(tools::limit_rad(selected->ypr_in_world[0]));
  }
  estimator_ = TargetEstimator(initial_state.vector(), covariance_diagonal, {FilterMethod::EKF});
}

std::unique_ptr<OutpostModel> OutpostTargetV2::clone() const
{
  return std::make_unique<OutpostTargetV2>(*this);
}

void OutpostTargetV2::begin_frame() {}

void OutpostTargetV2::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(dimension, dimension);
  F(0, 1) = dt;
  F(2, 3) = dt;
  F(4, 5) = dt;
  F(6, 7) = dt;
  const auto a = std::pow(dt, 4) / 4.0;
  const auto b = std::pow(dt, 3) / 2.0;
  const auto c = dt * dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(dimension, dimension);
  for (const auto position : {0, 2}) {
    Q(position, position) = a * config_.q_xy;
    Q(position, position + 1) = b * config_.q_xy;
    Q(position + 1, position) = b * config_.q_xy;
    Q(position + 1, position + 1) = c * config_.q_xy;
  }
  Q(4, 4) = a * config_.q_z;
  Q(4, 5) = b * config_.q_z;
  Q(5, 4) = b * config_.q_z;
  Q(5, 5) = c * config_.q_z;
  Q(6, 6) = a * config_.q_yaw;
  Q(6, 7) = b * config_.q_yaw;
  Q(7, 6) = b * config_.q_yaw;
  Q(7, 7) = c * config_.q_yaw_rate;
  Q(8, 8) = a * config_.q_dz;
  Q(9, 9) = a * config_.q_dz;
  estimator_.predict_vector(F, Q, [F](const Eigen::VectorXd & values) {
    Eigen::VectorXd predicted = F * values;
    OutpostStateV2 state(predicted);
    state.set_yaw(tools::limit_rad(state.yaw()));
    predicted = state.vector();
    return predicted;
  });
  clamp_height_offsets();
  apply_yaw_rate();
}

OutpostUpdateResult OutpostTargetV2::update(const std::vector<Armor> & armors)
{
  if (armors.empty()) return {};
  const auto predicted = state();
  const auto shoot_yaw = std::atan2(predicted.center_y(), predicted.center_x());
  const auto frontal = std::any_of(armors.begin(), armors.end(), [&](const Armor & armor) {
    return std::abs(tools::limit_rad(armor.ypr_in_world[0] - shoot_yaw)) < config_.frontal_angle_gate;
  });
  if (!frontal) return {};

  const auto & armor = armors.front();
  const auto id = match_id(predicted, armor);
  const auto position_error = (armor.xyz_in_world - armor_center(predicted, id)).norm();
  const auto yaw_error = std::abs(tools::limit_rad(
    armor.ypr_in_world[0] - tools::limit_rad(predicted.yaw() + id * 2.0 * CV_PI / 3.0)));
  if (position_error >= config_.position_match_gate || yaw_error >= config_.yaw_match_gate) {
    if (++mismatch_count_ > config_.mismatch_reset_count) {
      auto values = state();
      values.set_yaw_rate(0.0);
      estimator_.set_state_vector(values.vector());
      direction_data_.clear();
      direction_ = Direction::unknown;
      mismatch_count_ = 0;
    }
    return {};
  }

  mismatch_count_ = 0;
  if (!config_.is_fit_yaw_rate) {
    update_direction(armor.ypr_in_world[0]);
    apply_yaw_rate();
  }
  const auto H = observation_jacobian(state(), id);
  const auto center_angle = std::abs(tools::limit_rad(
    armor.ypr_in_world[0] - std::atan2(predicted.center_y(), predicted.center_x())));
  const Eigen::VectorXd observation{
    {armor.ypd_in_world[0], armor.ypd_in_world[1], armor.ypd_in_world[2], armor.ypr_in_world[0]}};
  Eigen::VectorXd diagonal{
    {config_.r_yaw, config_.r_pitch,
     config_.r_distance * observation[2] * observation[2] * (center_angle * center_angle + 1.0),
     config_.r_armor_yaw}};
  auto model = [this, id](const Eigen::VectorXd & values) {
    const OutpostStateV2 state(values);
    const auto ypd = tools::xyz2ypd(armor_center(state, id));
    return Eigen::Vector4d{
      ypd[0], ypd[1], ypd[2], tools::limit_rad(state.yaw() + id * 2.0 * CV_PI / 3.0)};
  };
  auto residual = [](const Eigen::VectorXd & observation, const Eigen::VectorXd & prediction) {
    Eigen::VectorXd result = observation - prediction;
    result[0] = tools::limit_rad(result[0]);
    result[1] = tools::limit_rad(result[1]);
    result[3] = tools::limit_rad(result[3]);
    return result;
  };
  if (!estimator_.update_vector(observation, H, diagonal.asDiagonal(), model, residual)) return {};
  clamp_height_offsets();
  apply_yaw_rate();
  return {true, id, {id}};
}

OutpostSnapshot OutpostTargetV2::snapshot() const
{
  const auto outpost_state = state();
  TargetState result;
  result.set_center_x(outpost_state.center_x());
  result.set_velocity_x(outpost_state.velocity_x());
  result.set_center_y(outpost_state.center_y());
  result.set_velocity_y(outpost_state.velocity_y());
  result.set_center_z(outpost_state.center_z());
  result.set_velocity_z(outpost_state.velocity_z());
  result.set_yaw(outpost_state.yaw());
  result.set_yaw_rate(outpost_state.yaw_rate());
  result.set_radius(config_.radius);
  return {
    result,
    outpost_state,
    state_vector(),
    armor_pose_list(),
    estimator_.diagnostics(),
    estimator_.last_nis(),
    direction_ == Direction::anticlockwise || direction_ == Direction::clockwise,
    estimator_.state_vector().allFinite()};
}

OutpostStateV2 OutpostTargetV2::state() const { return OutpostStateV2(estimator_.state_vector()); }

Eigen::VectorXd OutpostTargetV2::state_vector() const { return estimator_.state_vector(); }

std::vector<PredictedArmorPose> OutpostTargetV2::armor_pose_list() const
{
  std::vector<PredictedArmorPose> armor_poses;
  const auto outpost_state = state();
  armor_poses.reserve(OUTPOST_ARMOR_COUNT);
  for (int id = 0; id < OUTPOST_ARMOR_COUNT; ++id) {
    armor_poses.push_back({
      armor_center(outpost_state, id),
      tools::limit_rad(outpost_state.yaw() + id * 2.0 * CV_PI / 3.0),
      OUTPOST_MOUNT_PITCH});
  }
  return armor_poses;
}

const TargetEstimatorDiagnostics & OutpostTargetV2::diagnostics() const { return estimator_.diagnostics(); }

Eigen::Vector3d OutpostTargetV2::armor_center(const OutpostStateV2 & state, int id) const
{
  const auto yaw = state.yaw() + id * 2.0 * CV_PI / 3.0;
  const auto z = state.center_z() +
    (id == 1 ? state.height_offset_1() : (id == 2 ? state.height_offset_2() : 0.0));
  return {
    state.center_x() - config_.radius * std::cos(yaw),
    state.center_y() - config_.radius * std::sin(yaw), z};
}

Eigen::MatrixXd OutpostTargetV2::observation_jacobian(
  const OutpostStateV2 & state, int id) const
{
  const auto yaw = state.yaw() + id * 2.0 * CV_PI / 3.0;
  Eigen::MatrixXd armor_jacobian = Eigen::MatrixXd::Zero(4, dimension);
  armor_jacobian(0, 0) = 1.0;
  armor_jacobian(1, 2) = 1.0;
  armor_jacobian(2, 4) = 1.0;
  armor_jacobian(0, 6) = config_.radius * std::sin(yaw);
  armor_jacobian(1, 6) = -config_.radius * std::cos(yaw);
  armor_jacobian(3, 6) = 1.0;
  if (id == 1) armor_jacobian(2, 8) = 1.0;
  if (id == 2) armor_jacobian(2, 9) = 1.0;
  Eigen::Matrix4d coordinate_jacobian = Eigen::Matrix4d::Zero();
  coordinate_jacobian.topLeftCorner<3, 3>() = tools::xyz2ypd_jacobian(armor_center(state, id));
  coordinate_jacobian(3, 3) = 1.0;
  return coordinate_jacobian * armor_jacobian;
}

int OutpostTargetV2::match_id(const OutpostStateV2 & state, const Armor & armor) const
{
  auto best_id = 0;
  auto min_error = std::numeric_limits<double>::infinity();
  for (int id = 0; id < OUTPOST_ARMOR_COUNT; ++id) {
    const auto predicted_yaw = state.yaw() + id * 2.0 * CV_PI / 3.0;
    const auto error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - predicted_yaw));
    if (error < min_error) {
      min_error = error;
      best_id = id;
    }
  }
  return best_id;
}

void OutpostTargetV2::update_direction(double observed_yaw)
{
  if (direction_ != Direction::unknown && direction_ != Direction::stable) return;
  direction_data_.push_back(observed_yaw);
  if (static_cast<int>(direction_data_.size()) < config_.direction_sample_count) return;
  int stable = 0;
  int anticlockwise = 0;
  int clockwise = 0;
  for (std::size_t index = static_cast<std::size_t>(config_.direction_compare_interval);
       index < direction_data_.size(); ++index) {
    const auto delta = tools::limit_rad(
      direction_data_[index] - direction_data_[index - config_.direction_compare_interval]);
    if (std::abs(delta) > config_.direction_jump_threshold) continue;
    if (delta > config_.direction_stable_threshold)
      ++anticlockwise;
    else if (delta < -config_.direction_stable_threshold)
      ++clockwise;
    else
      ++stable;
  }
  if (anticlockwise > clockwise && anticlockwise > stable)
    direction_ = Direction::anticlockwise;
  else if (clockwise > anticlockwise && clockwise > stable)
    direction_ = Direction::clockwise;
  else if (clockwise == anticlockwise)
    direction_ = Direction::unknown;
  else
    direction_ = Direction::stable;
  direction_data_.clear();
}

void OutpostTargetV2::apply_yaw_rate()
{
  auto outpost_state = state();
  outpost_state.set_yaw(tools::limit_rad(outpost_state.yaw()));
  if (!config_.is_fit_yaw_rate) {
    if (direction_ == Direction::anticlockwise) outpost_state.set_yaw_rate(config_.yaw_rate_magnitude);
    if (direction_ == Direction::clockwise) outpost_state.set_yaw_rate(-config_.yaw_rate_magnitude);
  }
  estimator_.set_state_vector(outpost_state.vector());
}

void OutpostTargetV2::clamp_height_offsets()
{
  auto outpost_state = state();
  outpost_state.set_height_offset_1(std::clamp(outpost_state.height_offset_1(), -0.25, 0.25));
  outpost_state.set_height_offset_2(std::clamp(outpost_state.height_offset_2(), -0.25, 0.25));
  estimator_.set_state_vector(outpost_state.vector());
}

}  // namespace auto_aim
