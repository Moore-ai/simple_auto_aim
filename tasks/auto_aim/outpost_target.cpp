#include "outpost_target.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
constexpr std::array<double, OUTPOST_ARMOR_COUNT> OUTPOST_ARMOR_HEIGHT_OFFSETS{
  -0.102, 0.0, 0.102};
}

OutpostTarget::OutpostTarget(
  const Armor & armor, const Eigen::VectorXd & covariance_diagonal,
  const OutpostFilterConfig & config)
: config_(config)
{
  Eigen::VectorXd initial = Eigen::VectorXd::Zero(OutpostState::dimension);
  initial[0] = armor.xyz_in_world.x() + OUTPOST_RADIUS * std::cos(armor.ypr_in_world[0]);
  initial[2] = armor.xyz_in_world.y() + OUTPOST_RADIUS * std::sin(armor.ypr_in_world[0]);
  initial[4] = armor.xyz_in_world.z() - OUTPOST_ARMOR_HEIGHT_OFFSETS[0];
  initial[6] = armor.ypr_in_world[0];
  estimator_ = TargetEstimator(initial, covariance_diagonal, {FilterMethod::EKF});
  current_yaws_[0] = armor.ypr_in_world[0];
  enforce_yaw_rate();
}

std::unique_ptr<OutpostModel> OutpostTarget::clone() const
{
  return std::make_unique<OutpostTarget>(*this);
}

void OutpostTarget::begin_frame()
{
  previous_yaws_ = current_yaws_;
  current_yaws_.fill(std::nullopt);
}

void OutpostTarget::predict(double dt)
{
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(OutpostState::dimension, OutpostState::dimension);
  F(0, 1) = dt;
  F(2, 3) = dt;
  F(4, 5) = dt;
  F(6, 7) = dt;

  const auto a = dt * dt * dt * dt / 4.0;
  const auto b = dt * dt * dt / 2.0;
  const auto c = dt * dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(OutpostState::dimension, OutpostState::dimension);
  for (const auto position : {0, 2, 4}) {
    Q(position, position) = a * config_.accel_var;
    Q(position, position + 1) = b * config_.accel_var;
    Q(position + 1, position) = b * config_.accel_var;
    Q(position + 1, position + 1) = c * config_.accel_var;
  }

  enforce_yaw_rate();
  estimator_.predict_vector(F, Q, [F](const Eigen::VectorXd & state) {
    Eigen::VectorXd predicted = F * state;
    predicted[6] = tools::limit_rad(predicted[6]);
    return predicted;
  });
}

OutpostUpdateResult OutpostTarget::update(const std::vector<Armor> & armors)
{
  OutpostUpdateResult result;
  for (const auto & armor : armors) {
    const auto id = match_armor(armor);
    update(armor, id);
    result.updated = true;
    result.armor_id = id;
    result.armor_ids.push_back(id);
  }
  return result;
}

void OutpostTarget::update(const Armor & armor, int id)
{
  update_direction(id, armor.ypr_in_world[0]);
  const auto current_state = state();
  const auto H = observation_jacobian(current_state, id);
  const auto center_yaw = std::atan2(armor.xyz_in_world.y(), armor.xyz_in_world.x());
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::VectorXd R_diagonal{
    {config_.observation_yaw_var, config_.observation_pitch_var,
     std::log(std::abs(delta_angle) + 1.0) + 1.0,
     std::log(std::abs(armor.ypd_in_world[2]) + 1.0) / 200.0 +
       config_.observation_armor_yaw_base}};

  auto measurement_model = [this, id](const Eigen::VectorXd & values) -> Eigen::Vector4d {
    const OutpostState state(values);
    const auto ypd = tools::xyz2ypd(armor_center(state, id));
    const auto yaw = tools::limit_rad(state.yaw() + id * 2.0 * CV_PI / OUTPOST_ARMOR_COUNT);
    return {ypd[0], ypd[1], ypd[2], yaw};
  };
  auto residual = [](const Eigen::VectorXd & observation, const Eigen::VectorXd & prediction) {
    Eigen::VectorXd result = observation - prediction;
    result[0] = tools::limit_rad(result[0]);
    result[1] = tools::limit_rad(result[1]);
    result[3] = tools::limit_rad(result[3]);
    return result;
  };
  const Eigen::VectorXd observation{
    {armor.ypd_in_world[0], armor.ypd_in_world[1], armor.ypd_in_world[2],
     armor.ypr_in_world[0]}};
  estimator_.update_vector(observation, H, R_diagonal.asDiagonal(), measurement_model, residual);
  constrain_velocity();
  enforce_yaw_rate();
}

OutpostState OutpostTarget::state() const { return OutpostState(estimator_.state_vector()); }

std::optional<OutpostState> OutpostTarget::outpost_state() const { return state(); }

TargetState OutpostTarget::compatibility_state() const
{
  TargetState result;
  const auto outpost_state = state();
  result.set_center_x(outpost_state.center_x());
  result.set_velocity_x(outpost_state.velocity_x());
  result.set_center_y(outpost_state.center_y());
  result.set_velocity_y(outpost_state.velocity_y());
  result.set_center_z(outpost_state.center_z());
  result.set_velocity_z(outpost_state.velocity_z());
  result.set_yaw(outpost_state.yaw());
  result.set_yaw_rate(outpost_state.yaw_rate());
  result.set_radius(OUTPOST_RADIUS);
  result.set_radius_diff(0.0);
  result.set_height_diff(0.0);
  return result;
}

Eigen::VectorXd OutpostTarget::state_vector() const { return estimator_.state_vector(); }

std::vector<PredictedArmorPose> OutpostTarget::armor_pose_list() const
{
  std::vector<PredictedArmorPose> result;
  result.reserve(OUTPOST_ARMOR_COUNT);
  const auto current_state = state();
  for (int id = 0; id < OUTPOST_ARMOR_COUNT; ++id) {
    const auto yaw =
      tools::limit_rad(current_state.yaw() + id * 2.0 * CV_PI / OUTPOST_ARMOR_COUNT);
    result.push_back({armor_center(current_state, id), yaw, OUTPOST_MOUNT_PITCH});
  }
  return result;
}

double OutpostTarget::last_nis() const { return estimator_.last_nis(); }

const TargetEstimatorDiagnostics & OutpostTarget::diagnostics() const
{
  return estimator_.diagnostics();
}

bool OutpostTarget::has_bad_nis_convergence(double failure_rate) const
{
  return estimator_.has_bad_nis_convergence(failure_rate);
}

bool OutpostTarget::direction_locked() const { return direction_ != 0; }

bool OutpostTarget::all_finite() const { return estimator_.state_vector().allFinite(); }

int OutpostTarget::match_armor(const Armor & armor) const
{
  const auto poses = armor_pose_list();
  auto best_id = 0;
  auto min_error = std::numeric_limits<double>::infinity();
  for (int id = 0; id < static_cast<int>(poses.size()); ++id) {
    const auto ypd = tools::xyz2ypd(poses[id].center);
    const auto error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - poses[id].yaw)) +
      std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    if (error < min_error) {
      min_error = error;
      best_id = id;
    }
  }
  return best_id;
}

Eigen::Vector3d OutpostTarget::armor_center(const OutpostState & state, int id) const
{
  const auto yaw = tools::limit_rad(state.yaw() + id * 2.0 * CV_PI / OUTPOST_ARMOR_COUNT);
  return {
    state.center_x() - OUTPOST_RADIUS * std::cos(yaw),
    state.center_y() - OUTPOST_RADIUS * std::sin(yaw),
    state.center_z() + OUTPOST_ARMOR_HEIGHT_OFFSETS[id]};
}

Eigen::MatrixXd OutpostTarget::observation_jacobian(const OutpostState & state, int id) const
{
  const auto yaw = tools::limit_rad(state.yaw() + id * 2.0 * CV_PI / OUTPOST_ARMOR_COUNT);
  Eigen::MatrixXd armor_jacobian =
    Eigen::MatrixXd::Zero(4, OutpostState::dimension);
  armor_jacobian(0, 0) = 1.0;
  armor_jacobian(1, 2) = 1.0;
  armor_jacobian(2, 4) = 1.0;
  armor_jacobian(3, 6) = 1.0;
  armor_jacobian(0, 6) = OUTPOST_RADIUS * std::sin(yaw);
  armor_jacobian(1, 6) = -OUTPOST_RADIUS * std::cos(yaw);

  Eigen::Matrix4d coordinate_jacobian = Eigen::Matrix4d::Zero();
  coordinate_jacobian.topLeftCorner<3, 3>() =
    tools::xyz2ypd_jacobian(armor_center(state, id));
  coordinate_jacobian(3, 3) = 1.0;
  return coordinate_jacobian * armor_jacobian;
}

void OutpostTarget::update_direction(int id, double observed_yaw)
{
  if (id < 0 || id >= OUTPOST_ARMOR_COUNT) return;
  if (!current_yaws_[id] && previous_yaws_[id]) {
    const auto delta = tools::limit_rad(observed_yaw - *previous_yaws_[id]);
    if (std::abs(delta) > 1e-6) {
      direction_vote_ = std::clamp(direction_vote_ + (delta > 0.0 ? 1 : -1), -3, 3);
      if (direction_vote_ == 3) direction_ = 1;
      if (direction_vote_ == -3) direction_ = -1;
    }
  }
  current_yaws_[id] = observed_yaw;
}

void OutpostTarget::enforce_yaw_rate()
{
  Eigen::MatrixXd transform =
    Eigen::MatrixXd::Identity(OutpostState::dimension, OutpostState::dimension);
  transform.row(7).setZero();
  estimator_.transform_state(transform);
  auto current_state = estimator_.state_vector();
  current_state[6] = tools::limit_rad(current_state[6]);
  current_state[7] = direction_ * OUTPOST_ANGULAR_SPEED;
  estimator_.set_state_vector(current_state);
}

void OutpostTarget::constrain_velocity()
{
  if (!config_.velocity_clamp_enabled) return;
  auto current_state = estimator_.state_vector();
  for (const auto index : {1, 3, 5}) {
    current_state[index] = std::clamp(
      current_state[index], -config_.max_linear_speed, config_.max_linear_speed);
  }
  estimator_.set_state_vector(current_state);
}

}  // namespace auto_aim
