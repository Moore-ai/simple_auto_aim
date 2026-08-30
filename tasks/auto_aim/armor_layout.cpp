#include "armor_layout.hpp"

#include <cmath>

#include "tools/math_tools.hpp"

namespace auto_aim
{
Eigen::Vector3d ArmorLayout::armor_xyz(const TargetState & state, int id) const
{
  if (armor_num_ <= 0) return Eigen::Vector3d::Zero();

  const auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
  const auto radius = state.radius(use_l_h);
  return {
    state.center_x() - radius * std::cos(angle),
    state.center_y() - radius * std::sin(angle),
    state.armor_height(use_l_h)};
}

Eigen::Matrix<double, 3, TargetState::dimension> ArmorLayout::armor_xyz_jacobian(
  const TargetState & state, int id) const
{
  Eigen::Matrix<double, 3, TargetState::dimension> result =
    Eigen::Matrix<double, 3, TargetState::dimension>::Zero();
  if (armor_num_ <= 0) return result;

  const auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = armor_num_ == 4 && (id == 1 || id == 3);
  const auto radius = state.radius(use_l_h);
  result(0, static_cast<Eigen::Index>(TargetStateComponent::center_x)) = 1.0;
  result(1, static_cast<Eigen::Index>(TargetStateComponent::center_y)) = 1.0;
  result(2, static_cast<Eigen::Index>(TargetStateComponent::center_z)) = 1.0;
  result(0, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = radius * std::sin(angle);
  result(1, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = -radius * std::cos(angle);
  result(0, static_cast<Eigen::Index>(TargetStateComponent::radius)) = -std::cos(angle);
  result(1, static_cast<Eigen::Index>(TargetStateComponent::radius)) = -std::sin(angle);
  if (use_l_h) {
    result(0, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = -std::cos(angle);
    result(1, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = -std::sin(angle);
    result(2, static_cast<Eigen::Index>(TargetStateComponent::height_diff)) = 1.0;
  }
  return result;
}

Eigen::MatrixXd ArmorLayout::armor_xyza_jacobian(const TargetState & state, int id) const
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(4, TargetState::dimension);
  if (armor_num_ <= 0) return result;

  result.topRows<3>() = armor_xyz_jacobian(state, id);
  result(3, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = 1.0;
  return result;
}

std::vector<Eigen::Vector4d> ArmorLayout::armor_xyza_list(const TargetState & state) const
{
  std::vector<Eigen::Vector4d> result;
  if (armor_num_ <= 0) return result;
  result.reserve(static_cast<std::size_t>(armor_num_));
  for (int id = 0; id < armor_num_; ++id) {
    const auto xyz = armor_xyz(state, id);
    const auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
    result.push_back({xyz.x(), xyz.y(), xyz.z(), angle});
  }
  return result;
}

std::vector<PredictedArmorPose> ArmorLayout::armor_pose_list(const TargetState & state) const
{
  const auto xyza = armor_xyza_list(state);
  std::vector<PredictedArmorPose> result;
  result.reserve(xyza.size());
  const auto pitch = armor_mount_pitch(name_);
  for (const auto & armor : xyza) result.push_back({armor.head<3>(), armor[3], pitch});
  return result;
}
}  // namespace auto_aim
