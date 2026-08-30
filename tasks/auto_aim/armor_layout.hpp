#ifndef AUTO_AIM__ARMOR_LAYOUT_HPP
#define AUTO_AIM__ARMOR_LAYOUT_HPP

#include <Eigen/Dense>
#include <vector>

#include "armor.hpp"
#include "target_state.hpp"

namespace auto_aim
{

// Layout seam: converts a tracked vehicle state into its armor geometry.
// Filtering and observation sources stay outside this module.
class ArmorLayout
{
public:
  ArmorLayout() = default;
  ArmorLayout(ArmorName name, int armor_num) : name_{name}, armor_num_{armor_num} {}

  void configure(ArmorName name, int armor_num)
  {
    name_ = name;
    armor_num_ = armor_num;
  }

  Eigen::Vector3d armor_xyz(const TargetState & state, int id) const;
  Eigen::Matrix<double, 3, TargetState::dimension> armor_xyz_jacobian(
    const TargetState & state, int id) const;
  Eigen::MatrixXd armor_xyza_jacobian(const TargetState & state, int id) const;
  std::vector<Eigen::Vector4d> armor_xyza_list(const TargetState & state) const;
  std::vector<PredictedArmorPose> armor_pose_list(const TargetState & state) const;

private:
  ArmorName name_{ArmorName::not_armor};
  int armor_num_{0};
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_LAYOUT_HPP
