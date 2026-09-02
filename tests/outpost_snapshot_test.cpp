#include <cassert>

#include "tasks/auto_aim/outpost_target.hpp"
#include "tasks/auto_aim/outpost_target_v2.hpp"

namespace
{

auto_aim::Armor outpost_armor()
{
  auto_aim::Armor armor(
    19, 0.99F, cv::Rect{}, {{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}});
  armor.xyz_in_world = {2.0, 0.0, 0.3};
  armor.ypr_in_world = {0.0, 0.0, 0.0};
  armor.ypd_in_world = {0.0, 0.0, armor.xyz_in_world.norm()};
  return armor;
}

}  // namespace

int main()
{
  const auto armor = outpost_armor();
  const auto covariance = Eigen::VectorXd::Ones(auto_aim::OutpostState::dimension);
  const auto_aim::OutpostTarget current(armor, covariance, {});
  const auto current_snapshot = current.snapshot();
  assert(std::holds_alternative<auto_aim::OutpostState>(current_snapshot.debug_state));
  assert(current_snapshot.state_vector.size() == auto_aim::OutpostState::dimension);
  assert(current_snapshot.armor_poses.size() == 3);
  assert(current_snapshot.compatibility_state.all_finite());

  const auto_aim::OutpostTargetV2 v2({armor}, Eigen::VectorXd::Ones(10), {});
  const auto v2_snapshot = v2.snapshot();
  assert(std::holds_alternative<auto_aim::OutpostStateV2>(v2_snapshot.debug_state));
  assert(v2_snapshot.state_vector.size() == auto_aim::OutpostStateV2::dimension);
  assert(v2_snapshot.armor_poses.size() == 3);
  assert(v2_snapshot.compatibility_state.all_finite());
}
