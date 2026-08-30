#include <cassert>
#include <cmath>

#include "tasks/auto_aim/armor_layout.hpp"

int main()
{
  auto_aim::TargetState state;
  state.set_center_x(1.0);
  state.set_center_y(2.0);
  state.set_center_z(3.0);
  state.set_yaw(0.2);
  state.set_radius(0.4);
  state.set_radius_diff(0.1);
  state.set_height_diff(0.05);

  const auto layout = auto_aim::ArmorLayout(auto_aim::ArmorName::four, 4);
  const auto armors = layout.armor_xyza_list(state);
  assert(armors.size() == 4);
  assert(std::abs(armors[0].x() - (1.0 - 0.4 * std::cos(0.2))) < 1e-12);
  assert(std::abs(armors[0].y() - (2.0 - 0.4 * std::sin(0.2))) < 1e-12);
  assert(std::abs(armors[0].z() - 3.0) < 1e-12);
  assert(std::abs(armors[1].x() - (1.0 - 0.5 * std::cos(0.2 + CV_PI / 2))) < 1e-12);
  assert(std::abs(armors[1].z() - 3.05) < 1e-12);

  const auto jacobian = layout.armor_xyza_jacobian(state, 1);
  assert(jacobian.rows() == 4 && jacobian.cols() == auto_aim::TargetState::dimension);
  assert(std::abs(
           jacobian(3, static_cast<Eigen::Index>(auto_aim::TargetStateComponent::yaw)) - 1.0) <
         1e-12);
  assert(jacobian.allFinite());
  return 0;
}
