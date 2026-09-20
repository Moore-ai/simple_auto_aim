#include <cassert>

#include "tasks/auto_buff_v2/rune_ekf_state.hpp"

int main()
{
  const auto values = (Eigen::Matrix<double, 6, 1>() << 1, 2, 3, 4, 5, 6).finished();
  auto_buff_v2::RuneEkfState state(values);
  assert(state.center_x() == 1);
  assert(state.center_y() == 2);
  assert(state.center_z() == 3);
  assert(state.rotation_speed() == 4);
  assert(state.rotation_angle() == 5);
  assert(state.face_yaw() == 6);

  state.set_center_x(7);
  state.set_rotation_speed(8);
  state.set_face_yaw(9);
  assert(state.center_x() == 7);
  assert(state.rotation_speed() == 8);
  assert(state.face_yaw() == 9);
  assert(state.all_finite());
}
