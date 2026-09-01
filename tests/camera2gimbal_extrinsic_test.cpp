#include <cassert>
#include <cmath>

#include <yaml-cpp/yaml.h>

#include "tools/camera2gimbal_extrinsic.hpp"

namespace
{
void expect_near(double actual, double expected)
{
  assert(std::abs(actual - expected) < 1e-9);
}

void expect_rotation(
  const Eigen::Matrix3d & actual, const Eigen::Matrix3d & expected, double tolerance = 1e-9)
{
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      assert(std::abs(actual(row, col) - expected(row, col)) < tolerance);
    }
  }
}

void test_loads_legacy_matrix_extrinsic_by_default()
{
  const auto yaml = YAML::Load(R"(
R_camera2gimbal: [0, -1, 0, 1, 0, 0, 0, 0, 1]
t_camera2gimbal: [0.1, -0.2, 0.3]
)");

  const auto extrinsic = tools::load_camera2gimbal_extrinsic(yaml);

  expect_near(extrinsic.rotation(0, 1), -1.0);
  expect_near(extrinsic.rotation(1, 0), 1.0);
  expect_near(extrinsic.translation.x(), 0.1);
  expect_near(extrinsic.translation.y(), -0.2);
  expect_near(extrinsic.translation.z(), 0.3);
}

void test_loads_xyz_ypr_extrinsic_when_selected()
{
  const auto yaml = YAML::Load(R"(
camera2gimbal_mode: xyz_ypr
camera2gimbal_xyz: [0.1, -0.2, 0.3]
camera2gimbal_ypr: [1.5707963267948966, 0, 0]
)");

  const auto extrinsic = tools::load_camera2gimbal_extrinsic(yaml);

  const Eigen::Matrix3d expected_rotation{
    {1, 0, 0},
    {0, 0, 1},
    {0, -1, 0}};
  expect_rotation(extrinsic.rotation, expected_rotation);
  expect_near(extrinsic.translation.x(), 0.1);
  expect_near(extrinsic.translation.y(), -0.2);
  expect_near(extrinsic.translation.z(), 0.3);
}

void test_standard3_xyz_ypr_matches_hfut_camera_to_barrel_extrinsic()
{
  const auto extrinsic = tools::load_camera2gimbal_extrinsic(YAML::LoadFile("configs/standard3.yaml"));
  const Eigen::Matrix3d expected_rotation{
    {0, 0, 1},
    {-1, 0, 0},
    {0, -1, 0}};

  expect_rotation(extrinsic.rotation, expected_rotation, 1e-10);
  expect_near(extrinsic.translation.x(), 0.11593);
  expect_near(extrinsic.translation.y(), -0.06429);
  expect_near(extrinsic.translation.z(), 0.03250);
}
}  // namespace

int main()
{
  test_loads_legacy_matrix_extrinsic_by_default();
  test_loads_xyz_ypr_extrinsic_when_selected();
  test_standard3_xyz_ypr_matches_hfut_camera_to_barrel_extrinsic();
}
