#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr double EPSILON = 1e-6;

bool near(double lhs, double rhs) { return std::abs(lhs - rhs) < EPSILON; }

std::filesystem::path write_test_config(const std::string & wait_armor, bool speed_enable = true)
{
  static unsigned int config_id = 0;
  auto config = YAML::LoadFile("configs/standard.yaml");
  config["yaw_offset"] = 0.0;
  config["pitch_offset"] = 0.0;
  config["fire_thresh"] = 0.01;
  config["decision_speed_enable"] = speed_enable;
  config["decision_speed"] = 1.0;
  config["speed_hysteresis"] = 0.0;
  config["high_speed_delay_time"] = 0.0;
  config["low_speed_delay_time"] = 0.0;
  config["anti_spin_enable"] = true;
  config["anti_spin_wait_armor"] = wait_armor;
  config["max_iter"] = 100;

  const auto path = std::filesystem::temp_directory_path() /
    ("simple_auto_aim_anti_spin_" + std::to_string(++config_id) + ".yaml");
  std::ofstream output(path);
  assert(output);
  output << config;
  return path;
}

auto_aim::Armor armor_observation(const Eigen::Vector4d & pose)
{
  auto armor = auto_aim::Armor(
    0, 1.0F, cv::Rect{}, std::vector<cv::Point2f>{{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F},
                                                  {0.0F, 1.0F}});
  armor.xyz_in_world = pose.head<3>();
  armor.ypr_in_world = Eigen::Vector3d{pose[3], 0.0, 0.0};
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  return armor;
}
}  // namespace

int main()
{
  constexpr double bullet_speed = 22.0;
  auto_aim::Target target(10.0, 2.0, 1.0, 0.5);

  const auto low_config = write_test_config("low");
  auto_aim::Planner low_planner(low_config.string());
  std::filesystem::remove(low_config);
  const auto low_plan = low_planner.plan(target, bullet_speed);
  assert(low_plan.control);
  assert(near(low_plan.debug_xyza.x(), 9.0));
  assert(near(low_plan.debug_xyza.y(), 0.0));
  assert(near(low_plan.debug_xyza.z(), 0.0));

  auto phase_shifted_target = target;
  phase_shifted_target.predict(0.25);
  const auto phase_shifted_config = write_test_config("low");
  auto_aim::Planner phase_shifted_planner(phase_shifted_config.string());
  std::filesystem::remove(phase_shifted_config);
  const auto phase_shifted_plan = phase_shifted_planner.plan(phase_shifted_target, bullet_speed);
  assert(phase_shifted_plan.control);
  assert(near(phase_shifted_plan.fly_time, low_plan.fly_time));
  assert(near(phase_shifted_plan.target_yaw, low_plan.target_yaw));
  assert(near(phase_shifted_plan.target_pitch, low_plan.target_pitch));

  const auto high_config = write_test_config("high");
  auto_aim::Planner high_planner(high_config.string());
  std::filesystem::remove(high_config);
  const auto high_plan = high_planner.plan(target, bullet_speed);
  assert(high_plan.control);
  assert(near(high_plan.debug_xyza.x(), 9.0));
  assert(near(high_plan.debug_xyza.y(), 0.0));
  assert(near(high_plan.debug_xyza.z(), 0.5));
  assert(near(high_plan.target_yaw, low_plan.target_yaw));
  assert(!near(high_plan.target_pitch, low_plan.target_pitch));

  const auto disabled_config = write_test_config("low", false);
  auto_aim::Planner disabled_planner(disabled_config.string());
  std::filesystem::remove(disabled_config);
  const auto disabled_plan = disabled_planner.plan(target, bullet_speed);
  assert(disabled_plan.control);
  assert(!near(disabled_plan.debug_xyza.x(), 10.0) || !near(disabled_plan.debug_xyza.y(), 0.0));

  auto_aim::Target slow_target(10.0, 0.5, 1.0, 0.5);
  const auto slow_config = write_test_config("low");
  auto_aim::Planner slow_planner(slow_config.string());
  std::filesystem::remove(slow_config);
  const auto slow_plan = slow_planner.plan(slow_target, bullet_speed);
  assert(slow_plan.control);
  assert(!near(slow_plan.debug_xyza.x(), 10.0) || !near(slow_plan.debug_xyza.y(), 0.0));

  auto_aim::Target passing_low_armor(10.0, 2.0, 0.2, 0.5);
  passing_low_armor.predict(-0.45);
  assert(!passing_low_armor.convergened());
  const auto unconverged_config = write_test_config("low");
  auto_aim::Planner unconverged_planner(unconverged_config.string());
  std::filesystem::remove(unconverged_config);
  const auto unconverged_plan = unconverged_planner.plan(passing_low_armor, bullet_speed);
  assert(!unconverged_plan.fire);

  auto converged_passing_target = auto_aim::Target(10.0, 2.0, 0.2, 0.5);
  converged_passing_target.predict(-0.45);
  for (int i = 0; i < 4; ++i) {
    converged_passing_target.update(armor_observation(converged_passing_target.armor_xyza_list()[0]));
  }
  assert(converged_passing_target.convergened());
  const auto passing_config = write_test_config("low");
  auto_aim::Planner passing_planner(passing_config.string());
  std::filesystem::remove(passing_config);
  const auto passing_plan = passing_planner.plan(converged_passing_target, bullet_speed);
  assert(passing_plan.fire);

  auto_aim::Target passing_high_armor = passing_low_armor;
  passing_high_armor.predict(M_PI_2 / 2.0);
  const auto wrong_height_config = write_test_config("low");
  auto_aim::Planner wrong_height_planner(wrong_height_config.string());
  std::filesystem::remove(wrong_height_config);
  const auto wrong_height_plan = wrong_height_planner.plan(passing_high_armor, bullet_speed);
  assert(!wrong_height_plan.fire);

  bool rejected_multiple_heights = false;
  const auto invalid_config = write_test_config("low_and_high");
  try {
    auto_aim::Planner invalid_planner(invalid_config.string());
  } catch (const std::runtime_error &) {
    rejected_multiple_heights = true;
  }
  std::filesystem::remove(invalid_config);
  assert(rejected_multiple_heights);

  return 0;
}
