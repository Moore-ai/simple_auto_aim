#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr double BULLET_SPEED = 22.0;
constexpr double RATE = 0.8 * CV_PI;
constexpr double RADIUS = 0.2765;
const Eigen::Vector2d CENTER{5.0, 1.0};
// Phase IDs deliberately differ from physical height order: medium, high, low.
constexpr std::array<double, 3> HEIGHTS{0.4, 0.502, 0.298};

bool near(double lhs, double rhs, double tolerance = 1e-5)
{
  return std::abs(lhs - rhs) < tolerance;
}

auto_aim::Armor observation(double phase, int id)
{
  auto_aim::Armor armor(
    19, 0.99F, cv::Rect{}, {{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}});
  const auto yaw = phase + id * 2.0 * CV_PI / 3.0;
  armor.xyz_in_world = {
    CENTER.x() - RADIUS * std::cos(yaw), CENTER.y() - RADIUS * std::sin(yaw), HEIGHTS[id]};
  armor.ypr_in_world = {tools::limit_rad(yaw), 0.0, 0.0};
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  return armor;
}

auto_aim::Target make_target(int frames, bool all_heights = true)
{
  auto_aim::OutpostTargetV2Config config;
  config.direction_sample_count = 3;
  config.direction_compare_interval = 1;
  config.frontal_angle_gate = CV_PI;
  config.q_xy = 0.0;
  config.q_z = 0.0;
  const auto initial_phase = std::atan2(CENTER.y(), CENTER.x());
  auto target = auto_aim::Target::make_outpost_v2(
    {observation(initial_phase, 0)}, std::chrono::steady_clock::now(),
    Eigen::VectorXd{{0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1}}, config);
  for (int frame = 1; frame <= frames; ++frame) {
    target.predict(0.01);
    const auto phase = initial_phase + RATE * 0.01 * frame;
    int id = 0;
    if (all_heights) {
      double error = CV_PI;
      for (int candidate = 0; candidate < 3; ++candidate) {
        const auto angle = std::abs(tools::limit_rad(
          phase + candidate * 2.0 * CV_PI / 3.0 - initial_phase));
        if (angle < error) {
          error = angle;
          id = candidate;
        }
      }
    }
    assert(target.update_outpost({observation(phase, id)}));
  }
  return target;
}

auto_aim::Plan plan(
  const auto_aim::Target & target, const std::string & height, bool enabled = true,
  bool anti_spin = false, double fire_gate = 0.002, bool iterate = false)
{
  static unsigned int config_id = 0;
  auto config = YAML::LoadFile("configs/standard.yaml");
  config["yaw_offset"] = 0.0;
  config["pitch_offset"] = 0.0;
  config["fire_thresh"] = 0.01;
  config["decision_speed_enable"] = anti_spin;
  config["decision_speed"] = 1.0;
  config["speed_hysteresis"] = 0.0;
  config["anti_spin_enable"] = anti_spin;
  config["anti_spin_wait_armor"] = "high";
  config["outpost_wait_enable"] = enabled;
  config["outpost_wait_armor"] = height;
  config["outpost_wait_fire_thresh"] = fire_gate;
  config["fly_time_iteration"]["enable"] = iterate;
  config["max_iter"] = 100;
  const auto path = std::filesystem::temp_directory_path() /
    ("simple_auto_aim_outpost_wait_" + std::to_string(++config_id) + ".yaml");
  {
    std::ofstream output(path);
    assert(output);
    output << config;
  }
  try {
    auto_aim::Planner planner(path.string());
    std::filesystem::remove(path);
    return planner.plan(target, BULLET_SPEED);
  } catch (...) {
    std::filesystem::remove(path);
    throw;
  }
}
}  // namespace

int main()
{
  auto target = make_target(450);
  assert(target.convergened());
  assert(target.all_armor_ids_seen());
  const auto armors = target.armor_xyza_list();
  for (int id = 0; id < 3; ++id) assert(near(armors[id].z(), HEIGHTS[id], 0.01));

  const auto center_yaw = std::atan2(1.0, 5.0);
  const auto passing_distance = std::sqrt(26.0) - RADIUS;
  const std::array<std::string, 3> labels{"medium", "high", "low"};
  for (int id = 0; id < 3; ++id) {
    const auto result = plan(target, labels[id]);
    assert(result.control);
    assert(result.outpost_wait_active);
    assert(near(result.target_yaw, center_yaw));
    assert(near(result.debug_xyza.head<2>().norm(), passing_distance));
    assert(near(result.debug_xyza.z(), armors[id].z()));
    assert(near(result.yaw_vel, 0.0));
    assert(near(result.pitch_vel, 0.0));

    auto shifted = target;
    shifted.predict(0.3);
    const auto shifted_plan = plan(shifted, labels[id], true, false, 0.002, true);
    assert(near(shifted_plan.target_yaw, result.target_yaw));
    assert(near(shifted_plan.target_pitch, result.target_pitch));
    assert(near(shifted_plan.fly_time, result.fly_time));

    // At impact plus the existing two-step shoot offset, only this phase is near-side frontal.
    auto passing = target;
    const auto phase = target.state().yaw();
    const auto dt = tools::limit_rad(center_yaw - phase - id * 2.0 * CV_PI / 3.0) / RATE -
      result.fly_time - 0.02;
    passing.predict(dt);
    assert(plan(passing, labels[id]).fire);
    assert(!plan(passing, labels[(id + 1) % 3]).fire);
    assert(!plan(passing, labels[id], true, false, 0.0).fire);

    auto rear = passing;
    rear.predict(CV_PI / RATE);
    assert(!plan(rear, labels[id]).fire);
  }

  auto unconverged = make_target(0);
  for (int id = 0; id < 3; ++id) {
    assert(unconverged.update_outpost({observation(center_yaw, id)}));
  }
  assert(unconverged.all_armor_ids_seen());
  assert(!unconverged.convergened());
  const auto waiting = plan(unconverged, "low");
  assert(waiting.control);
  assert(waiting.outpost_wait_active);
  assert(!waiting.fire);
  assert(near(waiting.target_yaw, center_yaw));
  assert(near(waiting.debug_xyza.z(), unconverged.state().center_z()));

  auto incomplete = make_target(20, false);
  assert(incomplete.convergened());
  assert(!incomplete.all_armor_ids_seen());
  const auto incomplete_plan = plan(incomplete, "high");
  assert(incomplete_plan.control);
  assert(!incomplete_plan.fire);
  assert(near(incomplete_plan.debug_xyza.z(), incomplete.state().center_z()));

  const auto ordinary = plan(target, "low", false);
  assert(ordinary.control);
  assert(!ordinary.outpost_wait_active);
  assert(!near(ordinary.target_yaw, center_yaw));
  const auto outpost_both_enabled = plan(target, "low", true, true);
  assert(near(outpost_both_enabled.debug_xyza.z(), armors[2].z()));
  assert(!outpost_both_enabled.anti_spin_active);
  assert(outpost_both_enabled.outpost_wait_active);
  const auto outpost_anti_spin_only = plan(target, "low", false, true);
  assert(!outpost_anti_spin_only.anti_spin_active);
  assert(near(outpost_anti_spin_only.target_yaw, ordinary.target_yaw));
  assert(near(outpost_anti_spin_only.target_pitch, ordinary.target_pitch));

  auto_aim::Target non_outpost(5.0, 2.0, 0.2, 0.1);
  const auto normal = plan(non_outpost, "low");
  const auto disabled = plan(non_outpost, "low", false);
  assert(near(normal.target_yaw, disabled.target_yaw));
  assert(near(normal.target_pitch, disabled.target_pitch));
  const auto normal_both_enabled = plan(non_outpost, "low", true, true);
  const auto normal_anti_spin_only = plan(non_outpost, "low", false, true);
  assert(normal_both_enabled.anti_spin_active);
  assert(!normal_both_enabled.outpost_wait_active);
  assert(normal_anti_spin_only.anti_spin_active);
  assert(near(normal_both_enabled.target_yaw, normal_anti_spin_only.target_yaw));
  assert(near(normal_both_enabled.target_pitch, normal_anti_spin_only.target_pitch));

  bool rejected = false;
  try {
    plan(target, "all");
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  assert(rejected);
}
