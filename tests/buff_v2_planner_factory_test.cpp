#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "tasks/auto_buff_v2/planners/buff_planner_factory.hpp"

int main()
{
  using namespace std::chrono_literals;
  constexpr char path[] = "/tmp/buff_v2_planner_factory_test.yaml";
  // 打符 MPC 必须能够在没有自瞄规划器配置时运行。
  std::ofstream(path) <<
    "buff_v2:\n"
    "  mpc:\n"
    "    rho: 1.0\n"
    "    max_iter: 10\n"
    "    max_yaw_acc: 50\n"
    "    Q_yaw: [9e6, 0]\n"
    "    R_yaw: [1]\n"
    "    max_pitch_acc: 100\n"
    "    Q_pitch: [9e6, 0]\n"
    "    R_pitch: [1]\n";

  const auto start = std::chrono::steady_clock::now();
  auto_buff_v2::RuneEstimate target;
  target.center = {3, 0, 1};
  target.start_timestamp = start - 4s;
  target.timestamp = start;
  target.inactive[0] = true;
  io::GimbalState gimbal;
  auto_buff_v2::BuffConfig::Planner config;
  config.ballistic.type = "vacuum";

  config.mode = "mpc";
  auto mpc = auto_buff_v2::make_buff_planner(config, path);
  const auto mpc_plan = mpc->plan(1, target, 20, gimbal, start);
  assert(mpc_plan.control && mpc_plan.debug_valid);
  assert(!mpc_plan.fire);
  assert(mpc_plan.njust_aim_debug && !mpc_plan.njust_aim_debug->shoot_phase);
  assert(std::abs(mpc_plan.fly_time - 0.176280533350353) < 1e-9);

  config.mode = "njust";
  auto njust = auto_buff_v2::make_buff_planner(config, path);
  const auto njust_plan = njust->plan(1, target, 20, gimbal, start);
  assert(njust_plan.control);
  assert(!njust_plan.debug_valid);
  assert(!njust_plan.fire);
  assert(njust_plan.njust_aim_debug && !njust_plan.njust_aim_debug->shoot_phase);
  assert(std::abs(njust_plan.fly_time - 0.176280533350353) < 1e-9);
  gimbal.yaw = njust_plan.yaw;
  gimbal.pitch = njust_plan.pitch;
  const auto shooting_plan = njust->plan(1, target, 20, gimbal, start + 450ms);
  assert(shooting_plan.fire);
  auto coast = target;
  coast.predicted = true;
  // Njust 的纯预测状态沿用原有开火判断；SP 的纯预测状态禁止开火。
  const auto coast_plan = njust->plan(1, coast, 20, gimbal, start + 455ms);
  assert(coast_plan.control && coast_plan.fire);
  coast.slot_id = 0;
  const auto sp_coast_plan = njust->plan(1, coast, 20, gimbal, start + 456ms);
  assert(sp_coast_plan.control && !sp_coast_plan.fire);
  assert(shooting_plan.njust_aim_debug && shooting_plan.njust_aim_debug->shoot_phase);
  assert(shooting_plan.njust_aim_debug->yaw_error == 0);
  assert(shooting_plan.njust_aim_debug->pitch_error == 0);
  assert(!njust->plan(1, std::nullopt, 20, gimbal, start + 460ms).njust_aim_debug);
  assert(!njust->plan(1, std::nullopt, 20, gimbal, start + 460ms).control);
  assert(!njust->plan(2, target, 20, gimbal, start + 470ms).fire);

  // 独立真空弹道测试场景：观测距今 80 毫秒，附加延迟 40 毫秒。
  // 高速目标达到五次解算上限；正弦运动目标在两次解算内收敛。
  for (const auto & mode : {"mpc", "njust"}) {
    config.mode = mode;
    auto moving_planner = auto_buff_v2::make_buff_planner(config, path);
    auto moving_target = target;
    moving_target.rotation_speed = 100;
    const auto fast = moving_planner->plan(3, moving_target, 20, gimbal, start + 80ms);
    assert(fast.control);
    assert(std::abs(fast.fly_time - 0.153672682779469) < 1e-9);
    assert(std::abs(fast.debug_xyza.y() + 0.395697924416826) < 1e-8);
    assert(std::abs(fast.debug_xyza.z() - 0.422571950272403) < 1e-8);

    moving_target.start_timestamp = start - 7s;
    moving_target.sine_valid = true;
    moving_target.sine_v = 1;
    moving_target.sine_a = 0.8;
    moving_target.sine_omega = 2;
    moving_target.sine_phase = 0.2;
    const auto sine = moving_planner->plan(4, moving_target, 20, gimbal, start + 80ms);
    assert(sine.control);
    assert(std::abs(sine.fly_time - 0.175304713611082) < 1e-9);
    assert(std::abs(sine.debug_xyza.y() + 0.276559893734488) < 1e-8);
    assert(std::abs(sine.debug_xyza.z() - 1.643051028439865) < 1e-8);
  }

  // 两种规划器均使用所选的 Climber 瞄准器及其回退弹速。
  config.aimer = "climber";
  config.ballistic.type = "climber";
  target.center.z() = 0;
  for (const auto & mode : {"mpc", "njust"}) {
    config.mode = mode;
    auto climber_planner = auto_buff_v2::make_buff_planner(config, path);
    assert(climber_planner->plan(1, target, 0, gimbal, start).control);
    const auto climber_plan = climber_planner->plan(1, target, 0, gimbal, start + 10ms);
    assert(climber_plan.control);
    assert(!climber_plan.njust_aim_debug);
    assert(std::abs(climber_plan.fly_time - 0.129172918752555) < 1e-9);
    assert(!climber_plan.fire);
    assert(climber_planner->plan(1, target, 0, gimbal, start + 800ms).fire);
    assert(!climber_planner->plan(1, std::nullopt, 0, gimbal, start + 810ms).control);
    assert(!climber_planner->plan(2, target, 0, gimbal, start + 820ms).fire);
  }

  config.mode = "unknown";
  bool rejected = false;
  try {
    auto_buff_v2::make_buff_planner(config, path);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}
