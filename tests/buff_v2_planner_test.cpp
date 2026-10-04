#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_buff_v2/planners/mpc_buff_tracking.hpp"
#include "tools/ballistic_solver.hpp"

static_assert(std::is_same_v<
              decltype(std::declval<auto_buff_v2::MpcBuffTracking>().prepare(
                std::uint64_t{}, std::optional<auto_buff_v2::RuneEstimate>{}, 0.0,
                auto_buff_v2::Timestamp{})),
              std::optional<auto_buff_v2::MpcBuffTrackingRequest>>);

int main()
{
  using namespace std::chrono_literals;
  const auto start = std::chrono::steady_clock::now();
  auto_buff_v2::RuneEstimate target;
  target.center = {3, 0, 1};
  target.start_timestamp = start - 4s;
  target.timestamp = start;
  target.inactive[0] = true;
  auto_buff_v2::MpcBuffTracking::Config config;
  config.aimer_parameters = YAML::Load("{njust: {shoot_delay: 0}}");
  config.ballistic_model = "vacuum";
  const char * mpc_config_path = "/tmp/buff_v2_mpc_test.yaml";
  std::ofstream(mpc_config_path) <<
    "ballistic_model: vacuum\n"
    "yaw_offset: 0\n"
    "pitch_offset: 0\n"
    "fire_thresh: 0.0035\n"
    "decision_speed: 7\n"
    "high_speed_delay_time: 0\n"
    "low_speed_delay_time: 0\n"
    "rho: 1.0\n"
    "max_iter: 10\n"
    "bullet_speed_min: 10\n"
    "bullet_speed_max: 25\n"
    "bullet_speed_default: 23.4\n"
    "max_yaw_acc: 50\n"
    "Q_yaw: [9e6, 0]\n"
    "R_yaw: [1]\n"
    "max_pitch_acc: 100\n"
    "Q_pitch: [9e6, 0]\n"
    "R_pitch: [1]\n";
  auto_aim::Planner mpc_planner(mpc_config_path);
  auto_buff_v2::MpcBuffTracking planner(config);
  const auto request = planner.prepare(1, target, 20, start);
  assert(request);
  assert(std::abs(request->distance - target.center.head<2>().norm()) < 1e-12);
  // A stationary blade is at z=1.7: report its ballistic time, not center.norm()/speed.
  assert(std::abs(request->fly_time - 0.176280533350353) < 1e-9);
  for (const auto [height, expected_time] :
       {std::pair{0.0, 0.155435314095256}, std::pair{4.0, 0.294925697745734},
        std::pair{-4.0, 0.216191845381263}}) {
    auto height_target = target;
    height_target.center.z() = height;
    height_target.rotation_speed = 1.0;
    auto height_config = config;
    height_config.aimer_parameters = YAML::Load("{njust: {shoot_delay: 0.1}}");
    height_config.ballistic_model = "vacuum";
    auto_buff_v2::MpcBuffTracking height_planner(height_config);
    const auto height_request = height_planner.prepare(1, height_target, 20, start);
    assert(height_request);
    assert(std::abs(height_request->fly_time - expected_time) < 1e-9);
  }
  const auto mpc_plan = mpc_planner.plan(request->trajectory, request->yaw0, request->distance);
  assert(mpc_plan.control && mpc_plan.debug_valid && !mpc_plan.fire);
  io::GimbalState gimbal;
  assert(!planner.fire_advice(*request, mpc_plan, gimbal, start));
  gimbal.yaw = mpc_plan.yaw;
  gimbal.pitch = mpc_plan.pitch;
  assert(planner.fire_advice(*request, mpc_plan, gimbal, start + 450ms));
  gimbal.yaw = mpc_plan.yaw + 0.05;
  assert(!planner.fire_advice(*request, mpc_plan, gimbal, start + 470ms));
  gimbal.yaw = mpc_plan.yaw;
  gimbal.pitch = mpc_plan.pitch + 0.03;
  assert(!planner.fire_advice(*request, mpc_plan, gimbal, start + 480ms));

  gimbal.pitch = mpc_plan.pitch;
  assert(!planner.prepare(1, std::nullopt, 20, start + 460ms));
  const auto reacquired = planner.prepare(2, target, 20, start + 470ms);
  assert(reacquired);
  const auto reacquired_plan =
    mpc_planner.plan(reacquired->trajectory, reacquired->yaw0, reacquired->distance);
  assert(reacquired_plan.control);
  assert(!planner.fire_advice(*reacquired, reacquired_plan, gimbal, start + 470ms));

  target.inactive.fill(false);
  assert(!planner.prepare(2, target, 20, start + 500ms));

  target.inactive[0] = true;
  target.start_timestamp = start - 7s;
  target.sine_valid = true;
  target.sine_v = 1;
  target.sine_a = 0.8;
  target.sine_omega = 2;
  target.sine_phase = 0.2;
  const auto accelerating_request = planner.prepare(2, target, 20, start + 510ms);
  assert(accelerating_request);
  const auto accelerating = mpc_planner.plan(
    accelerating_request->trajectory, accelerating_request->yaw0,
    accelerating_request->distance);
  assert(accelerating.control);
  assert(std::abs(accelerating.pitch_acc) > 0.01);

  const auto njust = tools::make_ballistic_solver("njust");
  const auto vacuum = tools::make_ballistic_solver("vacuum");
  const auto njust_solution = njust->solve(20, 5, 0);
  const auto vacuum_solution = vacuum->solve(20, 5, 0);
  assert(njust_solution);
  assert(vacuum_solution);
  assert(njust_solution->fly_time > vacuum_solution->fly_time);
  assert(njust_solution->pitch > vacuum_solution->pitch);

  tools::BallisticSolverConfig low_drag_config;
  low_drag_config.njust_air_resistance = 0.001;
  tools::BallisticSolverConfig high_drag_config;
  high_drag_config.njust_air_resistance = 0.006;
  const auto low_drag = tools::make_ballistic_solver("njust", low_drag_config);
  const auto high_drag = tools::make_ballistic_solver("njust", high_drag_config);
  const auto low_drag_solution = low_drag->solve(20, 5, 0);
  const auto high_drag_solution = high_drag->solve(20, 5, 0);
  assert(low_drag_solution && high_drag_solution);
  assert(high_drag_solution->fly_time > low_drag_solution->fly_time);
  assert(high_drag_solution->pitch > low_drag_solution->pitch);

  auto_buff_v2::MpcBuffTracking::Config njust_config;
  njust_config.ballistic_model = "njust";
  njust_config.ballistic_config.njust_air_resistance = 0.001;
  auto_buff_v2::MpcBuffTracking::Config high_drag_planner_config = njust_config;
  high_drag_planner_config.ballistic_config.njust_air_resistance = 0.006;
  auto_buff_v2::MpcBuffTracking::Config vacuum_config = njust_config;
  vacuum_config.ballistic_model = "vacuum";
  auto_buff_v2::MpcBuffTracking njust_planner(njust_config);
  auto_buff_v2::MpcBuffTracking high_drag_planner(high_drag_planner_config);
  auto_buff_v2::MpcBuffTracking vacuum_planner(vacuum_config);
  const auto njust_request = njust_planner.prepare(1, target, 20, start);
  const auto high_drag_request = high_drag_planner.prepare(1, target, 20, start);
  const auto vacuum_request = vacuum_planner.prepare(1, target, 20, start);
  assert(njust_request && high_drag_request && vacuum_request);
  const auto njust_plan =
    mpc_planner.plan(njust_request->trajectory, njust_request->yaw0, njust_request->distance);
  const auto high_drag_plan = mpc_planner.plan(
    high_drag_request->trajectory, high_drag_request->yaw0, high_drag_request->distance);
  const auto vacuum_plan =
    mpc_planner.plan(vacuum_request->trajectory, vacuum_request->yaw0, vacuum_request->distance);
  assert(njust_plan.control && high_drag_plan.control && vacuum_plan.control);
  assert(high_drag_plan.pitch < njust_plan.pitch);
  assert(njust_plan.pitch < vacuum_plan.pitch);

  bool invalid_model_rejected = false;
  try {
    static_cast<void>(tools::make_ballistic_solver("invalid"));
  } catch (const std::invalid_argument &) {
    invalid_model_rejected = true;
  }
  assert(invalid_model_rejected);
}
