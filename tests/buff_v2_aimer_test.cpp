#include <cassert>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "tasks/auto_buff_v2/aimers/rune_aimer_factory.hpp"
#include "tools/ballistic_solver.hpp"

int main()
{
  using namespace std::chrono_literals;
  static_assert(std::is_abstract_v<auto_buff_v2::RuneAimer>);
  const auto now = auto_buff_v2::Timestamp{} + 10s;
  auto_buff_v2::RuneEstimate target;
  target.center = {3, 0, 0};
  target.timestamp = now;
  target.start_timestamp = now - 7s;
  target.inactive[0] = true;
  auto_buff_v2::BuffConfig::Planner config;
  config.ballistic.type = "vacuum";
  io::GimbalState gimbal;
  auto njust = auto_buff_v2::make_rune_aimer(config);
  auto aimed = njust->aim(1, target, 20, gimbal, now);
  assert(aimed && aimed->plan.control);
  assert(std::abs(aimed->plan.fly_time - 0.155479363125162) < 1e-9);
  assert(!njust->fire_advice(target.center, aimed->plan.debug_xyza.head<3>(),
                            aimed->plan, gimbal, now));
  gimbal.yaw = aimed->plan.yaw;
  gimbal.pitch = aimed->plan.pitch;
  assert(njust->fire_advice(target.center, aimed->plan.debug_xyza.head<3>(),
                           aimed->plan, gimbal, now + 450ms));

  config.aimer = "climber";
  config.ballistic.type = "climber";
  auto climber = auto_buff_v2::make_rune_aimer(config);
  aimed = climber->aim(1, target, 0, gimbal, now);
  // Climber uses the 24 m/s fallback and a two-pass vacuum solution.
  assert(aimed && aimed->plan.control);  // Climber keeps control during a blade switch.
  assert(aimed->plan.yaw_vel == 0 && aimed->plan.pitch_vel == 0);
  aimed = climber->aim(1, target, 0, gimbal, now + 10ms);
  assert(aimed && aimed->plan.control);
  assert(std::abs(aimed->plan.fly_time - 0.129172918752555) < 1e-9);
  assert(!climber->fire_advice(target.center, aimed->plan.debug_xyza.head<3>(),
                             aimed->plan, gimbal, now + 10ms));
  assert(climber->fire_advice(target.center, aimed->plan.debug_xyza.head<3>(),
                            aimed->plan, gimbal, now + 800ms));
  assert(!climber->fire_advice(target.center, aimed->plan.debug_xyza.head<3>(),
                             aimed->plan, gimbal, now + 810ms));
  assert(!climber->aim(1, std::nullopt, 20, gimbal, now + 820ms));
  aimed = climber->aim(2, target, 20, gimbal, now + 830ms);
  assert(aimed && aimed->plan.control);

  auto fast = climber->aim(3, target, 29, gimbal, now);
  assert(fast && fast->bullet_speed == 24);
  assert(std::abs(fast->solution.fly_time - 0.129172918752555) < 1e-9);

  auto_buff_v2::BuffConfig::Planner drag_config;
  drag_config.aimer = "climber";
  drag_config.ballistic.type = "climber";
  drag_config.ballistic.climber_air_resistance_k = 0.02;
  auto drag = auto_buff_v2::make_rune_aimer(drag_config);
  const auto drag_aim = drag->aim(1, target, 20, gimbal, now);
  assert(drag_aim && drag_aim->plan.control);
  // Independent linear-drag solution: v=20, d=3, h=0.7, k=0.02, g=9.7833.
  assert(std::abs(drag_aim->solution.angles.y() + 0.266350973473905) < 1e-9);
  assert(std::abs(drag_aim->solution.fly_time - 0.155724913567701) < 1e-9);

  const auto solver = tools::make_ballistic_solver(
    drag_config.ballistic.type,
    {drag_config.ballistic.njust_air_resistance, drag_config.ballistic.climber_air_resistance_k});
  const auto direct = solver->solve(20, 3, 0.7);
  assert(direct);
  assert(std::abs(direct->pitch - 0.266350973473905) < 1e-9);
  assert(std::abs(direct->fly_time - 0.155724913567701) < 1e-9);
  assert(!solver->solve(20, 0.09, 0));
  assert(!solver->solve(20, 3, 10));  // Outside Climber's +/-60 degree bound.
  assert(!solver->solve(20, 1000, 1000));
  auto invalid_ballistic = drag_config.ballistic;
  invalid_ballistic.climber_air_resistance_k = -1;
  bool invalid_drag_rejected = false;
  try {
    tools::make_ballistic_solver(
      invalid_ballistic.type,
      {invalid_ballistic.njust_air_resistance, invalid_ballistic.climber_air_resistance_k});
  } catch (const std::invalid_argument &) {
    invalid_drag_rejected = true;
  }
  assert(invalid_drag_rejected);

  auto predicted = target;
  predicted.predicted = true;
  auto predicted_aimer = auto_buff_v2::make_rune_aimer(config);
  assert(predicted_aimer->aim(1, predicted, 20, gimbal, now));
  const auto predicted_aim = predicted_aimer->aim(1, predicted, 20, gimbal, now + 800ms);
  assert(predicted_aim && predicted_aim->plan.control);
  assert(!predicted_aimer->fire_advice(
    predicted.center, predicted_aim->solution.point, predicted_aim->plan, gimbal, now + 800ms));

  // Independent two-pass vacuum fixture: 3 m center, 0.7 m radius, 1 rad/s.
  auto moving = target;
  moving.center.z() = 1;
  moving.rotation_speed = 1;
  auto moving_aimer = auto_buff_v2::make_rune_aimer(config);
  const auto moving_aim = moving_aimer->aim(1, moving, 20, gimbal, now);
  assert(moving_aim);
  assert(std::abs(moving_aim->plan.yaw + 0.068000890410280) < 1e-8);
  assert(std::abs(moving_aim->plan.pitch + 0.544433913622821) < 1e-7);  // Plan stores float angles.
  assert(std::abs(moving_aim->plan.fly_time - 0.175758538515391) < 1e-8);
  assert(moving.timestamp == now && moving.rotation_angle == 0);
  // Backward MPC sample starts from the original state at timestamp - predict_time.
  const auto moving_again = moving_aimer->aim(1, moving, 20, {}, now);
  assert(moving_again && moving_again->plan.control);
  assert(std::abs(moving_again->plan.yaw_vel + 0.228735494896525) < 1e-7);
  assert(std::abs(moving_again->plan.pitch_vel - 0.034675216929050) < 1e-7);
  assert(std::abs(moving_again->plan.pitch_acc + 76.1937416186538) < 1e-5);
  moving.rotation_speed = 100;
  // Climber accepts this 15 ms discrepancy, below its 30 ms threshold.
  assert(moving_aimer->aim(2, moving, 20, gimbal, now));

  moving.center.z() = 2;
  moving.rotation_speed = 20;
  // Independent vacuum fixture has a 34.9 ms discrepancy and must be rejected.
  assert(!moving_aimer->aim(3, moving, 20, gimbal, now));

  auto switching_aimer = auto_buff_v2::make_rune_aimer(config);
  auto switching = target;
  for (int i = 0; i < 5; ++i) {
    switching.center = {0, i % 2 == 0 ? 3.0 : -3.0, 0};
    const auto time = now + i * 10ms;
    const auto switching_aim = switching_aimer->aim(1, switching, 20, gimbal, time);
    assert(switching_aim && switching_aim->plan.control);
    assert(!switching_aimer->fire_advice(
      switching.center, switching_aim->plan.debug_xyza.head<3>(),
      switching_aim->plan, gimbal, time));
  }

  for (const auto & parameters :
       {"{predict_time: 0}", "{fire_gap_time: -1}"}) {
    config.aimer_parameters = YAML::Load(std::string("{climber: ") + parameters + "}");
    bool rejected = false;
    try {
      auto_buff_v2::make_rune_aimer(config);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    assert(rejected);
  }
  config.aimer_parameters = YAML::Node{};

  for (const auto & type : {"njust", "climber"}) {
    config.aimer = type;
    auto aimer = auto_buff_v2::make_rune_aimer(config);
    auto invalid = target;
    invalid.inactive.fill(false);
    assert(!aimer->aim(1, invalid, 20, gimbal, now));
    invalid = target;
    invalid.center = {1000, 0, 1000};
    assert(!aimer->aim(1, invalid, 20, gimbal, now));
    invalid = target;
    invalid.start_timestamp = now;
    assert(!aimer->aim(1, invalid, 20, gimbal, now));
  }
  config.aimer = "unknown";
  bool rejected = false;
  try {
    auto_buff_v2::make_rune_aimer(config);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
}
