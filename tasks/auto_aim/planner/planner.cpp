#include "planner.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
bool finite_solver_result(const TinySolver * solver)
{
  return solver && solver->solution && solver->work && solver->solution->x.allFinite() &&
    solver->solution->u.allFinite() && solver->work->x.allFinite() && solver->work->u.allFinite();
}

bool valid_plan(const Plan & plan)
{
  return std::isfinite(plan.target_yaw) && std::isfinite(plan.target_pitch) &&
    std::isfinite(plan.yaw) && std::isfinite(plan.yaw_vel) && std::isfinite(plan.yaw_acc) &&
    std::isfinite(plan.pitch) && std::isfinite(plan.pitch_vel) && std::isfinite(plan.pitch_acc) &&
    plan.debug_xyza.allFinite() && std::isfinite(plan.fly_time);
}

Plan invalid_plan(const char * reason)
{
  tools::logger()->warn("[Planner] MPC plan rejected: {}", reason);
  return {};
}
}  // namespace

Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");
  if (yaml["decision_speed_enable"]) decision_speed_enable_ = yaml["decision_speed_enable"].as<bool>();
  if (yaml["extra_delay"]) extra_delay_ = yaml["extra_delay"].as<double>();
  if (yaml["speed_hysteresis"]) speed_hysteresis_ = yaml["speed_hysteresis"].as<double>();
  if (const auto selection_yaml = yaml["armor_selection_hysteresis"]; selection_yaml) {
    ArmorSelectionHysteresisConfig selection_config;
    selection_config.enable = tools::read<bool>(selection_yaml, "enable");
    selection_config.switch_margin = tools::read<double>(selection_yaml, "switch_margin");
    selection_config.switch_confirm_frames =
      tools::read<int>(selection_yaml, "switch_confirm_frames");
    armor_selection_hysteresis_enabled_ = selection_config.enable;
    armor_selector_ = ArmorSelectionHysteresis(selection_config);
  }
  if (decision_speed_enable_) {
    tools::logger()->info("[Planner] decision speed delay: enable=true, center={:.1f}, hyst={:.1f}, low={:.3f}, high={:.3f}",
                          decision_speed_, speed_hysteresis_, low_speed_delay_time_, high_speed_delay_time_);
  } else {
    tools::logger()->info("[Planner] decision speed delay: enable=false, fixed delay={:.3f}", extra_delay_);
  }
  rho_ = tools::read<double>(yaml, "rho");
  max_iter_ = tools::read<int>(yaml, "max_iter");

  // 读取 AIMD 自适应延迟配置
  auto aimd_yaml = yaml["aimd"];
  tools::AdaptiveDelayController::Config aimd_cfg;
  aimd_cfg.enable = aimd_yaml ? tools::read<bool>(aimd_yaml, "enable") : false;
  if (aimd_cfg.enable) {
    aimd_cfg.initial_delay = tools::read<double>(aimd_yaml, "initial_delay");
    aimd_cfg.min_delay = tools::read<double>(aimd_yaml, "min_delay");
    aimd_cfg.max_delay = tools::read<double>(aimd_yaml, "max_delay");
    aimd_cfg.add_step = tools::read<double>(aimd_yaml, "add_step");
    aimd_cfg.mul_factor = tools::read<double>(aimd_yaml, "mul_factor");
    aimd_cfg.fire_wait_threshold = tools::read<int>(aimd_yaml, "fire_wait_threshold");
    aimd_cfg.max_linear_speed = tools::read<double>(aimd_yaml, "max_linear_speed");
    aimd_cfg.max_angular_speed = tools::read<double>(aimd_yaml, "max_angular_speed");
  }
  aimd_ctrl_.init(aimd_cfg);
  aimd_enabled_ = aimd_cfg.enable;
  bullet_speed_min_ = tools::read<double>(yaml, "bullet_speed_min");
  bullet_speed_max_ = tools::read<double>(yaml, "bullet_speed_max");
  bullet_speed_default_ = tools::read<double>(yaml, "bullet_speed_default");

  // 读取机动自适应配置
  auto maneuver_yaml = yaml["maneuver_adapt"];
  if (maneuver_yaml) {
    maneuver_.enable = tools::read<bool>(maneuver_yaml, "enable");
    if (maneuver_.enable) {
      maneuver_.nis_threshold = std::max(tools::read<double>(maneuver_yaml, "nis_threshold"), 1.0);
      maneuver_.damping_factor = std::clamp(tools::read<double>(maneuver_yaml, "damping_factor"), 0.0, 1.0);
      maneuver_.ema_alpha = std::clamp(tools::read<double>(maneuver_yaml, "ema_alpha"), 0.01, 1.0);
    }
  }

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (!std::isfinite(bullet_speed)) return invalid_plan("non-finite bullet speed");
  if (bullet_speed < bullet_speed_min_ || bullet_speed > bullet_speed_max_) {
    tools::logger()->warn(
      "[Planner] Bullet speed {:.2f} outside [{:.2f}, {:.2f}], using default {:.2f}", bullet_speed,
      bullet_speed_min_, bullet_speed_max_, bullet_speed_default_);
    bullet_speed = bullet_speed_default_;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable || !std::isfinite(bullet_traj.fly_time)) {
    return invalid_plan("invalid bullet trajectory");
  }
  auto fly_time = bullet_traj.fly_time;
  target.predict(bullet_traj.fly_time);
  const auto selected_armor =
    armor_selection_hysteresis_enabled_ ? select_armor(target) : -1;

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed, selected_armor)(0);
    traj = get_trajectory(target, yaw0, bullet_speed, selected_armor);
  } catch (const std::exception & e) {
    tools::logger()->warn("[Planner] Unsolvable target at bullet speed {:.2f}: {}", bullet_speed, e.what());
    return {};
  }
  if (!traj.allFinite() || !std::isfinite(yaw0)) return invalid_plan("non-finite reference trajectory");

  // 3. 机动自适应：NIS 高时衰减轨迹速度，使跟踪更平滑
  if (maneuver_.enable) {
    double nis = target.last_nis();
    if (std::isfinite(nis)) {
      nis_avg_ = maneuver_.ema_alpha * nis + (1.0 - maneuver_.ema_alpha) * nis_avg_;
      double alpha = std::clamp(nis_avg_ / maneuver_.nis_threshold, 0.0, 1.0);
      double damp = 1.0 - alpha * maneuver_.damping_factor;
      traj.row(1) *= damp;
      traj.row(3) *= damp;
    }
  }

  // 4. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  const auto yaw_status = tiny_solve(yaw_solver_);
  if (yaw_status != 0 && !yaw_nonconvergence_logged_) {
    tools::logger()->warn(
      "[Planner] Yaw MPC did not converge: status={}, iterations={}, using finite partial solution",
      yaw_status,
      yaw_solver_ ? yaw_solver_->solution->iter : 0);
    yaw_nonconvergence_logged_ = true;
  } else if (yaw_status == 0) {
    yaw_nonconvergence_logged_ = false;
  }
  if (!finite_solver_result(yaw_solver_)) {
    tools::logger()->warn(
      "[Planner] Yaw MPC output is invalid: status={}, iterations={}", yaw_status,
      yaw_solver_ ? yaw_solver_->solution->iter : 0);
    return {};
  }

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  const auto pitch_status = tiny_solve(pitch_solver_);
  if (pitch_status != 0 && !pitch_nonconvergence_logged_) {
    tools::logger()->warn(
      "[Planner] Pitch MPC did not converge: status={}, iterations={}, using finite partial solution",
      pitch_status,
      pitch_solver_ ? pitch_solver_->solution->iter : 0);
    pitch_nonconvergence_logged_ = true;
  } else if (pitch_status == 0) {
    pitch_nonconvergence_logged_ = false;
  }
  if (!finite_solver_result(pitch_solver_)) {
    tools::logger()->warn(
      "[Planner] Pitch MPC output is invalid: status={}, iterations={}", pitch_status,
      pitch_solver_ ? pitch_solver_->solution->iter : 0);
    return {};
  }

  Plan plan;
  plan.control = true;
  plan.debug_xyza = debug_xyza;
  plan.fly_time = fly_time;
  plan.debug_valid = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  if (!valid_plan(plan)) return invalid_plan("non-finite MPC output");
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) {
    armor_selector_.clear();
    aimd_ctrl_.reset();
    last_fire_advice_ = false;
    nis_avg_ = 0.0;
    return {false};
  }

  // 转速阈值延迟（带滞回）
  const auto state = target->state();
  double delay_time;
  if (decision_speed_enable_) {
    double speed = std::abs(state.yaw_rate());
    if (high_speed_state_) {
      if (speed < decision_speed_ - speed_hysteresis_) high_speed_state_ = false;
    } else {
      if (speed > decision_speed_ + speed_hysteresis_) high_speed_state_ = true;
    }
    delay_time = high_speed_state_ ? high_speed_delay_time_ : low_speed_delay_time_;
  } else {
    delay_time = extra_delay_;
  }

  // AIMD 自适应延迟
  if (aimd_enabled_) {
    double v_ang = std::abs(state.yaw_rate());
    double v_lin = std::hypot(state.velocity_x(), state.velocity_y());
    aimd_ctrl_.update(last_fire_advice_, v_lin, v_ang);
    delay_time = aimd_ctrl_.getDelay();
  }

  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  target->predict(future);

  auto result = plan(*target, bullet_speed);
  last_fire_advice_ = result.fire;
  return result;
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  const auto setup_status =
    tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), rho_, 2, 1, HORIZON, 0);
  if (setup_status != 0) {
    tools::logger()->error("[Planner] Failed to initialize yaw MPC solver: status={}", setup_status);
    throw std::runtime_error("failed to initialize yaw MPC solver");
  }

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  const auto bound_status = tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);
  if (bound_status != 0) {
    tools::logger()->error("[Planner] Failed to configure yaw MPC constraints: status={}", bound_status);
    throw std::runtime_error("failed to configure yaw MPC constraints");
  }

  yaw_solver_->settings->max_iter = max_iter_;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  const auto setup_status =
    tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), rho_, 2, 1, HORIZON, 0);
  if (setup_status != 0) {
    tools::logger()->error("[Planner] Failed to initialize pitch MPC solver: status={}", setup_status);
    throw std::runtime_error("failed to initialize pitch MPC solver");
  }

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  const auto bound_status = tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);
  if (bound_status != 0) {
    tools::logger()->error("[Planner] Failed to configure pitch MPC constraints: status={}", bound_status);
    throw std::runtime_error("failed to configure pitch MPC constraints");
  }

  pitch_solver_->settings->max_iter = max_iter_;
}

int Planner::select_armor(const Target & target)
{
  std::vector<double> scores;
  for (const auto & xyza : target.armor_xyza_list()) scores.push_back(xyza.head<2>().norm());
  return armor_selector_.select(scores);
}

Eigen::Matrix<double, 2, 1> Planner::aim(
  const Target & target, double bullet_speed, int selected_armor)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  const auto armors = target.armor_xyza_list();
  if (selected_armor >= 0 && selected_armor < static_cast<int>(armors.size())) {
    const auto & xyza = armors[selected_armor];
    xyz = xyza.head<3>();
    yaw = xyza[3];
    min_dist = xyza.head<2>().norm();
  } else {
    for (const auto & xyza : armors) {
      auto dist = xyza.head<2>().norm();
      if (dist < min_dist) {
        min_dist = dist;
        xyz = xyza.head<3>();
        yaw = xyza[3];
      }
    }
  }
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(
  Target & target, double yaw0, double bullet_speed, int selected_armor)
{
  Trajectory traj;

  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed, selected_armor);

  target.predict(DT);  // [0] = -HALF_HORIZON * DT -> [HHALF_HORIZON] = 0
  auto yaw_pitch = aim(target, bullet_speed, selected_armor);

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed, selected_armor);

    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

}  // namespace auto_aim
