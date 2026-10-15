#include "sp_rune_model.hpp"

#include <algorithm>
#include <cmath>

#include "tools/math_tools.hpp"
#include "../rune_predictor.hpp"

namespace auto_buff_v2
{
SPRuneModel::SPRuneModel(BuffConfig::Camera camera, Config config, bool big_rune)
: solver_(std::move(camera), config), config_(std::move(config)),
  motion_fitter_(config_.fit_iterations, config_.fit_threshold,
    config_.frequency_min, config_.frequency_max, config_.fit_history),
  big_rune_(big_rune)
{
  reset();
}

void SPRuneModel::set_q_gimbal2world(const Eigen::Quaterniond & q)
{
  solver_.set_q_gimbal2world(q);
}

void SPRuneModel::reset()
{
  if (tracker_) {
    tracker_->reset();
  } else {
    tracker_ = std::make_unique<SPTracker>(big_rune_, config_);
  }
  state_.reset();
  motion_fitter_.reset();
  visible_slots_.fill(false);
  blade_ids_.fill(-1);
  blade_tracks_ = {};
  reference_slot_ = -1;
  pose_center_.setZero();
  diagnostic_.clear();
}

bool SPRuneModel::update(const RuneElements & elements, Timestamp timestamp)
{
  if (state_ && timestamp < state_->timestamp) {
    reset();
    diagnostic_ = "timestamp reversed";
    return false;
  }
  if (state_) {
    const auto future = RunePredictor{}.predict(*state_, timestamp);
    const double delta = future.rotation_angle - state_->rotation_angle;
    for (auto & track : blade_tracks_) if (track.active) track.roll += delta;
  }
  const auto observations = solver_.solve_all(elements);
  std::array<std::optional<SPRuneObservation>, 5> slots;
  std::optional<SPRuneObservation> observation;
  for (auto & track : blade_tracks_) {
    if (track.active && ++track.missed > config_.max_blade_missed_frames) track = {};
  }
  if (big_rune_) {
    for (const auto & obs : observations) {
      if (!slots[obs.slot_id] || slots[obs.slot_id]->score < obs.score)
        slots[obs.slot_id] = obs;
    }
    int selected = reference_slot_;
    if (selected < 0 || !slots[selected]) {
      selected = -1;
      for (int i = 0; i < kSPSlotCount; ++i) {
        if (slots[i]) {
          selected = i;
          break;
        }
      }
    }
    if (selected >= 0) {
      // 以一个图像槽位为参考，再根据解算得到的物理自转角
      // 推导其余槽位。透视投影可能导致槽位重合或跳号，
      // 直接量化图像角度无法表示刚性的五叶模型。
      std::array<std::optional<SPRuneObservation>, 5> associated;
      const double reference_roll = slots[selected]->ypr_in_world[2];
      for (const auto & obs : observations) {
        const int offset = static_cast<int>(std::round(tools::limit_rad(
          reference_roll - obs.ypr_in_world[2]) / kSPSlotAngle));
        const int slot = (selected + offset + 5) % 5;
        if (!associated[slot] || associated[slot]->score < obs.score) {
          associated[slot] = obs;
          associated[slot]->slot_id = slot;
        }
      }
      slots = associated;
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      int count = 0;
      for (const auto & obs : slots) {
        if (!obs) continue;
        center += tools::ypd2xyz(obs->ypd_in_world);
        ++count;
      }
      center /= count;
      observation = slots[selected];
      observation->ypd_in_world = tools::xyz2ypd(center);
      tracker_->set_slot(selected);
      reference_slot_ = selected;
    }
  } else if (!observations.empty()) {
    // 存在多个检测候选时，保留物理位置最近的扇叶。
    observation = *std::min_element(observations.begin(), observations.end(),
      [&](const auto & a, const auto & b) {
        if (!state_) return a.score > b.score;
        return std::abs(tools::limit_rad(a.ypr_in_world[2] - state_->rotation_angle)) <
               std::abs(tools::limit_rad(b.ypr_in_world[2] - state_->rotation_angle));
      });
    reference_slot_ = observation->slot_id;
  }
  if (!tracker_->update(observation, timestamp)) {
    state_.reset();
    visible_slots_.fill(false);
    reference_slot_ = -1;
    diagnostic_ = observation ? "SP reinitializing" : "missing inactive bullseye or valid PnP";
    return false;
  }
  const auto & x = tracker_->state();
  if (!x.allFinite()) {
    reset();
    diagnostic_ = "SP model diverged";
    return false;
  }
  RuneEstimate estimate;
  estimate.start_timestamp = tracker_->start_timestamp();
  estimate.requires_warmup = false;
  estimate.update_count = state_ ? state_->update_count + 1 : 1;
  estimate.timestamp = timestamp;
  if (observation) {
    pose_center_ = tools::ypd2xyz(observation->ypd_in_world);
  }
  estimate.center = pose_center_;
  estimate.target_radius = config_.target_radius;
  // 当前 Climber 的 set_pose_anchor 为大小符均重建物理上垂直的
  // 面板；含噪声的 PnP 俯仰角仅作为观测，不用于瞄准几何。
  estimate.slot_id = reference_slot_;
  estimate.predicted = tracker_->predicted();
  estimate.face_yaw = x[4];
  // 预测未来位置时，位姿锚点保持在最近一次观测到的中心。
  // x[1] 仍参与 EKF 中心校正和协方差传播。
  estimate.center_yaw_speed = 0;
  estimate.rotation_angle = x[5];
  if (observation) {
    visible_slots_.fill(false);
    if (big_rune_) {
      for (int slot = 0; slot < 5; ++slot) visible_slots_[slot] = slots[slot].has_value();
    } else {
      visible_slots_[reference_slot_] = true;
    }
  }
  // 共用预测器使用正向自转角偏移；图像平面槽位的编号方向相反。
  for (int offset = 0; offset < 5; ++offset)
    estimate.inactive[offset] = visible_slots_[(reference_slot_ - offset + 5) % 5];
  estimate.prediction_method = RunePredictionMethod::ekf;
  estimate.rotation_speed = x[6];
  if (big_rune_) {
    // 未来角度通过积分 EKF 参数得到，与诊断拟合独立。
    const double elapsed = tracker_->time();
    const int direction = tracker_->direction();
    estimate.prediction_method = RunePredictionMethod::weighted_least_squares;
    estimate.sine_valid = true;
    estimate.sine_v = direction * (config_.speed_baseline - x[7]);
    estimate.sine_a = direction * x[7];
    estimate.sine_omega = x[8];
    estimate.sine_phase = x[8] * elapsed + x[9];
    estimate.sine_t = elapsed;
    estimate.rotation_speed = direction * x[6];
    if (observation && x[6] >= 0 && x[6] < config_.fit_max_speed) {
      motion_fitter_.add_data(elapsed, x[6]);
      motion_fitter_.fit();
    }
    if (motion_fitter_.valid()) {
      const auto & fit = motion_fitter_.best_result_;
      estimate.has_fitted_motion = true;
      estimate.fitted_rotation_speed = direction * motion_fitter_.sine_function(
        elapsed, fit.A, fit.omega, fit.phi, fit.C);
      if (std::abs(estimate.fitted_rotation_speed) > config_.fit_output_limit)
        estimate.fitted_rotation_speed = 0;
    }
  }
  state_ = estimate;
  if (observation) update_blade_tracks();
  state_->blade_id = reference_slot_ >= 0 ? blade_ids_[reference_slot_] : -1;
  diagnostic_ = estimate.predicted ? "SP coasting (fire inhibited)" : "SP tracking";
  return true;
}

void SPRuneModel::update_blade_tracks()
{
  blade_ids_.fill(-1);
  std::array<bool, 5> used{};
  // 按全局最小角度差关联，与 Climber BigTargetGroup 一致。
  while (true) {
    double best = config_.slot_match_angle;
    int best_slot = -1, best_id = -1;
    for (int slot = 0; slot < 5; ++slot) {
      if (!visible_slots_[slot] || blade_ids_[slot] >= 0) continue;
      const double roll = state_->rotation_angle + (reference_slot_ - slot) * kSPSlotAngle;
      for (int id = 0; id < 5; ++id) {
        if (used[id] || !blade_tracks_[id].active) continue;
        const double cost = std::abs(tools::limit_rad(roll - blade_tracks_[id].roll));
        if (cost < best) {
          best = cost;
          best_slot = slot;
          best_id = id;
        }
      }
    }
    if (best_id < 0) break;
    blade_ids_[best_slot] = best_id;
    used[best_id] = true;
  }
  for (int slot = 0; slot < 5; ++slot) {
    if (!visible_slots_[slot]) continue;
    if (blade_ids_[slot] < 0) {
      int available = -1;
      for (int id = 0; id < kSPSlotCount; ++id) {
        if (used[id]) continue;
        if (!blade_tracks_[id].active) {
          available = id;
          break;
        }
        if (available < 0 || blade_tracks_[id].missed > blade_tracks_[available].missed)
          available = id;
      }
      if (available < 0) continue;
      blade_ids_[slot] = available;
      used[available] = true;
    }
    auto & track = blade_tracks_[blade_ids_[slot]];
    track.active = true;
    track.roll = state_->rotation_angle + (reference_slot_ - slot) * kSPSlotAngle;
    track.missed = 0;
  }
}

std::vector<RuneEstimate> SPRuneModel::candidates() const
{
  if (!state_) return {};
  std::vector<RuneEstimate> result;
  for (int slot = 0; slot < 5; ++slot) {
    if (!visible_slots_[slot]) continue;
    auto candidate = *state_;
    candidate.rotation_angle += (reference_slot_ - slot) * kSPSlotAngle;
    candidate.slot_id = slot;
    candidate.blade_id = blade_ids_[slot];
    candidate.inactive.fill(false);
    candidate.inactive[0] = true;
    result.push_back(candidate);
  }
  return result;
}

std::vector<RuneReprojectedFeature> SPRuneModel::reprojected_features() const
{
  if (!state_) return {};
  std::vector<RuneReprojectedFeature> result;
  if (const auto icon = reprojected_center()) result.push_back({0, *icon});
  for (int blade = 0; blade < 5; ++blade) {
    const double angle = state_->rotation_angle + (reference_slot_ - blade) * kSPSlotAngle;
    const Eigen::Vector3d world = state_->center +
      Eigen::AngleAxisd(state_->face_yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::Vector3d(0, -config_.target_radius * std::sin(angle),
        config_.target_radius * std::cos(angle));
    if (const auto pixel = solver_.project(world)) result.push_back({blade + 1, *pixel});
  }
  return result;
}

std::optional<cv::Point2f> SPRuneModel::reprojected_center() const
{
  if (!state_) return std::nullopt;
  const Eigen::Vector3d r = state_->center +
    tools::rotation_matrix({state_->face_yaw, 0, state_->rotation_angle}) *
      Eigen::Vector3d(0, 0, config_.r_center_z);
  return solver_.project(r);
}
}  // namespace auto_buff_v2
