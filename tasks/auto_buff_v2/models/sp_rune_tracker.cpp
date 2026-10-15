#include "sp_rune_tracker.hpp"

#include <cmath>

#include "tools/math_tools.hpp"

namespace auto_buff_v2
{
namespace
{
Eigen::Vector3d blade_position(const Eigen::VectorXd & x, double radius, double pitch)
{
  return tools::ypd2xyz(Eigen::Vector3d(x[0], x[2], x[3])) +
    tools::rotation_matrix(Eigen::Vector3d(x[4], pitch, x[5])) * Eigen::Vector3d(0, 0, radius);
}
}  // namespace

void SPTracker::initialize(const SPRuneObservation & observation, double elapsed)
{
  last_time_ = elapsed;
  const int size = big_rune_ ? 10 : 7;
  Eigen::VectorXd x = Eigen::VectorXd::Zero(size);
  const Eigen::Vector3d center_ypd = !big_rune_ &&
    observation.detected_r_in_world.norm() > 1e-6 ?
    tools::xyz2ypd(observation.detected_r_in_world) : observation.ypd_in_world;
  x.head<7>() << center_ypd[0], 0, center_ypd[1], center_ypd[2],
    observation.ypr_in_world[0], observation.ypr_in_world[2],
    big_rune_ ? config_.initial_big_speed :
      (direction() == 0 ? 1 : direction()) * config_.initial_small_speed;
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(size, size) * config_.initial_covariance;
  if (big_rune_) {
    if (slot_ < 0) slot_ = (static_cast<int>(std::round(x[5] / (kSPSlotAngle))) % 5 + 5) % 5;
    last_canonical_ = tools::limit_rad(x[5] + slot_ * kSPSlotAngle);
    unwrapped_phase_ = last_canonical_ - slot_ * kSPSlotAngle;
    phase_initialized_ = true;
    x[5] = unwrapped_phase_;
    x.tail<3>() << config_.initial_amplitude, config_.initial_frequency, config_.initial_phase;
    P(6, 6) = config_.big_speed_covariance;
    P(9, 9) = config_.big_phase_covariance;
  } else {
    P(6, 6) = config_.small_speed_covariance;
  }
  const auto add = [big = big_rune_](const Eigen::VectorXd & a,
                                   const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd result = a + b;
    for (const int index : {0, 2, 4, 5}) result[index] = tools::limit_rad(result[index]);
    if (big) result[9] = tools::limit_rad(result[9]);
    return result;
  };
  ekf_ = tools::ExtendedKalmanFilter(x, P, add);
  initialized_ = true;
}

void SPTracker::predict(double dt)
{
  const int size = ekf_.x.size();
  Eigen::MatrixXd F = Eigen::MatrixXd::Identity(size, size);
  F(0, 1) = dt;
  F(5, 6) = big_rune_ ? direction() * dt : dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(size, size);
  const double variance =
    big_rune_ ? config_.big_yaw_acceleration_noise : config_.small_yaw_acceleration_noise;
  Q(0, 0) = std::pow(dt, 4) / 4 * variance;
  Q(0, 1) = Q(1, 0) = std::pow(dt, 3) / 2 * variance;
  Q(1, 1) = dt * dt * variance;
  const double time = last_time_ + dt;
  if (big_rune_) {
    const double a = ekf_.x[7], w = ekf_.x[8], phi = ekf_.x[9];
    F(6, 6) = 0;
    F(6, 7) = std::sin(w * time + phi) - 1;
    F(6, 8) = time * a * std::cos(w * time + phi);
    F(6, 9) = a * std::cos(w * time + phi);
    Q(5, 5) = config_.big_rotation_angle_noise;
    Q(6, 6) = config_.big_rotation_speed_noise;
    Q(9, 9) = config_.big_phase_noise;
  } else {
    const double v = config_.small_rotation_acceleration_noise;
    Q(5, 5) = std::pow(dt, 4) / 4 * v;
    Q(5, 6) = Q(6, 5) = std::pow(dt, 3) / 2 * v;
    Q(6, 6) = dt * dt * v;
  }
  const auto transition = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd result = x;
    result[0] = tools::limit_rad(x[0] + dt * x[1]);
    if (big_rune_) {
      const double a = x[7], w = x[8], phi = x[9];
      result[5] = x[5] + direction() *
        (-a / w * std::cos(w * time + phi) + a / w * std::cos(w * last_time_ + phi) +
         (config_.speed_baseline - a) * dt);
      result[6] = a * std::sin(w * time + phi) + config_.speed_baseline - a;
    } else {
      result[5] = x[5] + dt * x[6];
    }
    for (const int index : {2, 4, 5}) result[index] = tools::limit_rad(result[index]);
    return result;
  };
  ekf_.predict(F, Q, transition);
}

void SPTracker::set_slot(int slot)
{
  if (slot < 0 || slot >= 5) return;
  if (initialized_ && phase_initialized_ && slot_ >= 0 && slot != slot_) {
    const double canonical = ekf_.x[5] + slot_ * kSPSlotAngle;
    ekf_.x[5] = canonical - slot * kSPSlotAngle;
    unwrapped_phase_ = ekf_.x[5];
    last_canonical_ = tools::limit_rad(canonical);
  }
  slot_ = slot;
}

void SPTracker::reset()
{
  initialized_ = false;
  predicted_ = false;
  lost_count_ = 0;
  direction_votes_ = 0;
  start_timestamp_.reset();
  last_time_ = 0;
  slot_ = -1;
  phase_initialized_ = false;
}

bool SPTracker::update(const SPRuneObservation & observation, Timestamp timestamp)
{
  return update(std::optional<SPRuneObservation>(observation), timestamp);
}

bool SPTracker::update(const std::optional<SPRuneObservation> & input, Timestamp timestamp)
{
  if (!start_timestamp_) start_timestamp_ = timestamp;
  const double elapsed = std::chrono::duration<double>(timestamp - *start_timestamp_).count();
  const double dt = elapsed - last_time_;
  if (dt < 0 || (initialized_ && dt >= config_.max_prediction_gap)) {
    reset();
    if (!input) return false;
    return update(input, timestamp);
  }
  auto observation = input;
  if (observation && (!observation->ypd_in_world.allFinite() ||
      !observation->ypr_in_world.allFinite() || !observation->blade_ypd_in_world.allFinite() ||
      observation->ypd_in_world[2] <= config_.min_depth)) observation.reset();
  if (!observation) {
    if (!initialized_ || lost_count_ >= config_.max_lost_frames) {
      reset();
      return false;
    }
    predict(dt);
    last_time_ = elapsed;
    ++lost_count_;
    predicted_ = true;
    return ekf_.x.allFinite() && ekf_.P.allFinite();
  }
  if (!initialized_) initialize(*observation, elapsed);
  if (!correct(*observation, elapsed)) {
    reset();
    return false;
  }
  lost_count_ = 0;
  predicted_ = false;
  if (!ekf_.x.allFinite() || !ekf_.P.allFinite() || ekf_.x[3] <= 0 ||
      (!big_rune_ && std::abs(std::abs(ekf_.x[6]) - config_.initial_small_speed) >
        config_.small_speed_tolerance) ||
      (big_rune_ && (std::abs(ekf_.x[6]) > config_.big_speed_limit ||
       ekf_.x[7] > config_.amplitude_max * config_.divergence_factor ||
       ekf_.x[7] < config_.amplitude_min / config_.divergence_factor ||
       ekf_.x[8] > config_.frequency_max * config_.divergence_factor ||
       ekf_.x[8] < config_.frequency_min / config_.divergence_factor))) {
    // 当前帧发散时，不向下游瞄准模块输出该帧状态。
    initialized_ = false;
    return false;
  }
  return true;
}

bool SPTracker::correct(const SPRuneObservation & observation, double elapsed)
{
  double roll = observation.ypr_in_world[2];
  if (big_rune_ && phase_initialized_) {
    const double canonical = tools::limit_rad(roll + slot_ * kSPSlotAngle);
    unwrapped_phase_ += tools::limit_rad(canonical - last_canonical_);
    last_canonical_ = canonical;
    roll = unwrapped_phase_;
  }
  if (std::abs(roll - ekf_.x[5]) > config_.phase_switch_threshold) {
    for (int i = -5; i <= 5; ++i) {
      if (std::abs(ekf_.x[5] + i * kSPSlotAngle - roll) < config_.phase_match_tolerance) {
        ekf_.x[5] += i * kSPSlotAngle;
        break;
      }
    }
  }
  const double delta = tools::limit_rad(roll - ekf_.x[5]);
  if (std::abs(direction_votes_) <= config_.direction_vote_limit &&
      std::abs(delta) <= config_.direction_vote_max_delta)
    direction_votes_ += delta < 0 ? -1 : 1;
  if (!big_rune_ && std::abs(direction_votes_) >= config_.direction_vote_min_count &&
      direction() * ekf_.x[6] < 0)
    ekf_.x[6] *= -1;
  predict(elapsed - last_time_);

  const Eigen::Vector3d center_ypd = !big_rune_ &&
    observation.detected_r_in_world.norm() > 1e-6 ?
    tools::xyz2ypd(observation.detected_r_in_world) : observation.ypd_in_world;
  if (std::abs(tools::limit_rad(center_ypd[0] - ekf_.x[0])) > config_.max_center_angle_residual ||
      std::abs(tools::limit_rad(center_ypd[1] - ekf_.x[2])) > config_.max_center_angle_residual ||
      std::abs(center_ypd[2] - ekf_.x[3]) > config_.max_center_distance_residual ||
      std::abs(tools::limit_rad(roll - ekf_.x[5])) > config_.max_roll_residual) return false;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, ekf_.x.size());
  H(0, 0) = H(1, 2) = H(2, 3) = H(3, 5) = 1;
  Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
  R.diagonal() << config_.observation_yaw, config_.observation_pitch,
    config_.observation_distance, config_.observation_rotation_angle;
  Eigen::Vector4d z;
  z << center_ypd, roll;
  const auto subtract_pose = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b)
    -> Eigen::VectorXd {
    Eigen::VectorXd result = a - b;
    for (const int index : {0, 1, 3}) result[index] = tools::limit_rad(result[index]);
    return result;
  };
  ekf_.update(z, H, R, subtract_pose);

  if (big_rune_) {
    observed_pitch_ = observation.ypr_in_world[1];
    const Eigen::MatrixXd Hb = blade_jacobian();
    Eigen::Matrix3d Rb = Eigen::Matrix3d::Zero();
    Rb.diagonal() << config_.blade_observation_yaw, config_.blade_observation_pitch,
      config_.blade_observation_distance;
    const auto observe_blade = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      return tools::xyz2ypd(blade_position(x, config_.target_radius, observed_pitch_));
    };
    const auto subtract_blade = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b)
      -> Eigen::VectorXd {
      Eigen::VectorXd result = a - b;
      result[0] = tools::limit_rad(result[0]);
      result[1] = tools::limit_rad(result[1]);
      return result;
    };
    ekf_.update(observation.blade_ypd_in_world, Hb, Rb, observe_blade, subtract_blade);
    ekf_.x[0] = center_ypd[0];
    ekf_.x[2] = center_ypd[1];
    ekf_.x[3] = center_ypd[2];
  }
  // 与上游一致，在运动状态和协方差校正后，仅锚定位姿分量。
  ekf_.x[4] = observation.ypr_in_world[0];
  ekf_.x[5] = big_rune_ ? roll : observation.ypr_in_world[2];
  observed_pitch_ = observation.ypr_in_world[1];
  last_time_ = elapsed;
  return true;
}

Eigen::MatrixXd SPTracker::blade_jacobian() const
{
  const auto & x = ekf_.x;
  Eigen::MatrixXd H0 = Eigen::MatrixXd::Zero(5, x.size());
  H0(0, 0) = H0(1, 2) = H0(2, 3) = H0(3, 4) = H0(4, 5) = 1;
  Eigen::Matrix<double, 5, 5> H1 = Eigen::Matrix<double, 5, 5>::Identity();
  H1.topLeftCorner<3, 3>() = tools::ypd2xyz_jacobian(Eigen::Vector3d(x[0], x[2], x[3]));
  const double cy = std::cos(x[4]), sy = std::sin(x[4]);
  const double cr = std::cos(x[5]), sr = std::sin(x[5]);
  Eigen::Matrix<double, 3, 5> H2;
  // clang-format off
  H2 << 1, 0, 0, config_.target_radius * cy * sr,  config_.target_radius * sy * cr,
        0, 1, 0, config_.target_radius * sy * sr, -config_.target_radius * cy * cr,
        0, 0, 1,              0,        -config_.target_radius * sr;
  // clang-format on
  return tools::xyz2ypd_jacobian(blade_position(x, config_.target_radius, observed_pitch_)) *
         H2 * H1 * H0;
}
}  // namespace auto_buff_v2
