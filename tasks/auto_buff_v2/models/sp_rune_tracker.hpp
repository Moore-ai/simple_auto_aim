#ifndef AUTO_BUFF_V2__SP_RUNE_TRACKER_HPP
#define AUTO_BUFF_V2__SP_RUNE_TRACKER_HPP

#include "rune_model.hpp"
#include "sp_rune_config.hpp"
#include "sp_rune_solver.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_buff_v2
{
// 移植小符七维与大符十维球坐标 EKF。
// 状态顺序：[中心方位角、方位角速度、中心俯仰角、距离、面板朝向角、自转角、自转角速度、振幅、角频率、相位]。
class SPTracker
{
public:
  SPTracker(bool big_rune, SPRuneConfig config) : big_rune_(big_rune), config_(config)
  {
    config_.validate();
  }
  bool update(const SPRuneObservation & observation, Timestamp timestamp);
  bool update(const std::optional<SPRuneObservation> & observation, Timestamp timestamp);
  void reset();
  const Eigen::VectorXd & state() const { return ekf_.x; }
  int direction() const { return direction_votes_ > 0 ? 1 : (direction_votes_ < 0 ? -1 : 0); }
  bool predicted() const { return predicted_; }
  const Eigen::MatrixXd & covariance() const { return ekf_.P; }
  void set_slot(int slot);
  int slot() const { return slot_; }
  Timestamp start_timestamp() const { return start_timestamp_.value(); }
  double time() const { return last_time_; }

private:
  void initialize(const SPRuneObservation & observation, double elapsed);
  bool correct(const SPRuneObservation & observation, double elapsed);
  void predict(double dt);
  Eigen::MatrixXd blade_jacobian() const;

  bool big_rune_;
  SPRuneConfig config_;
  bool initialized_ = false;
  bool predicted_ = false;
  int slot_ = -1;
  bool phase_initialized_ = false;
  double unwrapped_phase_ = 0;
  double last_canonical_ = 0;
  double observed_pitch_ = 0;
  int direction_votes_ = 0;
  int lost_count_ = 0;
  std::optional<Timestamp> start_timestamp_;
  double last_time_ = 0;
  tools::ExtendedKalmanFilter ekf_;
};
}  // namespace auto_buff_v2

#endif
