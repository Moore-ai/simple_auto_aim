#ifndef AUTO_BUFF_V2__SP_RUNE_TRACKER_HPP
#define AUTO_BUFF_V2__SP_RUNE_TRACKER_HPP

#include "rune_model.hpp"
#include "sp_rune_config.hpp"
#include "sp_rune_solver.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_buff_v2
{
// Port of SmallTarget's 7-state and BigTarget's 10-state spherical EKFs.
// [center yaw, yaw rate, center pitch, distance, face yaw, roll, speed, a, w, phi].
class SPTracker
{
public:
  SPTracker(bool big_rune, SPRuneConfig config) : big_rune_(big_rune), config_(config) {}
  void update(const SPRuneObservation & observation, Timestamp timestamp);
  const Eigen::VectorXd & state() const { return ekf_.x; }
  int direction() const { return direction_votes_ > 0 ? 1 : -1; }

private:
  void initialize(const SPRuneObservation & observation, Timestamp timestamp);
  void predict(double dt);
  Eigen::MatrixXd blade_jacobian() const;

  bool big_rune_;
  SPRuneConfig config_;
  bool initialized_ = false;
  int direction_votes_ = 0;
  Timestamp start_timestamp_{};
  double last_time_ = 0;
  tools::ExtendedKalmanFilter ekf_;
};
}  // namespace auto_buff_v2

#endif
