#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>

#include <array>
#include <optional>
#include <vector>

#include "armor.hpp"
#include "outpost_state.hpp"
#include "target_estimator.hpp"

namespace auto_aim
{

struct OutpostFilterConfig
{
  double accel_var = 10.0;
  double observation_yaw_var = 4e-3;
  double observation_pitch_var = 4e-3;
  double observation_armor_yaw_base = 9e-2;
  bool velocity_clamp_enabled = false;
  double max_linear_speed = 5.0;
};

class OutpostTarget
{
public:
  OutpostTarget(
    const Armor & armor, const Eigen::VectorXd & covariance_diagonal,
    const OutpostFilterConfig & config);

  void begin_frame();
  void predict(double dt);
  void update(const Armor & armor, int id);

  OutpostState state() const;
  TargetState compatibility_state() const;
  Eigen::VectorXd state_vector() const;
  std::vector<PredictedArmorPose> armor_pose_list() const;

  double last_nis() const;
  const TargetEstimatorDiagnostics & diagnostics() const;
  bool has_bad_nis_convergence(double failure_rate) const;
  bool direction_locked() const;
  bool all_finite() const;

private:
  OutpostFilterConfig config_;
  TargetEstimator estimator_;
  std::array<std::optional<double>, OUTPOST_ARMOR_COUNT> previous_yaws_{};
  std::array<std::optional<double>, OUTPOST_ARMOR_COUNT> current_yaws_{};
  int direction_vote_ = 0;
  int direction_ = 0;

  Eigen::Vector3d armor_center(const OutpostState & state, int id) const;
  Eigen::MatrixXd observation_jacobian(const OutpostState & state, int id) const;
  void update_direction(int id, double observed_yaw);
  void enforce_yaw_rate();
  void constrain_velocity();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
