#ifndef AUTO_AIM__OUTPOST_TARGET_V2_HPP
#define AUTO_AIM__OUTPOST_TARGET_V2_HPP

#include <optional>
#include <vector>

#include "outpost_model.hpp"
#include "outpost_state_v2.hpp"

namespace auto_aim
{

struct OutpostTargetV2Config
{
  double radius = 0.2765;
  double yaw_rate_magnitude = 0.8 * CV_PI;
  bool is_fit_yaw_rate = false;
  double q_xy = 10.0;
  double q_z = 10.0;
  double q_yaw = 10.0;
  double q_yaw_rate = 10.0;
  double q_dz = 10.0;
  double r_yaw = 1e-3;
  double r_pitch = 1e-3;
  double r_distance = 1e-3;
  double r_armor_yaw = 1e-3;
  double frontal_angle_gate = 35.0 * CV_PI / 180.0;
  double yaw_match_gate = 1.5;
  double position_match_gate = 0.05;
  int mismatch_reset_count = 5;
  int direction_sample_count = 30;
  int direction_compare_interval = 3;
  double direction_stable_threshold = 5e-3;
  double direction_jump_threshold = 4e-1;
};

class OutpostTargetV2 final : public OutpostModel
{
public:
  static constexpr Eigen::Index dimension = 10;

  OutpostTargetV2(
    const std::vector<Armor> & armors, const Eigen::VectorXd & covariance_diagonal,
    const OutpostTargetV2Config & config);

  std::unique_ptr<OutpostModel> clone() const override;
  void begin_frame() override;
  void predict(double dt) override;
  OutpostUpdateResult update(const std::vector<Armor> & armors) override;
  OutpostStateV2 state() const;
  TargetState compatibility_state() const override;
  std::optional<OutpostState> outpost_state() const override;
  std::optional<OutpostStateV2> outpost_state_v2() const override;
  Eigen::VectorXd state_vector() const override;
  std::vector<PredictedArmorPose> armor_pose_list() const override;
  double last_nis() const override;
  const TargetEstimatorDiagnostics & diagnostics() const override;
  bool has_bad_nis_convergence(double failure_rate) const override;
  bool direction_locked() const override;
  bool all_finite() const override;

private:
  enum class Direction { unknown, stable, anticlockwise, clockwise };

  OutpostTargetV2Config config_;
  TargetEstimator estimator_;
  Direction direction_ = Direction::unknown;
  std::vector<double> direction_data_;
  int mismatch_count_ = 0;

  Eigen::Vector3d armor_center(const OutpostStateV2 & state, int id) const;
  Eigen::MatrixXd observation_jacobian(const OutpostStateV2 & state, int id) const;
  int match_id(const OutpostStateV2 & state, const Armor & armor) const;
  void update_direction(double observed_yaw);
  void apply_yaw_rate();
  void clamp_height_offsets();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_V2_HPP
