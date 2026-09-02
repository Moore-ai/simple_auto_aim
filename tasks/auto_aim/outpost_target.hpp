#ifndef AUTO_AIM__OUTPOST_TARGET_HPP
#define AUTO_AIM__OUTPOST_TARGET_HPP

#include <Eigen/Dense>

#include <array>
#include <optional>
#include <vector>

#include "armor.hpp"
#include "outpost_model.hpp"

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

class OutpostTarget final : public OutpostModel
{
public:
  OutpostTarget(
    const Armor & armor, const Eigen::VectorXd & covariance_diagonal,
    const OutpostFilterConfig & config);

  std::unique_ptr<OutpostModel> clone() const override;
  void begin_frame() override;
  void predict(double dt) override;
  OutpostUpdateResult update(const std::vector<Armor> & armors) override;
  void update(const Armor & armor, int id);

  OutpostState state() const;
  OutpostSnapshot snapshot() const override;
  const TargetEstimatorDiagnostics & diagnostics() const override;

private:
  OutpostFilterConfig config_;
  TargetEstimator estimator_;
  std::array<std::optional<double>, OUTPOST_ARMOR_COUNT> previous_yaws_{};
  std::array<std::optional<double>, OUTPOST_ARMOR_COUNT> current_yaws_{};
  std::array<std::optional<double>, OUTPOST_ARMOR_COUNT> observed_heights_{};
  int height_phase_ = 0;
  int direction_vote_ = 0;
  int direction_ = 0;

  Eigen::Vector3d armor_center(const OutpostState & state, int id) const;
  Eigen::MatrixXd observation_jacobian(const OutpostState & state, int id) const;
  int match_armor(const Armor & armor) const;
  double armor_height_offset(int id) const;
  void update_height_phase(int id, double observed_height);
  void update_direction(int id, double observed_yaw);
  void enforce_yaw_rate();
  void constrain_velocity();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_TARGET_HPP
