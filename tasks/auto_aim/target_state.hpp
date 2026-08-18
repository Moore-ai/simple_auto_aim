#ifndef AUTO_AIM__TARGET_STATE_HPP
#define AUTO_AIM__TARGET_STATE_HPP

#include <Eigen/Dense>

namespace auto_aim
{

enum class TargetStateComponent
{
  center_x,
  velocity_x,
  center_y,
  velocity_y,
  center_z,
  velocity_z,
  yaw,
  yaw_rate,
  radius,
  radius_diff,
  height_diff
};

class TargetState
{
public:
  static constexpr Eigen::Index dimension = 11;

  TargetState();
  explicit TargetState(const Eigen::VectorXd & values);

  Eigen::VectorXd vector() const;
  bool all_finite() const;

  double center_x() const;
  double velocity_x() const;
  double center_y() const;
  double velocity_y() const;
  double center_z() const;
  double velocity_z() const;
  double yaw() const;
  double yaw_rate() const;
  double radius() const;
  double radius_diff() const;
  double height_diff() const;

  void set_center_x(double value);
  void set_velocity_x(double value);
  void set_center_y(double value);
  void set_velocity_y(double value);
  void set_center_z(double value);
  void set_velocity_z(double value);
  void set_yaw(double value);
  void set_yaw_rate(double value);
  void set_radius(double value);
  void set_radius_diff(double value);
  void set_height_diff(double value);

  double radius(bool long_axis) const;
  double armor_height(bool long_axis) const;

private:
  static Eigen::Index index(TargetStateComponent component);

  Eigen::VectorXd values_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_STATE_HPP
