#ifndef AUTO_AIM__OUTPOST_STATE_V2_HPP
#define AUTO_AIM__OUTPOST_STATE_V2_HPP

#include <Eigen/Dense>

#include <stdexcept>

namespace auto_aim
{

class OutpostStateV2
{
public:
  static constexpr Eigen::Index dimension = 10;

  explicit OutpostStateV2(const Eigen::VectorXd & values) : values_{values}
  {
    if (values_.size() != dimension) {
      throw std::invalid_argument("OutpostStateV2 must contain exactly 10 values");
    }
  }

  Eigen::VectorXd vector() const { return values_; }
  bool all_finite() const { return values_.allFinite(); }

  double center_x() const { return values_[0]; }
  double velocity_x() const { return values_[1]; }
  double center_y() const { return values_[2]; }
  double velocity_y() const { return values_[3]; }
  double center_z() const { return values_[4]; }
  double velocity_z() const { return values_[5]; }
  double yaw() const { return values_[6]; }
  double yaw_rate() const { return values_[7]; }
  double height_offset_1() const { return values_[8]; }
  double height_offset_2() const { return values_[9]; }

  void set_center_x(double value) { values_[0] = value; }
  void set_velocity_x(double value) { values_[1] = value; }
  void set_center_y(double value) { values_[2] = value; }
  void set_velocity_y(double value) { values_[3] = value; }
  void set_center_z(double value) { values_[4] = value; }
  void set_velocity_z(double value) { values_[5] = value; }
  void set_yaw(double value) { values_[6] = value; }
  void set_yaw_rate(double value) { values_[7] = value; }
  void set_height_offset_1(double value) { values_[8] = value; }
  void set_height_offset_2(double value) { values_[9] = value; }

private:
  Eigen::VectorXd values_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_STATE_V2_HPP
