#ifndef AUTO_AIM__OUTPOST_STATE_HPP
#define AUTO_AIM__OUTPOST_STATE_HPP

#include <Eigen/Dense>

namespace auto_aim
{

class OutpostState
{
public:
  static constexpr Eigen::Index dimension = 8;

  explicit OutpostState(const Eigen::VectorXd & values) : values_{values}
  {
    if (values_.size() != dimension) {
      throw std::invalid_argument("OutpostState must contain exactly 11 values");
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

private:
  Eigen::VectorXd values_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_STATE_HPP
