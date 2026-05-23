#include "extended_kalman_filter.hpp"

namespace tools
{

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  return update(z, H, R, [&](const Eigen::VectorXd & x) { return H * x; }, z_subtract);
}

Eigen::VectorXd ExtendedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  Eigen::VectorXd x_prior = x;
  Eigen::MatrixXd K = P * H.transpose() * (H * P * H.transpose() + R).inverse();

  P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();

  x = x_add(x, K * z_subtract(z, h(x)));

  Eigen::VectorXd residual = z_subtract(z, h(x));
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  compute_nis(residual, S);

  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);
  data["nees"] = nees;

  data["residual_yaw"] = residual[0];
  data["residual_pitch"] = residual[1];
  data["residual_distance"] = residual[2];
  data["residual_angle"] = residual[3];

  return x;
}

std::unique_ptr<FilterBase> ExtendedKalmanFilter::clone() const
{
  return std::make_unique<ExtendedKalmanFilter>(*this);
}

}  // namespace tools
