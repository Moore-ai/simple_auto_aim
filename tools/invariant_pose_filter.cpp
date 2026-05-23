#include "invariant_pose_filter.hpp"

#include <cmath>

namespace tools
{

Eigen::VectorXd InvariantPoseFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  Eigen::VectorXd x_prior = x;

  Eigen::VectorXd z_pred = h(x);

  Eigen::VectorXd innov_world = z_subtract(z, z_pred);

  int n_obs = z.size();

  Eigen::VectorXd innov = innov_world;
  Eigen::MatrixXd H_body = H;
  Eigen::MatrixXd R_body = R;
  if (n_obs >= 2) {
    double cy = x(6);
    double c = std::cos(cy);
    double s = std::sin(cy);

    innov(0) = c * innov_world(0) + s * innov_world(1);
    innov(1) = -s * innov_world(0) + c * innov_world(1);

    Eigen::VectorXd tmp_row0 = H.row(0);
    H_body.row(0) = c * tmp_row0 + s * H.row(1);
    H_body.row(1) = -s * tmp_row0 + c * H.row(1);

    Eigen::VectorXd tmp_rrow0 = R.row(0);
    R_body.row(0) = c * tmp_rrow0 + s * R.row(1);
    R_body.row(1) = -s * tmp_rrow0 + c * R.row(1);
    Eigen::VectorXd tmp_rcol0 = R_body.col(0);
    R_body.col(0) = c * tmp_rcol0 + s * R_body.col(1);
    R_body.col(1) = -s * tmp_rcol0 + c * R_body.col(1);
  }

  Eigen::MatrixXd S = H_body * P * H_body.transpose() + R_body;
  Eigen::MatrixXd K = P * H_body.transpose() * S.inverse();

  x = x_add(x, K * innov);

  Eigen::MatrixXd I_KH = I - K * H_body;
  P = I_KH * P * I_KH.transpose() + K * R_body * K.transpose();

  compute_nis(innov, S);

  Eigen::VectorXd dx = x - x_prior;
  data["nees"] = dx.transpose() * P.inverse() * dx;

  if (n_obs >= 4) {
    data["residual_yaw"] = innov_world(3);
  }
  if (n_obs >= 2) {
    data["residual_distance"] = innov_world.segment(0, 2).norm();
    data["residual_angle"] = std::atan2(innov_world(1), innov_world(0));
  }

  return x;
}

std::unique_ptr<FilterBase> InvariantPoseFilter::clone() const
{
  return std::make_unique<InvariantPoseFilter>(*this);
}

}  // namespace tools
