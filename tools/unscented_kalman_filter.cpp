#include "tools/unscented_kalman_filter.hpp"

#include "tools/logger.hpp"

namespace tools
{

UnscentedKalmanFilter::UnscentedKalmanFilter(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add,
  double alpha, double beta, double kappa)
: FilterBase(x0, P0, x_add),
  n_(x0.rows()),
  alpha_(alpha),
  beta_(beta),
  kappa_(kappa)
{
  lambda_ = alpha_ * alpha_ * (n_ + kappa_) - n_;
  compute_weights();
}

void UnscentedKalmanFilter::compute_weights()
{
  int total = 2 * n_ + 1;
  Wm_.resize(total);
  Wc_.resize(total);

  Wm_(0) = lambda_ / (n_ + lambda_);
  Wc_(0) = Wm_(0) + (1.0 - alpha_ * alpha_ + beta_);
  for (int i = 1; i < total; ++i) {
    double w = 0.5 / (n_ + lambda_);
    Wm_(i) = w;
    Wc_(i) = w;
  }
}

Eigen::MatrixXd UnscentedKalmanFilter::generate_sigma_points() const
{
  Eigen::MatrixXd sigma(n_, 2 * n_ + 1);
  sigma.col(0) = x;

  Eigen::MatrixXd A = (n_ + lambda_) * P;

  // Cholesky with progressive regularization
  Eigen::LLT<Eigen::MatrixXd> llt(A);
  if (llt.info() != Eigen::Success) {
    double reg = 1e-6;
    for (int attempt = 0; attempt < 5; ++attempt) {
      A = (n_ + lambda_) * P + reg * Eigen::MatrixXd::Identity(n_, n_);
      llt.compute(A);
      if (llt.info() == Eigen::Success) break;
      reg *= 10.0;
    }
    if (llt.info() != Eigen::Success) {
      logger()->warn("[UKF] Cholesky failed, using diagonal regularization");
      A = (n_ + lambda_) * P +
          reg * Eigen::MatrixXd::Identity(n_, n_);
      llt.compute(A);
    }
  }

  Eigen::MatrixXd L = llt.matrixL();
  for (int i = 0; i < n_; ++i) {
    sigma.col(i + 1) = x + L.col(i);
    sigma.col(n_ + i + 1) = x - L.col(i);
  }
  return sigma;
}

Eigen::VectorXd UnscentedKalmanFilter::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  (void)F;  // UKF ignores F, propagates sigma points through f

  Eigen::MatrixXd sigma = generate_sigma_points();

  // Propagate sigma points through f
  int total = 2 * n_ + 1;
  Eigen::MatrixXd Ysig(n_, total);
  for (int i = 0; i < total; ++i) {
    Ysig.col(i) = f(sigma.col(i));
  }

  // Weighted mean
  x.setZero();
  for (int i = 0; i < total; ++i) {
    x += Wm_(i) * Ysig.col(i);
  }

  // Weighted covariance
  P.setZero();
  for (int i = 0; i < total; ++i) {
    Eigen::VectorXd diff = Ysig.col(i) - x;
    P += Wc_(i) * diff * diff.transpose();
  }
  P += Q;

  return x;
}

Eigen::VectorXd UnscentedKalmanFilter::update(
  const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract)
{
  (void)H;  // UKF ignores H, computes Kalman gain from sigma points

  Eigen::VectorXd x_prior = x;

  // Generate sigma points from prior state
  Eigen::MatrixXd sigma = generate_sigma_points();
  int total = 2 * n_ + 1;
  int obs_dim = z.rows();

  // Propagate sigma points through observation model
  Eigen::MatrixXd Zsig(obs_dim, total);
  for (int i = 0; i < total; ++i) {
    Zsig.col(i) = h(sigma.col(i));
  }

  // Weighted mean observation
  Eigen::VectorXd z_pred(obs_dim);
  z_pred.setZero();
  for (int i = 0; i < total; ++i) {
    z_pred += Wm_(i) * Zsig.col(i);
  }

  // Innovation covariance Pzz
  Eigen::MatrixXd Pzz(obs_dim, obs_dim);
  Pzz.setZero();
  for (int i = 0; i < total; ++i) {
    Eigen::VectorXd diff = Zsig.col(i) - z_pred;
    Pzz += Wc_(i) * diff * diff.transpose();
  }
  Pzz += R;

  // Cross covariance Pxz
  Eigen::MatrixXd Pxz(n_, obs_dim);
  Pxz.setZero();
  for (int i = 0; i < total; ++i) {
    Eigen::VectorXd x_diff = sigma.col(i) - x_prior;
    Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;
    Pxz += Wc_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  Eigen::MatrixXd K = Pxz * Pzz.inverse();

  // Update state
  Eigen::VectorXd innovation = z_subtract(z, z_pred);
  x = x_add(x_prior, K * innovation);

  // Update covariance (Joseph form alternative: K * Pzz * K^T is standard UKF)
  P = P - K * Pzz * K.transpose();

  // NIS
  compute_nis(innovation, Pzz);

  // NEES
  double nees = (x - x_prior).transpose() * P.inverse() * (x - x_prior);
  data["nees"] = nees;

  data["residual_yaw"] = innovation[3];
  data["residual_distance"] = innovation.head(2).norm();
  data["residual_angle"] = std::atan2(innovation[1], innovation[0]);

  return x;
}

std::unique_ptr<FilterBase> UnscentedKalmanFilter::clone() const
{
  return std::make_unique<UnscentedKalmanFilter>(*this);
}

}  // namespace tools
