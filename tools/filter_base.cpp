#include "filter_base.hpp"

#include <numeric>

namespace tools
{

FilterBase::FilterBase(
  const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add)
: x(x0), P(P0), I(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())), x_add(x_add)
{
  data["residual_yaw"] = 0.0;
  data["residual_pitch"] = 0.0;
  data["residual_distance"] = 0.0;
  data["residual_angle"] = 0.0;
  data["nis"] = 0.0;
  data["nees"] = 0.0;
  data["nis_fail"] = 0.0;
  data["nees_fail"] = 0.0;
  data["recent_nis_failures"] = 0.0;
}

Eigen::VectorXd FilterBase::predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q)
{
  return predict(F, Q, [&](const Eigen::VectorXd & x) { return F * x; });
}

Eigen::VectorXd FilterBase::predict(
  const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
  std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f)
{
  P = F * P * F.transpose() + Q;
  x = f(x);
  return x;
}

void FilterBase::compute_nis(
  const Eigen::VectorXd & residual, const Eigen::MatrixXd & S)
{
  double nis = residual.transpose() * S.inverse() * residual;
  last_nis = nis;
  data["nis"] = nis;

  constexpr double nis_threshold = 0.711;
  if (nis > nis_threshold) {
    data["nis_fail"] = 1.0;
    recent_nis_failures.push_back(1);
  } else {
    recent_nis_failures.push_back(0);
  }
  if (recent_nis_failures.size() > window_size) recent_nis_failures.pop_front();

  double denom = window_size > 0 ? static_cast<double>(window_size) : 1.0;
  double recent_rate = std::accumulate(
    recent_nis_failures.begin(), recent_nis_failures.end(), 0) / denom;
  data["recent_nis_failures"] = recent_rate;
}

}  // namespace tools
