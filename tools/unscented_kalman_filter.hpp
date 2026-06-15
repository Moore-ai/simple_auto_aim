#ifndef TOOLS__UNSCENTED_KALMAN_FILTER_HPP
#define TOOLS__UNSCENTED_KALMAN_FILTER_HPP

#include <memory>

#include "tools/filter_base.hpp"

namespace tools
{

class UnscentedKalmanFilter : public FilterBase
{
public:
  UnscentedKalmanFilter(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; },
    double alpha = 0.001, double beta = 2.0, double kappa = 0.0);

  Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f) override;

  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract) override;

  std::unique_ptr<FilterBase> clone() const override;

private:
  int n_;
  double alpha_;
  double beta_;
  double kappa_;
  double lambda_;
  Eigen::VectorXd Wm_;
  Eigen::VectorXd Wc_;

  void compute_weights();
  Eigen::MatrixXd generate_sigma_points() const;
};

}  // namespace tools

#endif  // TOOLS__UNSCENTED_KALMAN_FILTER_HPP
