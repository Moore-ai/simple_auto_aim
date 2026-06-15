#ifndef TOOLS__FILTER_BASE_HPP
#define TOOLS__FILTER_BASE_HPP

#include <Eigen/Dense>
#include <deque>
#include <functional>
#include <map>
#include <memory>

namespace tools
{

class FilterBase
{
public:
  Eigen::VectorXd x;
  Eigen::MatrixXd P;

  FilterBase() = default;

  FilterBase(
    const Eigen::VectorXd & x0, const Eigen::MatrixXd & P0,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add =
      [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) { return a + b; });

  virtual ~FilterBase() = default;

  virtual std::unique_ptr<FilterBase> clone() const = 0;

  virtual Eigen::VectorXd predict(const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q);

  virtual Eigen::VectorXd predict(
    const Eigen::MatrixXd & F, const Eigen::MatrixXd & Q,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> f);

  virtual Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract) = 0;

  std::map<std::string, double> data;
  std::deque<int> recent_nis_failures{0};
  size_t window_size = 100;
  double last_nis = 0.0;

protected:
  Eigen::MatrixXd I;
  std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> x_add;

  void compute_nis(const Eigen::VectorXd & residual, const Eigen::MatrixXd & S);
};

}  // namespace tools

#endif  // TOOLS__FILTER_BASE_HPP
