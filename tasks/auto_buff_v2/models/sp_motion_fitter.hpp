// 从 Climber 6f8ae25 移植的诊断速度拟合，不参与 EKF 更新。
#pragma once

#include <Eigen/Dense>
#include <deque>
#include <random>
#include <vector>

namespace auto_buff_v2
{

class SPMotionFitter
{
public:
  struct Result
  {
    double A = 0.0;
    double omega = 0.0;
    double phi = 0.0;
    double C = 0.0;
    int inliers = 0;
  };
  Result best_result_;
  bool valid() const
  {
    return best_result_.inliers > 0;
  }
  void reset()
  {
    fit_data_.clear(); best_result_ = {};
  }

  SPMotionFitter(int max_iterations,
    double threshold,
    double min_omega,
    double max_omega,
    int history);

  void add_data(double t, double v);

  void fit();

  double sine_function(double t, double A, double omega, double phi, double C)
  {
    return A * std::sin(omega * t + phi) + C;
  }

private:
  int max_iterations_;
  int history_;
  double threshold_;
  double min_omega_;
  double max_omega_;
  std::mt19937 gen_;
  std::deque<std::pair<double, double>> fit_data_;

  bool fit_partial_model(
    const std::vector<std::pair<double, double>> & sample, double omega, Eigen::Vector3d & params);

  int evaluate_inliers(double A, double omega, double phi, double C);
};

}  // namespace auto_buff_v2
