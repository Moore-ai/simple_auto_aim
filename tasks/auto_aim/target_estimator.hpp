#ifndef AUTO_AIM__TARGET_ESTIMATOR_HPP
#define AUTO_AIM__TARGET_ESTIMATOR_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <memory>

#include "target_state.hpp"

namespace auto_aim
{

enum class FilterMethod { EKF, INEKF, UKF };

struct TargetEstimatorConfig
{
  FilterMethod method = FilterMethod::EKF;
  double sigma_alpha = 0.001;
  double sigma_beta = 2.0;
  double sigma_kappa = 0.0;
};

struct TargetEstimatorDiagnostics
{
  double residual_yaw = 0.0;
  double residual_pitch = 0.0;
  double residual_distance = 0.0;
  double residual_angle = 0.0;
  double nis = 0.0;
  double nees = 0.0;
  double nis_fail = 0.0;
  double nees_fail = 0.0;
  double recent_nis_failures = 0.0;
  std::size_t nis_window_size = 100;
};

class TargetEstimator
{
public:
  using Transition = std::function<TargetState(const TargetState &)>;
  using MeasurementModel = std::function<Eigen::VectorXd(const TargetState &)>;
  using ResidualFunction =
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)>;

  TargetEstimator();
  TargetEstimator(
    const TargetState & initial_state, const Eigen::VectorXd & covariance_diagonal,
    const TargetEstimatorConfig & config);
  ~TargetEstimator();

  TargetEstimator(const TargetEstimator & other);
  TargetEstimator & operator=(const TargetEstimator & other);
  TargetEstimator(TargetEstimator && other) noexcept;
  TargetEstimator & operator=(TargetEstimator && other) noexcept;

  TargetState state() const;
  Eigen::VectorXd state_vector() const;
  void set_state(const TargetState & state);

  void predict(
    const Eigen::MatrixXd & transition_matrix, const Eigen::MatrixXd & process_noise,
    const Transition & transition);
  bool update(
    const Eigen::VectorXd & observation, const Eigen::MatrixXd & jacobian,
    const Eigen::MatrixXd & observation_noise, const MeasurementModel & model,
    const ResidualFunction & residual);

  double last_nis() const;
  bool has_bad_nis_convergence(double failure_rate = 0.4) const;
  const TargetEstimatorDiagnostics & diagnostics() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  void refresh_diagnostics();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_ESTIMATOR_HPP
