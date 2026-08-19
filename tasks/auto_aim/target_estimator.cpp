#include "target_estimator.hpp"

#include <cmath>
#include <map>
#include <string>
#include <utility>

#include "tools/extended_kalman_filter.hpp"
#include "tools/invariant_pose_filter.hpp"
#include "tools/math_tools.hpp"
#include "tools/unscented_kalman_filter.hpp"

namespace auto_aim
{
struct TargetEstimator::Impl
{
  std::unique_ptr<tools::FilterBase> filter;
  TargetEstimatorDiagnostics diagnostics;
};

TargetEstimator::TargetEstimator() = default;

namespace
{
double diagnostic_value(const std::map<std::string, double> & data, const std::string & key)
{
  const auto it = data.find(key);
  return it == data.end() ? 0.0 : it->second;
}
}

TargetEstimator::TargetEstimator(
  const TargetState & initial_state, const Eigen::VectorXd & covariance_diagonal,
  const TargetEstimatorConfig & config)
: impl_{std::make_unique<Impl>()}
{
  const auto x0 = initial_state.vector();
  const auto P0 = covariance_diagonal.asDiagonal();
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    TargetState result(a + b);
    result.set_yaw(tools::limit_rad(result.yaw()));
    return result.vector();
  };

  if (config.method == FilterMethod::INEKF) {
    impl_->filter = std::make_unique<tools::InvariantPoseFilter>(x0, P0, x_add);
  } else if (config.method == FilterMethod::UKF) {
    impl_->filter = std::make_unique<tools::UnscentedKalmanFilter>(
      x0, P0, x_add, config.sigma_alpha, config.sigma_beta, config.sigma_kappa);
  } else {
    impl_->filter = std::make_unique<tools::ExtendedKalmanFilter>(x0, P0, x_add);
  }
  refresh_diagnostics();
}

TargetEstimator::~TargetEstimator() = default;

TargetEstimator::TargetEstimator(const TargetEstimator & other)
: impl_{std::make_unique<Impl>()}
{
  if (other.impl_ && other.impl_->filter) {
    impl_->filter = other.impl_->filter->clone();
    impl_->diagnostics = other.impl_->diagnostics;
  }
}

TargetEstimator & TargetEstimator::operator=(const TargetEstimator & other)
{
  if (this != &other) {
    impl_ = std::make_unique<Impl>();
    if (other.impl_ && other.impl_->filter) {
      impl_->filter = other.impl_->filter->clone();
      impl_->diagnostics = other.impl_->diagnostics;
    }
  }
  return *this;
}

TargetEstimator::TargetEstimator(TargetEstimator && other) noexcept = default;

TargetEstimator & TargetEstimator::operator=(TargetEstimator && other) noexcept = default;

TargetState TargetEstimator::state() const
{
  return impl_ && impl_->filter ? TargetState(impl_->filter->x) : TargetState{};
}

Eigen::VectorXd TargetEstimator::state_vector() const
{
  return state().vector();
}

void TargetEstimator::set_state(const TargetState & state)
{
  if (impl_ && impl_->filter) impl_->filter->x = state.vector();
}

void TargetEstimator::predict(
  const Eigen::MatrixXd & transition_matrix, const Eigen::MatrixXd & process_noise,
  const Transition & transition)
{
  if (!impl_ || !impl_->filter) return;
  impl_->filter->predict(
    transition_matrix, process_noise, [&](const Eigen::VectorXd & state_vector) {
      return transition(TargetState(state_vector)).vector();
    });
}

bool TargetEstimator::update(
  const Eigen::VectorXd & observation, const Eigen::MatrixXd & jacobian,
  const Eigen::MatrixXd & observation_noise, const MeasurementModel & model,
  const ResidualFunction & residual)
{
  if (!impl_ || !impl_->filter) return false;

  const auto state_before_update = impl_->filter->x;
  const auto covariance_before_update = impl_->filter->P;
  impl_->filter->update(
    observation, jacobian, observation_noise,
    [&](const Eigen::VectorXd & state_vector) {
      return model(TargetState(state_vector));
    }, residual);
  refresh_diagnostics();
  if (!impl_->filter->x.allFinite() || !impl_->filter->P.allFinite() ||
      !std::isfinite(impl_->filter->last_nis)) {
    impl_->filter->x = state_before_update;
    impl_->filter->P = covariance_before_update;
    refresh_diagnostics();
    return false;
  }
  return true;
}

double TargetEstimator::last_nis() const
{
  return impl_ && impl_->filter ? impl_->filter->last_nis : 0.0;
}

bool TargetEstimator::has_bad_nis_convergence(double failure_rate) const
{
  return diagnostics().recent_nis_failures >= failure_rate;
}

const TargetEstimatorDiagnostics & TargetEstimator::diagnostics() const
{
  static const TargetEstimatorDiagnostics empty;
  return impl_ ? impl_->diagnostics : empty;
}

void TargetEstimator::refresh_diagnostics()
{
  if (!impl_ || !impl_->filter) return;
  const auto & data = impl_->filter->data;
  impl_->diagnostics.residual_yaw = diagnostic_value(data, "residual_yaw");
  impl_->diagnostics.residual_pitch = diagnostic_value(data, "residual_pitch");
  impl_->diagnostics.residual_distance = diagnostic_value(data, "residual_distance");
  impl_->diagnostics.residual_angle = diagnostic_value(data, "residual_angle");
  impl_->diagnostics.nis = impl_->filter->last_nis;
  impl_->diagnostics.nees = diagnostic_value(data, "nees");
  impl_->diagnostics.nis_fail = diagnostic_value(data, "nis_fail");
  impl_->diagnostics.nees_fail = diagnostic_value(data, "nees_fail");
  impl_->diagnostics.recent_nis_failures = diagnostic_value(data, "recent_nis_failures");
  impl_->diagnostics.nis_window_size = impl_->filter->window_size;
}
}  // namespace auto_aim
