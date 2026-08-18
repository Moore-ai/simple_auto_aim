#ifndef AUTO_AIM__REPROJECTION_HPP
#define AUTO_AIM__REPROJECTION_HPP

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

#include <cmath>
#include <vector>

#include "target_state.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{

struct ReprojectionObservationConfig
{
  // Legacy fixed variances. They remain the fallback for older YAML files.
  double angle_var = 0.01;
  double pixel_var = 16.0;
  double length_var = 25.0;

  // Awakening-style length-dependent standard deviations.
  bool use_adaptive_noise = false;
  double r_sigma_px_by_length_ratio = 0.2;
  double r_sigma_length_by_length_ratio = 0.5;
  double r_sigma_angle = 0.1;
  double r_sigma_armor_lights_depth_diff = 0.1;

  // Matching and reprojection-only state constraints.
  double armor_match_gate = 200.0;
  double armor_match_gate_not_all_init = 1000.0;
  double armor_match_w_center_err = 5.0;
  double armor_match_w_angle_err = 10.0;
  double armor_match_w_side_length_err = 1.0;
  double light_match_length_ratio_gate = 0.2;
  double light_match_angle_gate = 0.2;
  double light_match_pos_gate_by_length_ratio = 5.0;
  double radius_min = 0.05;
  double radius_max = 1.0;
};

struct UVLBatch
{
  Eigen::VectorXd z;
  Eigen::MatrixXd H;
  Eigen::MatrixXd R;
};

// UVL: lightbar angle in the image, pixel center x/y, and pixel length.
inline Eigen::Vector4d uvl_from_endpoints(
  const cv::Point2f & top, const cv::Point2f & bottom)
{
  const cv::Point2f delta = bottom - top;
  return {
    std::atan2(static_cast<double>(delta.x), static_cast<double>(delta.y)),
    (static_cast<double>(top.x) + bottom.x) * 0.5,
    (static_cast<double>(top.y) + bottom.y) * 0.5,
    std::sqrt(static_cast<double>(delta.x) * delta.x + static_cast<double>(delta.y) * delta.y)};
}

inline Eigen::Vector4d uvl_residual(
  const Eigen::Vector4d & observation, const Eigen::Vector4d & prediction)
{
  Eigen::Vector4d residual = observation - prediction;
  residual[0] = tools::limit_rad(residual[0]);
  return residual;
}

inline Eigen::Vector4d uvl_noise_variances(
  double lightbar_length, const ReprojectionObservationConfig & config)
{
  if (!config.use_adaptive_noise) {
    return {config.angle_var, config.pixel_var, config.pixel_var, config.length_var};
  }

  const double sigma_pixel = config.r_sigma_px_by_length_ratio * lightbar_length;
  const double sigma_length = config.r_sigma_length_by_length_ratio * lightbar_length;
  return {
    config.r_sigma_angle * config.r_sigma_angle / 2.0,
    sigma_pixel * sigma_pixel / 2.0,
    sigma_pixel * sigma_pixel / 2.0,
    sigma_length * sigma_length / 2.0};
}

inline Eigen::VectorXd stack_uvl(const std::vector<Eigen::Vector4d> & observations)
{
  Eigen::VectorXd result(static_cast<Eigen::Index>(observations.size() * 4));
  for (std::size_t i = 0; i < observations.size(); ++i) {
    result.segment<4>(static_cast<Eigen::Index>(i * 4)) = observations[i];
  }
  return result;
}

inline UVLBatch make_uvl_batch(
  const std::vector<Eigen::Vector4d> & observations,
  const std::vector<Eigen::MatrixXd> & jacobians, double pixel_var, double length_var,
  double angle_var = 0.01)
{
  const auto count = observations.size();
  UVLBatch batch;
  batch.z = stack_uvl(observations);
  batch.H = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(count * 4), TargetState::dimension);
  batch.R = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(count * 4), static_cast<Eigen::Index>(count * 4));
  for (std::size_t i = 0; i < count; ++i) {
    batch.H.block<4, TargetState::dimension>(static_cast<Eigen::Index>(i * 4), 0) = jacobians[i];
    batch.R.diagonal().segment<4>(static_cast<Eigen::Index>(i * 4)) =
      Eigen::Vector4d{angle_var, pixel_var, pixel_var, length_var};
  }
  return batch;
}

}  // namespace auto_aim

#endif  // AUTO_AIM__REPROJECTION_HPP
