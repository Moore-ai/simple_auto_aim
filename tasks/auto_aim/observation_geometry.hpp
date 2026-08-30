#ifndef AUTO_AIM__OBSERVATION_GEOMETRY_HPP
#define AUTO_AIM__OBSERVATION_GEOMETRY_HPP

#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

#include <cmath>
#include <limits>
#include <optional>
#include <vector>

#include "armor.hpp"
#include "target_state.hpp"

namespace auto_aim
{

class Solver;

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

struct PredictedLightbarObservation
{
  cv::Point2f top;
  cv::Point2f bottom;
  Eigen::Vector4d uvl = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix<double, 4, TargetState::dimension> jacobian =
    Eigen::Matrix<double, 4, TargetState::dimension>::Zero();
  bool valid = false;
};

struct PredictedDepthDifference
{
  double value = std::numeric_limits<double>::quiet_NaN();
  Eigen::RowVector<double, TargetState::dimension> jacobian =
    Eigen::RowVector<double, TargetState::dimension>::Zero();
  bool valid = false;
};

// Geometry seam for image observations. It owns projection-to-observation
// conversion and geometric matching metrics; assignment lives in ObservationMatcher.
class ObservationGeometry
{
public:
  explicit ObservationGeometry(const Solver & solver);

  PredictedLightbarObservation project_lightbar(
    const Eigen::Vector3d & armor_center, double armor_yaw, bool right,
    ArmorType armor_type, ArmorName armor_name) const;

  PredictedLightbarObservation project_lightbar(
    const Eigen::Vector3d & armor_center, double armor_yaw, bool right,
    ArmorType armor_type, ArmorName armor_name,
    const Eigen::Matrix<double, 3, TargetState::dimension> & center_jacobian) const;

  PredictedDepthDifference project_light_depth_difference(
    const Eigen::Vector3d & armor_center, double armor_yaw, ArmorType armor_type,
    ArmorName armor_name,
    const Eigen::Matrix<double, 3, TargetState::dimension> & center_jacobian) const;

  std::optional<double> armor_match_cost(
    const std::vector<cv::Point2f> & measured_points, const Eigen::Vector3d & armor_center,
    double armor_yaw, ArmorType armor_type, ArmorName armor_name,
    const ReprojectionObservationConfig & config, bool all_armor_ids_seen) const;

  std::optional<double> lightbar_match_cost(
    const Lightbar & measured, const PredictedLightbarObservation & predicted,
    const ReprojectionObservationConfig & config) const;

private:
  const Solver & solver_;
};

// UVL: lightbar angle in the image, pixel center x/y, and pixel length.
Eigen::Vector4d uvl_from_endpoints(const cv::Point2f & top, const cv::Point2f & bottom);

Eigen::Vector4d uvl_residual(
  const Eigen::Vector4d & observation, const Eigen::Vector4d & prediction);

Eigen::Vector4d uvl_noise_variances(
  double lightbar_length, const ReprojectionObservationConfig & config);

Eigen::VectorXd stack_uvl(const std::vector<Eigen::Vector4d> & observations);

UVLBatch make_uvl_batch(
  const std::vector<Eigen::Vector4d> & observations,
  const std::vector<Eigen::MatrixXd> & jacobians, double pixel_var, double length_var,
  double angle_var = 0.01);

}  // namespace auto_aim

#endif  // AUTO_AIM__OBSERVATION_GEOMETRY_HPP
