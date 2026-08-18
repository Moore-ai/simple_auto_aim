#include "observation_geometry.hpp"

#include <algorithm>
#include <cmath>

#include "solver.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
Eigen::Matrix<double, 4, TargetState::dimension> parameter_jacobian(
  const Eigen::Matrix<double, 3, TargetState::dimension> & center_jacobian)
{
  Eigen::Matrix<double, 4, TargetState::dimension> result =
    Eigen::Matrix<double, 4, TargetState::dimension>::Zero();
  result.topRows<3>() = center_jacobian;
  result(3, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = 1.0;
  return result;
}

cv::Point2f polygon_center(const std::vector<cv::Point2f> & points)
{
  cv::Point2f center;
  for (const auto & point : points) center += point;
  return center * (1.0F / static_cast<float>(points.size()));
}

double edge_angle(const cv::Point2f & lhs, const cv::Point2f & rhs)
{
  const auto delta = rhs - lhs;
  return std::atan2(static_cast<double>(delta.y), static_cast<double>(delta.x));
}
}  // namespace

Eigen::Vector4d uvl_from_endpoints(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const cv::Point2f delta = bottom - top;
  return {
    std::atan2(static_cast<double>(delta.x), static_cast<double>(delta.y)),
    (static_cast<double>(top.x) + bottom.x) * 0.5,
    (static_cast<double>(top.y) + bottom.y) * 0.5,
    std::sqrt(static_cast<double>(delta.x) * delta.x + static_cast<double>(delta.y) * delta.y)};
}

Eigen::Vector4d uvl_residual(
  const Eigen::Vector4d & observation, const Eigen::Vector4d & prediction)
{
  Eigen::Vector4d residual = observation - prediction;
  residual[0] = tools::limit_rad(residual[0]);
  return residual;
}

Eigen::Vector4d uvl_noise_variances(
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

Eigen::VectorXd stack_uvl(const std::vector<Eigen::Vector4d> & observations)
{
  Eigen::VectorXd result(static_cast<Eigen::Index>(observations.size() * 4));
  for (std::size_t i = 0; i < observations.size(); ++i) {
    result.segment<4>(static_cast<Eigen::Index>(i * 4)) = observations[i];
  }
  return result;
}

UVLBatch make_uvl_batch(
  const std::vector<Eigen::Vector4d> & observations,
  const std::vector<Eigen::MatrixXd> & jacobians, double pixel_var, double length_var,
  double angle_var)
{
  const auto count = observations.size();
  UVLBatch batch;
  batch.z = stack_uvl(observations);
  batch.H = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(count * 4), TargetState::dimension);
  batch.R = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(count * 4), static_cast<Eigen::Index>(count * 4));
  for (std::size_t i = 0; i < count; ++i) {
    batch.H.block<4, TargetState::dimension>(static_cast<Eigen::Index>(i * 4), 0) = jacobians[i];
    batch.R.diagonal().segment<4>(static_cast<Eigen::Index>(i * 4)) =
      Eigen::Vector4d{angle_var, pixel_var, pixel_var, length_var};
  }
  return batch;
}

ObservationGeometry::ObservationGeometry(const Solver & solver) : solver_{solver} {}

PredictedLightbarObservation ObservationGeometry::project_lightbar(
  const Eigen::Vector3d & armor_center, double armor_yaw, bool right,
  ArmorType armor_type, ArmorName armor_name) const
{
  const auto center_jacobian =
    Eigen::Matrix<double, 3, TargetState::dimension>::Zero();
  return project_lightbar(
    armor_center, armor_yaw, right, armor_type, armor_name, center_jacobian);
}

PredictedLightbarObservation ObservationGeometry::project_lightbar(
  const Eigen::Vector3d & armor_center, double armor_yaw, bool right,
  ArmorType armor_type, ArmorName armor_name,
  const Eigen::Matrix<double, 3, TargetState::dimension> & center_jacobian) const
{
  PredictedLightbarObservation result;
  const auto projection = solver_.project_armor_with_jacobian(
    armor_center, armor_yaw, armor_type, armor_name);
  if (!projection.valid || projection.points.size() != 4 || projection.point_jacobian.size() != 4) {
    return result;
  }

  const auto top_index = right ? 1U : 0U;
  const auto bottom_index = right ? 2U : 3U;
  const auto delta = projection.points[bottom_index] - projection.points[top_index];
  const auto dx = static_cast<double>(delta.x);
  const auto dy = static_cast<double>(delta.y);
  const auto length_squared = dx * dx + dy * dy;
  if (length_squared <= 1e-12) return result;

  const auto state_parameter_jacobian = parameter_jacobian(center_jacobian);
  const auto top_jacobian = projection.point_jacobian[top_index] * state_parameter_jacobian;
  const auto bottom_jacobian = projection.point_jacobian[bottom_index] * state_parameter_jacobian;
  const auto delta_x_jacobian = bottom_jacobian.row(0) - top_jacobian.row(0);
  const auto delta_y_jacobian = bottom_jacobian.row(1) - top_jacobian.row(1);

  result.top = projection.points[top_index];
  result.bottom = projection.points[bottom_index];
  result.uvl = uvl_from_endpoints(result.top, result.bottom);
  result.jacobian.row(0) =
    (dy * delta_x_jacobian - dx * delta_y_jacobian) / length_squared;
  result.jacobian.row(1) =
    (top_jacobian.row(0) + bottom_jacobian.row(0)) * 0.5;
  result.jacobian.row(2) =
    (top_jacobian.row(1) + bottom_jacobian.row(1)) * 0.5;
  result.jacobian.row(3) =
    (dx * delta_x_jacobian + dy * delta_y_jacobian) / std::sqrt(length_squared);
  result.valid = result.uvl.allFinite() && result.jacobian.allFinite();
  return result;
}

PredictedDepthDifference ObservationGeometry::project_light_depth_difference(
  const Eigen::Vector3d & armor_center, double armor_yaw, ArmorType armor_type,
  ArmorName armor_name,
  const Eigen::Matrix<double, 3, TargetState::dimension> & center_jacobian) const
{
  PredictedDepthDifference result;
  const auto projection = solver_.project_armor_with_jacobian(
    armor_center, armor_yaw, armor_type, armor_name);
  if (!projection.valid) return result;

  result.value = projection.light_depth_diff;
  result.jacobian = projection.light_depth_diff_jacobian * parameter_jacobian(center_jacobian);
  result.valid = std::isfinite(result.value) && result.jacobian.allFinite();
  return result;
}

std::optional<double> ObservationGeometry::armor_match_cost(
  const std::vector<cv::Point2f> & measured_points, const Eigen::Vector3d & armor_center,
  double armor_yaw, ArmorType armor_type, ArmorName armor_name,
  const ReprojectionObservationConfig & config, bool all_armor_ids_seen) const
{
  if (measured_points.size() != 4) return std::nullopt;
  const auto projected = solver_.reproject_armor(
    armor_center, armor_yaw, armor_type, armor_name);
  if (projected.size() != 4) return std::nullopt;

  const auto measured_center = polygon_center(measured_points);
  const auto projected_center = polygon_center(projected);
  const auto center_error = cv::norm(measured_center - projected_center);
  double angle_error = 0.0;
  double measured_perimeter = 0.0;
  double projected_perimeter = 0.0;
  for (int edge = 0; edge < 4; ++edge) {
    const auto next = (edge + 1) % 4;
    angle_error += std::abs(tools::limit_rad(
      edge_angle(measured_points[edge], measured_points[next]) -
      edge_angle(projected[edge], projected[next])));
    measured_perimeter += cv::norm(measured_points[edge] - measured_points[next]);
    projected_perimeter += cv::norm(projected[edge] - projected[next]);
  }
  const auto side_length_error = std::abs(projected_perimeter - measured_perimeter) /
    std::max(projected_perimeter, 1e-6);
  const auto cost = config.armor_match_w_center_err * center_error +
    config.armor_match_w_angle_err * angle_error +
    config.armor_match_w_side_length_err * side_length_error;
  const auto gate = all_armor_ids_seen ? config.armor_match_gate :
                                         config.armor_match_gate_not_all_init;
  if (!std::isfinite(cost) || cost >= gate) return std::nullopt;
  return cost;
}

std::optional<double> ObservationGeometry::lightbar_match_cost(
  const Lightbar & measured, const PredictedLightbarObservation & predicted,
  const ReprojectionObservationConfig & config) const
{
  if (measured.top == measured.bottom || !predicted.valid) return std::nullopt;

  const auto observed_uvl = uvl_from_endpoints(measured.top, measured.bottom);
  const auto length_ratio = observed_uvl[3] / std::max(predicted.uvl[3], 1.0);
  const auto angle_error = std::abs(tools::limit_rad(observed_uvl[0] - predicted.uvl[0]));
  const auto position_error = cv::norm(measured.top - predicted.top) +
    cv::norm(measured.bottom - predicted.bottom);
  if (std::abs(length_ratio - 1.0) > config.light_match_length_ratio_gate ||
      angle_error > config.light_match_angle_gate ||
      position_error > predicted.uvl[3] * config.light_match_pos_gate_by_length_ratio) {
    return std::nullopt;
  }
  return position_error;
}
}  // namespace auto_aim
