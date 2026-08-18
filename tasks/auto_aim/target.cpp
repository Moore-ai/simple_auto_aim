#include "target.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "tools/extended_kalman_filter.hpp"
#include "tools/invariant_pose_filter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/unscented_kalman_filter.hpp"
#include "reprojection.hpp"
#include "solver.hpp"

namespace auto_aim
{
Target::Target(const Target & other)
: name(other.name),
  armor_type(other.armor_type),
  priority(other.priority),
  jumped(other.jumped),
  last_id(other.last_id),
  isinit(other.isinit),
  armor_num_(other.armor_num_),
  switch_count_(other.switch_count_),
  update_count_(other.update_count_),
  seen_armor_ids_(other.seen_armor_ids_),
  config_(other.config_),
  filter_method_(other.filter_method_),
  reprojection_mode_(other.reprojection_mode_),
  reprojection_config_(other.reprojection_config_),
  is_switch_(other.is_switch_),
  is_converged_(other.is_converged_),
  t_(other.t_)
{
  if (other.filter_) {
    filter_ = other.filter_->clone();
  }
}

Target & Target::operator=(const Target & other)
{
  if (this != &other) {
    name = other.name;
    armor_type = other.armor_type;
    priority = other.priority;
    jumped = other.jumped;
    last_id = other.last_id;
    isinit = other.isinit;
    armor_num_ = other.armor_num_;
    switch_count_ = other.switch_count_;
    update_count_ = other.update_count_;
    seen_armor_ids_ = other.seen_armor_ids_;
    config_ = other.config_;
    filter_method_ = other.filter_method_;
    reprojection_mode_ = other.reprojection_mode_;
    reprojection_config_ = other.reprojection_config_;
    is_switch_ = other.is_switch_;
    is_converged_ = other.is_converged_;
    t_ = other.t_;
    if (other.filter_) {
      filter_ = other.filter_->clone();
    } else {
      filter_.reset();
    }
  }
  return *this;
}

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, const FilterConfig & filter_config, FilterMethod filter_method,
  bool reprojection_mode, ReprojectionObservationConfig reprojection_config)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  seen_armor_ids_(static_cast<std::size_t>(std::max(armor_num, 0)), false),
  config_(filter_config),
  filter_method_(filter_method),
  reprojection_mode_(reprojection_mode),
  reprojection_config_(std::move(reprojection_config)),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};  //初始化预测量
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  if (filter_method_ == FilterMethod::INEKF) {
    filter_ = std::make_unique<tools::InvariantPoseFilter>(x0, P0, x_add);
  } else if (filter_method_ == FilterMethod::UKF) {
    filter_ = std::make_unique<tools::UnscentedKalmanFilter>(
      x0, P0, x_add, config_.ukf.sigma_alpha, config_.ukf.sigma_beta, config_.ukf.sigma_kappa);
  } else {
    filter_ = std::make_unique<tools::ExtendedKalmanFilter>(x0, P0, x_add);
  }
}

Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4), filter_method_(FilterMethod::EKF)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  filter_ = std::make_unique<tools::ExtendedKalmanFilter>(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // 状态转移矩阵
  // clang-format off
  Eigen::MatrixXd F{
    {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
  };
  // clang-format on

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = config_.process_noise.outpost_accel_var;
    v2 = config_.process_noise.outpost_angular_accel_var;
  } else {
    v1 = config_.process_noise.accel_var;
    v2 = config_.process_noise.angular_accel_var;
  }
  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  // 预测过程噪声偏差的方差
  // clang-format off
  Eigen::MatrixXd Q{
    {a * v1, b * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {b * v1, c * v1,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, a * v1, b * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0, b * v1, c * v1,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, a * v1, b * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0, b * v1, c * v1,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, a * v2, b * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0, b * v2, c * v2, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0},
    {     0,      0,      0,      0,      0,      0,      0,      0, 0, 0, 0}
  };
  // clang-format on

  // 防止夹角求和出现异常值
  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->filter_->x[7]) > 2)
    this->filter_->x[7] = this->filter_->x[7] > 0 ? 2.51 : -2.51;

  filter_->predict(F, Q, f);
  if (reprojection_mode_) constrain_reprojection_state();
}

void Target::update(const Armor & armor)
{
  // 装甲板匹配
  int id;
  auto min_angle_error = 1e10;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();

  std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
  for (int i = 0; i < armor_num_; i++) {
    xyza_i_list.push_back({xyza_list[i], i});
  }

  std::sort(
    xyza_i_list.begin(), xyza_i_list.end(),
    [](const std::pair<Eigen::Vector4d, int> & a, const std::pair<Eigen::Vector4d, int> & b) {
      Eigen::Vector3d ypd1 = tools::xyz2ypd(a.first.head(3));
      Eigen::Vector3d ypd2 = tools::xyz2ypd(b.first.head(3));
      return ypd1[2] < ypd2[2];
    });

  // 取前3个distance最小的装甲板
  for (int i = 0; i < 3; i++) {
    const auto & xyza = xyza_i_list[i].first;
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));

    if (std::abs(angle_error) < std::abs(min_angle_error)) {
      id = xyza_i_list[i].second;
      min_angle_error = angle_error;
    }
  }

  if (id != 0) jumped = true;

  if (id != last_id) {
    is_switch_ = true;
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;

  last_id = id;
  mark_armor_id(id);
  update_count_++;

  update_filter(armor, id);
  if (reprojection_mode_) constrain_reprojection_state();
}

namespace
{
struct PredictedUVL
{
  Eigen::Vector4d value = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
  Eigen::Matrix<double, 4, 11> jacobian = Eigen::Matrix<double, 4, 11>::Zero();
  bool valid = false;
};

Eigen::Matrix<double, 4, 11> projection_parameter_jacobian(
  const Eigen::Matrix<double, 3, 11> & center_jacobian)
{
  Eigen::Matrix<double, 4, 11> result = Eigen::Matrix<double, 4, 11>::Zero();
  result.topRows<3>() = center_jacobian;
  result(3, 6) = 1.0;
  return result;
}

PredictedUVL make_predicted_uvl(
  const Eigen::Vector3d & center, const Eigen::VectorXd & x,
  const ReprojectionMeasurement & measurement, int armor_num,
  ArmorType armor_type, ArmorName armor_name, const Solver & solver,
  const Eigen::Matrix<double, 3, 11> & center_jacobian)
{
  PredictedUVL result;
  const auto armor_angle = tools::limit_rad(
    x[6] + measurement.armor_id * 2 * CV_PI / std::max(armor_num, 1));
  const auto projection = solver.project_armor_with_jacobian(
    center, armor_angle, armor_type, armor_name);
  if (!projection.valid || projection.points.size() != 4 || projection.point_jacobian.size() != 4) {
    return result;
  }

  const auto top_index = measurement.right ? 1U : 0U;
  const auto bottom_index = measurement.right ? 2U : 3U;
  const auto delta = projection.points[bottom_index] - projection.points[top_index];
  const auto dx = static_cast<double>(delta.x);
  const auto dy = static_cast<double>(delta.y);
  const auto length_squared = dx * dx + dy * dy;
  if (length_squared <= 1e-12) return result;

  const auto parameter_jacobian = projection_parameter_jacobian(center_jacobian);
  const auto top_jacobian = projection.point_jacobian[top_index] * parameter_jacobian;
  const auto bottom_jacobian = projection.point_jacobian[bottom_index] * parameter_jacobian;
  const auto delta_x_jacobian = bottom_jacobian.row(0) - top_jacobian.row(0);
  const auto delta_y_jacobian = bottom_jacobian.row(1) - top_jacobian.row(1);

  result.value = uvl_from_endpoints(
    projection.points[top_index], projection.points[bottom_index]);
  result.jacobian.row(0) = (dy * delta_x_jacobian - dx * delta_y_jacobian) / length_squared;
  result.jacobian.row(1) = (top_jacobian.row(0) + bottom_jacobian.row(0)) * 0.5;
  result.jacobian.row(2) = (top_jacobian.row(1) + bottom_jacobian.row(1)) * 0.5;
  result.jacobian.row(3) =
    (dx * delta_x_jacobian + dy * delta_y_jacobian) / std::sqrt(length_squared);
  result.valid = result.value.allFinite() && result.jacobian.allFinite();
  return result;
}
}  // namespace

bool Target::update_reprojection(
  const std::vector<ReprojectionMeasurement> & measurements, const Solver & solver,
  const ReprojectionObservationConfig & observation_config)
{
  return update_reprojection_impl(measurements, {}, solver, observation_config, false);
}

bool Target::update_reprojection(
  const std::vector<ReprojectionMeasurement> & measurements,
  const std::vector<ReprojectionArmorMeasurement> & armor_measurements, const Solver & solver,
  const ReprojectionObservationConfig & observation_config)
{
  return update_reprojection_impl(
    measurements, armor_measurements, solver, observation_config, false);
}

bool Target::update_lightbar_assist(
  const std::vector<ReprojectionMeasurement> & measurements, const Solver & solver,
  const ReprojectionObservationConfig & observation_config)
{
  return update_reprojection_impl(measurements, {}, solver, observation_config, true);
}

bool Target::update_reprojection_impl(
  const std::vector<ReprojectionMeasurement> & measurements,
  const std::vector<ReprojectionArmorMeasurement> & armor_measurements, const Solver & solver,
  const ReprojectionObservationConfig & observation_config, bool auxiliary_only)
{
  if (measurements.empty() || filter_method_ != FilterMethod::EKF || !filter_) return false;

  std::vector<Eigen::Vector4d> observations;
  std::vector<PredictedUVL> predictions;
  observations.reserve(measurements.size());
  predictions.reserve(measurements.size());
  for (const auto & measurement : measurements) {
    const auto observation = uvl_from_endpoints(
      measurement.lightbar.top, measurement.lightbar.bottom);
    if (!observation.allFinite() || observation[3] < 1.0) return false;
    const auto prediction = make_predicted_uvl(
      h_armor_xyz(filter_->x, measurement.armor_id), filter_->x, measurement,
      armor_num_, armor_type, name,
      solver, h_armor_xyz_jacobian(filter_->x, measurement.armor_id));
    if (!prediction.valid) return false;
    observations.push_back(observation);
    predictions.push_back(prediction);
  }

  std::optional<double> observed_depth_diff;
  if (armor_measurements.size() == 1) {
    if (const auto depth_diff = solver.armor_lights_depth_diff(armor_measurements.front().armor)) {
      const auto max_depth_difference = armor_type == ArmorType::big ? 0.23 : 0.135;
      if (std::abs(*depth_diff) <= max_depth_difference) observed_depth_diff = *depth_diff;
    }
  }

  const auto rows = static_cast<Eigen::Index>(observations.size() * 4 +
                                              (observed_depth_diff ? 1 : 0));
  Eigen::VectorXd z = Eigen::VectorXd::Zero(rows);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(rows, 11);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, rows);
  Eigen::Index row = 0;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    z.segment<4>(row) = observations[i];
    H.block<4, 11>(row, 0) = predictions[i].jacobian;
    R.diagonal().segment<4>(row) = uvl_noise_variances(
      observations[i][3], observation_config);
    row += 4;
  }

  if (observed_depth_diff) {
    const auto & armor_measurement = armor_measurements.front();
    const auto id = armor_measurement.armor_id;
    const auto center = h_armor_xyz(filter_->x, id);
    const auto armor_angle = tools::limit_rad(
      filter_->x[6] + id * 2 * CV_PI / std::max(armor_num_, 1));
    const auto projection = solver.project_armor_with_jacobian(
      center, armor_angle, armor_type, name);
    if (!projection.valid) return false;
    z[row] = *observed_depth_diff;
    H.row(row) = projection.light_depth_diff_jacobian * projection_parameter_jacobian(
      h_armor_xyz_jacobian(filter_->x, id));
    const auto sigma = observation_config.r_sigma_armor_lights_depth_diff;
    R(row, row) = sigma * sigma / 2.0;
  }

  auto h = [&](const Eigen::VectorXd & x) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(rows);
    Eigen::Index current_row = 0;
    for (const auto & measurement : measurements) {
      const auto prediction = make_predicted_uvl(
        h_armor_xyz(x, measurement.armor_id), x, measurement,
        armor_num_, armor_type, name,
        solver, h_armor_xyz_jacobian(x, measurement.armor_id));
      result.segment<4>(current_row) = prediction.value;
      current_row += 4;
    }
    if (observed_depth_diff) {
      const auto id = armor_measurements.front().armor_id;
      const auto projection = solver.project_armor_with_jacobian(
        h_armor_xyz(x, id),
        tools::limit_rad(x[6] + id * 2 * CV_PI / std::max(armor_num_, 1)),
        armor_type, name);
      result[current_row] = projection.light_depth_diff;
    }
    return result;
  };
  auto z_subtract = [uvl_count = observations.size()](
                      const Eigen::VectorXd & observation,
                      const Eigen::VectorXd & prediction) {
    Eigen::VectorXd result = observation - prediction;
    for (std::size_t i = 0; i < uvl_count; ++i) {
      result[static_cast<Eigen::Index>(i * 4)] = tools::limit_rad(
        result[static_cast<Eigen::Index>(i * 4)]);
    }
    return result;
  };

  const auto x_before_update = filter_->x;
  const auto P_before_update = filter_->P;
  filter_->update(z, H, R, h, z_subtract);
  if (!filter_->x.allFinite() || !filter_->P.allFinite() || !std::isfinite(filter_->last_nis)) {
    filter_->x = x_before_update;
    filter_->P = P_before_update;
    return false;
  }
  if (!auxiliary_only) {
    last_id = measurements.front().armor_id;
    if (armor_measurements.empty()) {
      for (const auto & measurement : measurements) mark_armor_id(measurement.armor_id);
    } else {
      for (const auto & measurement : armor_measurements) mark_armor_id(measurement.armor_id);
    }
    update_count_ += static_cast<int>(measurements.size() + (observed_depth_diff ? 1 : 0));
    if (last_id != 0) jumped = true;
    constrain_reprojection_state();
  }

  if (config_.vel_clamp.enable) {
    filter_->x[1] = std::clamp(
      filter_->x[1], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[3] = std::clamp(
      filter_->x[3], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[5] = std::clamp(
      filter_->x[5], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[7] = std::clamp(
      filter_->x[7], -config_.vel_clamp.max_yaw_rate, config_.vel_clamp.max_yaw_rate);
  }
  return true;
}

void Target::update_filter(const Armor & armor, int id)
{
  if (filter_method_ == FilterMethod::INEKF) {
    // InEKF branch: uses H_xyza, xyz observation
    Eigen::MatrixXd H = h_jacobian_xyza(filter_->x, id);
    // 距离自适应观测噪声
    double d = armor.ypd_in_world[2];
    double denom = std::max(config_.inekf.dist_scale_denom, 1.0);
    double dist_scale = 1.0 + d * d / denom;
    Eigen::VectorXd R_dig{
      {config_.inekf.xy_var * dist_scale,
       config_.inekf.xy_var * dist_scale,
       config_.inekf.z_var * dist_scale,
       config_.inekf.yaw_var}
    };
    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
      Eigen::Vector3d xyz = h_armor_xyz(x, id);
      auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
      return {xyz[0], xyz[1], xyz[2], angle};
    };

    auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a - b;
      c[3] = tools::limit_rad(c[3]);  // yaw wrap only
      return c;
    };

    Eigen::VectorXd z{
      {armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2], armor.ypr_in_world[0]}
    };

    filter_->update(z, H, R, h, z_subtract);
  } else if (filter_method_ == FilterMethod::UKF) {
    // UKF branch: ignores H, uses sigma-point propagation through h
    double d = armor.ypd_in_world[2];
    double denom = std::max(config_.ukf.dist_scale_denom, 1.0);
    double dist_scale = 1.0 + d * d / denom;
    Eigen::VectorXd R_dig{
      {config_.ukf.xy_var * dist_scale,
       config_.ukf.xy_var * dist_scale,
       config_.ukf.z_var * dist_scale,
       config_.ukf.yaw_var}
    };
    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
      Eigen::Vector3d xyz = h_armor_xyz(x, id);
      auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
      return {xyz[0], xyz[1], xyz[2], angle};
    };

    auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a - b;
      c[3] = tools::limit_rad(c[3]);
      return c;
    };

    Eigen::VectorXd z{
      {armor.xyz_in_world[0], armor.xyz_in_world[1], armor.xyz_in_world[2], armor.ypr_in_world[0]}
    };

    filter_->update(z, Eigen::MatrixXd{}, R, h, z_subtract);
  } else {
    // EKF branch: uses ypda observation
    Eigen::MatrixXd H = h_jacobian(filter_->x, id);
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
    Eigen::VectorXd R_dig{
      {config_.ekf.yaw_var, config_.ekf.pitch_var, log(std::abs(delta_angle) + 1) + 1,
       log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + config_.ekf.armor_yaw_base}};

    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
      Eigen::VectorXd xyz = h_armor_xyz(x, id);
      Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
      auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
      return {ypd[0], ypd[1], ypd[2], angle};
    };

    auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a - b;
      c[0] = tools::limit_rad(c[0]);
      c[1] = tools::limit_rad(c[1]);
      c[3] = tools::limit_rad(c[3]);
      return c;
    };

    const Eigen::VectorXd & ypd = armor.ypd_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;
    Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

    filter_->update(z, H, R, h, z_subtract);
  }

  // 速度限幅：防止极端速度估计导致轨迹不平滑
  if (config_.vel_clamp.enable) {
    filter_->x[1] = std::clamp(filter_->x[1], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[3] = std::clamp(filter_->x[3], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[5] = std::clamp(filter_->x[5], -config_.vel_clamp.max_linear_speed, config_.vel_clamp.max_linear_speed);
    filter_->x[7] = std::clamp(filter_->x[7], -config_.vel_clamp.max_yaw_rate, config_.vel_clamp.max_yaw_rate);
  }
}

Eigen::VectorXd Target::ekf_x() const { return filter_->x; }

const tools::FilterBase & Target::filter() const {
    assert(filter_);
    return *filter_; 
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(filter_->x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(filter_->x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  const auto min_radius = reprojection_mode_ ? reprojection_config_.radius_min : 0.05;
  const auto max_radius = reprojection_mode_ ? reprojection_config_.radius_max : 0.5;
  const auto r = filter_->x[8];
  const auto l = filter_->x[8] + filter_->x[9];
  const auto inclusive = reprojection_mode_;
  const auto r_ok = inclusive ? (r >= min_radius && r <= max_radius) :
                                (r > min_radius && r < max_radius);
  const auto l_ok = inclusive ? (l >= min_radius && l <= max_radius) :
                                (l > min_radius && l < max_radius);

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", filter_->x[8], filter_->x[9]);
  return true;
}

void Target::mark_armor_id(int id)
{
  if (id >= 0 && id < static_cast<int>(seen_armor_ids_.size())) {
    seen_armor_ids_[static_cast<std::size_t>(id)] = true;
  }
}

bool Target::all_armor_ids_seen() const
{
  return !seen_armor_ids_.empty() &&
    std::all_of(seen_armor_ids_.begin(), seen_armor_ids_.end(), [](bool seen) { return seen; });
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::Matrix<double, 3, 11> Target::h_armor_xyz_jacobian(
  const Eigen::VectorXd & x, int id) const
{
  const auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto radius = use_l_h ? x[8] + x[9] : x[8];
  Eigen::Matrix<double, 3, 11> result = Eigen::Matrix<double, 3, 11>::Zero();
  result(0, 0) = 1.0;
  result(1, 2) = 1.0;
  result(2, 4) = 1.0;
  result(0, 6) = radius * std::sin(angle);
  result(1, 6) = -radius * std::cos(angle);
  result(0, 8) = -std::cos(angle);
  result(1, 8) = -std::sin(angle);
  if (use_l_h) {
    result(0, 9) = -std::cos(angle);
    result(1, 9) = -std::sin(angle);
    result(2, 10) = 1.0;
  }
  return result;
}

void Target::constrain_reprojection_state()
{
  if (!reprojection_mode_ || !filter_) return;
  const auto min_radius = std::max(reprojection_config_.radius_min, 1e-3);
  const auto max_radius = std::max(reprojection_config_.radius_max, min_radius);
  auto & x = filter_->x;
  if (!std::isfinite(x[8])) x[8] = min_radius;
  if (!std::isfinite(x[9])) x[9] = 0.0;
  const auto r1 = std::clamp(x[8], min_radius, max_radius);
  if (armor_num_ == 4) {
    const auto r2 = std::clamp(x[8] + x[9], min_radius, max_radius);
    x[8] = r1;
    x[9] = r2 - r1;
  } else {
    x[8] = r1;
  }
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0},
    {0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh},
    {0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0}
  };
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  // clang-format off
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {                0,                 0,                 0, 1}
  };
  // clang-format on

  return H_armor_ypda * H_armor_xyza;
}

Eigen::MatrixXd Target::h_jacobian_xyza(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);
  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  Eigen::MatrixXd H_xyza(4, 11);
  H_xyza <<
    1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl,     0,
    0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl,     0,
    0, 0, 0, 0, 1, 0,     0, 0,     0,     0, dz_dh,
    0, 0, 0, 0, 0, 0,     1, 0,     0,     0,     0;
  return H_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
