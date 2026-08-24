#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "observation_geometry.hpp"
#include "outpost_target.hpp"
#include "outpost_target_v2.hpp"
#include "solver.hpp"

namespace auto_aim
{
Target::Target() = default;

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
  is_switch_(other.is_switch_),
  is_converged_(other.is_converged_),
  outpost_model_(other.outpost_model_ ? other.outpost_model_->clone() : nullptr),
  estimator_(other.estimator_),
  filter_method_(other.filter_method_),
  reprojection_mode_(other.reprojection_mode_),
  reprojection_config_(other.reprojection_config_),
  t_(other.t_)
{
}

Target::~Target() = default;

Target::Target(Target && other) noexcept = default;

Target & Target::operator=(Target && other) noexcept = default;

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
    outpost_model_ = other.outpost_model_ ? other.outpost_model_->clone() : nullptr;
    filter_method_ = other.filter_method_;
    estimator_ = other.estimator_;
    reprojection_mode_ = other.reprojection_mode_;
    reprojection_config_ = other.reprojection_config_;
    is_switch_ = other.is_switch_;
    is_converged_ = other.is_converged_;
    t_ = other.t_;
  }
  return *this;
}

Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig, const FilterConfig & filter_config, FilterMethod filter_method,
  bool reprojection_mode, ReprojectionObservationConfig reprojection_config,
  std::optional<TargetGeometryPrior> geometry_prior)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  armor_num_(armor.name == ArmorName::outpost ? OUTPOST_ARMOR_COUNT : armor_num),
  switch_count_(0),
  update_count_(0),
  seen_armor_ids_(
    static_cast<std::size_t>(
      std::max(armor.name == ArmorName::outpost ? OUTPOST_ARMOR_COUNT : armor_num, 0)),
    false),
  config_(filter_config),
  is_switch_(false),
  is_converged_(false),
  filter_method_(armor.name == ArmorName::outpost ? FilterMethod::EKF : filter_method),
  reprojection_mode_(armor.name == ArmorName::outpost ? false : reprojection_mode),
  reprojection_config_(std::move(reprojection_config)),
  t_(t)
{
  priority = armor.priority;
  if (name == ArmorName::outpost) {
    outpost_model_ = std::make_unique<OutpostTarget>(
      armor, P0_dig,
      OutpostFilterConfig{
        config_.process_noise.outpost_accel_var, config_.ekf.yaw_var, config_.ekf.pitch_var,
        config_.ekf.armor_yaw_base, config_.vel_clamp.enable,
        config_.vel_clamp.max_linear_speed});
    return;
  }

  const auto use_geometry_prior = geometry_prior && std::isfinite(geometry_prior->radius) &&
    geometry_prior->radius > 0.0 && std::isfinite(geometry_prior->radius_diff) &&
    std::isfinite(geometry_prior->height_diff);
  const auto r = use_geometry_prior ? geometry_prior->radius : radius;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  // 旋转中心的坐标
  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  TargetState initial_state;
  initial_state.set_center_x(center_x);
  initial_state.set_center_y(center_y);
  initial_state.set_center_z(center_z);
  initial_state.set_yaw(ypr[0]);
  initial_state.set_radius(r);
  if (use_geometry_prior && armor_num_ == 4) {
    initial_state.set_radius_diff(geometry_prior->radius_diff);
    initial_state.set_height_diff(geometry_prior->height_diff);
  }
  estimator_ = TargetEstimator(
    initial_state, P0_dig,
    {filter_method_, config_.ukf.sigma_alpha, config_.ukf.sigma_beta, config_.ukf.sigma_kappa});
}

Target Target::make_outpost(
  const Armor & armor, std::chrono::steady_clock::time_point t, const Eigen::VectorXd & P0_dig,
  const OutpostFilterConfig & config)
{
  Target result;
  result.name = ArmorName::outpost;
  result.armor_type = armor.type;
  result.priority = armor.priority;
  result.jumped = false;
  result.last_id = 0;
  result.isinit = false;
  result.armor_num_ = OUTPOST_ARMOR_COUNT;
  result.switch_count_ = 0;
  result.update_count_ = 0;
  result.seen_armor_ids_.assign(OUTPOST_ARMOR_COUNT, false);
  result.is_switch_ = false;
  result.is_converged_ = false;
  result.filter_method_ = FilterMethod::EKF;
  result.reprojection_mode_ = false;
  result.t_ = t;
  result.outpost_model_ = std::make_unique<OutpostTarget>(armor, P0_dig, config);
  return result;
}

Target Target::make_outpost_v2(
  const std::vector<Armor> & armors, std::chrono::steady_clock::time_point t,
  const Eigen::VectorXd & P0_dig, const OutpostTargetV2Config & config)
{
  Target result;
  if (armors.empty()) return result;
  result.name = ArmorName::outpost;
  result.armor_type = armors.front().type;
  result.priority = armors.front().priority;
  result.jumped = false;
  result.last_id = 0;
  result.isinit = false;
  result.armor_num_ = OUTPOST_ARMOR_COUNT;
  result.switch_count_ = 0;
  result.update_count_ = 0;
  result.seen_armor_ids_.assign(OUTPOST_ARMOR_COUNT, false);
  result.is_switch_ = false;
  result.is_converged_ = false;
  result.filter_method_ = FilterMethod::EKF;
  result.reprojection_mode_ = false;
  result.t_ = t;
  result.outpost_model_ = std::make_unique<OutpostTargetV2>(armors, P0_dig, config);
  return result;
}

Target::Target(double x, double vyaw, double radius, double h)
: armor_num_(4), filter_method_(FilterMethod::EKF)
{
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  TargetState initial_state;
  initial_state.set_center_x(x);
  initial_state.set_yaw_rate(vyaw);
  initial_state.set_radius(radius);
  initial_state.set_height_diff(h);
  estimator_ = TargetEstimator(initial_state, P0_dig, {FilterMethod::EKF});
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  if (outpost_model_ && t != t_) outpost_model_->begin_frame();
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  if (outpost_model_) {
    outpost_model_->predict(dt);
    return;
  }

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
  const auto v1 = config_.process_noise.accel_var;
  const auto v2 = config_.process_noise.angular_accel_var;
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
  auto transition = [&](const TargetState & state) {
    TargetState predicted(F * state.vector());
    predicted.set_yaw(tools::limit_rad(predicted.yaw()));
    return predicted;
  };

  estimator_.predict(F, Q, transition);
  if (reprojection_mode_) constrain_reprojection_state();
}

void Target::update(const Armor & armor)
{
  if (outpost_model_) {
    update_outpost({armor});
    return;
  }

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

bool Target::update_outpost(const std::vector<Armor> & armors)
{
  if (!outpost_model_) return false;
  const auto result = outpost_model_->update(armors);
  if (!result.updated) return false;
  const auto & ids = result.armor_ids.empty() ? std::vector<int>{result.armor_id} :
                                                result.armor_ids;
  for (const auto id : ids) {
    jumped = jumped || id != 0;
    is_switch_ = id != last_id;
    if (is_switch_) ++switch_count_;
    last_id = id;
    mark_armor_id(id);
    ++update_count_;
  }
  return true;
}

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
  if (name == ArmorName::outpost || measurements.empty() || filter_method_ != FilterMethod::EKF) {
    return false;
  }

  ObservationGeometry geometry(solver);
  std::vector<Eigen::Vector4d> observations;
  std::vector<PredictedLightbarObservation> predictions;
  observations.reserve(measurements.size());
  predictions.reserve(measurements.size());
  const auto state = estimator_.state();
  for (const auto & measurement : measurements) {
    const auto observation = uvl_from_endpoints(
      measurement.lightbar.top, measurement.lightbar.bottom);
    if (!observation.allFinite() || observation[3] < 1.0) return false;
    const auto armor_angle = tools::limit_rad(
      state.yaw() + measurement.armor_id * 2 * CV_PI / std::max(armor_num_, 1));
    const auto prediction = geometry.project_lightbar(
      h_armor_xyz(state, measurement.armor_id), armor_angle, measurement.right,
      armor_type, name, h_armor_xyz_jacobian(state, measurement.armor_id));
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
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(rows, TargetState::dimension);
  Eigen::MatrixXd R = Eigen::MatrixXd::Zero(rows, rows);
  Eigen::Index row = 0;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    z.segment<4>(row) = observations[i];
    H.block<4, TargetState::dimension>(row, 0) = predictions[i].jacobian;
    R.diagonal().segment<4>(row) = uvl_noise_variances(
      observations[i][3], observation_config);
    row += 4;
  }

  if (observed_depth_diff) {
    const auto & armor_measurement = armor_measurements.front();
    const auto id = armor_measurement.armor_id;
    const auto center = h_armor_xyz(state, id);
    const auto armor_angle = tools::limit_rad(
      state.yaw() + id * 2 * CV_PI / std::max(armor_num_, 1));
    const auto projection = geometry.project_light_depth_difference(
      center, armor_angle, armor_type, name, h_armor_xyz_jacobian(state, id));
    if (!projection.valid) return false;
    z[row] = *observed_depth_diff;
    H.row(row) = projection.jacobian;
    const auto sigma = observation_config.r_sigma_armor_lights_depth_diff;
    R(row, row) = sigma * sigma / 2.0;
  }

  auto h = [&](const TargetState & state) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(rows);
    Eigen::Index current_row = 0;
    for (const auto & measurement : measurements) {
      const auto armor_angle = tools::limit_rad(
        state.yaw() + measurement.armor_id * 2 * CV_PI / std::max(armor_num_, 1));
      const auto prediction = geometry.project_lightbar(
        h_armor_xyz(state, measurement.armor_id), armor_angle, measurement.right,
        armor_type, name, h_armor_xyz_jacobian(state, measurement.armor_id));
      result.segment<4>(current_row) = prediction.uvl;
      current_row += 4;
    }
    if (observed_depth_diff) {
      const auto id = armor_measurements.front().armor_id;
      const auto projection = geometry.project_light_depth_difference(
        h_armor_xyz(state, id),
        tools::limit_rad(state.yaw() + id * 2 * CV_PI / std::max(armor_num_, 1)),
        armor_type, name, h_armor_xyz_jacobian(state, id));
      result[current_row] = projection.value;
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

  if (!estimator_.update(z, H, R, h, z_subtract)) return false;
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

  constrain_velocity();
  return true;
}

void Target::update_filter(const Armor & armor, int id)
{
  const auto state = estimator_.state();
  if (filter_method_ == FilterMethod::INEKF) {
    // InEKF branch: uses H_xyza, xyz observation
    Eigen::MatrixXd H = h_jacobian_xyza(state, id);
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

    auto h = [&](const TargetState & state) -> Eigen::Vector4d {
      Eigen::Vector3d xyz = h_armor_xyz(state, id);
      auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
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

    estimator_.update(z, H, R, h, z_subtract);
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

    auto h = [&](const TargetState & state) -> Eigen::Vector4d {
      Eigen::Vector3d xyz = h_armor_xyz(state, id);
      auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
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

    estimator_.update(z, Eigen::MatrixXd{}, R, h, z_subtract);
  } else {
    // EKF branch: uses ypda observation
    Eigen::MatrixXd H = h_jacobian(state, id);
    auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
    Eigen::VectorXd R_dig{
      {config_.ekf.yaw_var, config_.ekf.pitch_var, log(std::abs(delta_angle) + 1) + 1,
       log(std::abs(armor.ypd_in_world[2]) + 1) / 200 + config_.ekf.armor_yaw_base}};

    Eigen::MatrixXd R = R_dig.asDiagonal();

    auto h = [&](const TargetState & state) -> Eigen::Vector4d {
      Eigen::VectorXd xyz = h_armor_xyz(state, id);
      Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
      auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
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

    estimator_.update(z, H, R, h, z_subtract);
  }

  // 速度限幅：防止极端速度估计导致轨迹不平滑
  constrain_velocity();
}

TargetState Target::state() const
{
  return outpost_model_ ? outpost_model_->compatibility_state() : estimator_.state();
}

std::optional<OutpostState> Target::outpost_state() const
{
  return outpost_model_ ? outpost_model_->outpost_state() : std::nullopt;
}

std::optional<OutpostStateV2> Target::outpost_state_v2() const
{
  return outpost_model_ ? outpost_model_->outpost_state_v2() : std::nullopt;
}

Eigen::VectorXd Target::state_vector() const
{
  return outpost_model_ ? outpost_model_->state_vector() : estimator_.state_vector();
}

Eigen::VectorXd Target::ekf_x() const { return state_vector(); }

double Target::last_nis() const
{
  return outpost_model_ ? outpost_model_->last_nis() : estimator_.last_nis();
}

const TargetEstimatorDiagnostics & Target::diagnostics() const
{
  return outpost_model_ ? outpost_model_->diagnostics() : estimator_.diagnostics();
}

bool Target::has_bad_nis_convergence(double failure_rate) const
{
  return outpost_model_ ? outpost_model_->has_bad_nis_convergence(failure_rate) :
                           estimator_.has_bad_nis_convergence(failure_rate);
}

bool Target::geometry_cache_ready() const
{
  if (name == ArmorName::outpost) return false;
  if (update_count_ <= 3) return false;
  if (diverged() || has_bad_nis_convergence()) return false;

  const auto state = estimator_.state();
  return state.all_finite();
}

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  if (outpost_model_) {
    std::vector<Eigen::Vector4d> result;
    for (const auto & pose : outpost_model_->armor_pose_list()) {
      result.push_back({pose.center.x(), pose.center.y(), pose.center.z(), pose.yaw});
    }
    return result;
  }

  std::vector<Eigen::Vector4d> _armor_xyza_list;
  const auto state = estimator_.state();

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(state.yaw() + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(state, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

std::vector<PredictedArmorPose> Target::armor_pose_list() const
{
  if (outpost_model_) return outpost_model_->armor_pose_list();
  std::vector<PredictedArmorPose> poses;
  const auto xyza_list = armor_xyza_list();
  poses.reserve(xyza_list.size());
  const auto pitch = armor_mount_pitch(name);
  for (const auto & xyza : xyza_list) {
    poses.push_back({xyza.head<3>(), xyza[3], pitch});
  }
  return poses;
}

bool Target::diverged() const
{
  if (outpost_model_) return !outpost_model_->all_finite();
  const auto min_radius = reprojection_mode_ ? reprojection_config_.radius_min : 0.05;
  const auto max_radius = reprojection_mode_ ? reprojection_config_.radius_max : 0.5;
  const auto state = estimator_.state();
  const auto r = state.radius();
  const auto l = state.radius(true);
  const auto inclusive = reprojection_mode_;
  const auto r_ok = inclusive ? (r >= min_radius && r <= max_radius) :
                                (r > min_radius && r < max_radius);
  const auto l_ok = inclusive ? (l >= min_radius && l <= max_radius) :
                                (l > min_radius && l < max_radius);

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", state.radius(), state.radius(true));
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
  if (
    outpost_model_ && outpost_model_->direction_locked() && update_count_ > 10 &&
    !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const TargetState & state, int id) const
{
  auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = state.radius(use_l_h);
  auto armor_x = state.center_x() - r * std::cos(angle);
  auto armor_y = state.center_y() - r * std::sin(angle);
  auto armor_z = state.armor_height(use_l_h);

  return {armor_x, armor_y, armor_z};
}

Eigen::Matrix<double, 3, TargetState::dimension> Target::h_armor_xyz_jacobian(
  const TargetState & state, int id) const
{
  const auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  const auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const auto radius = state.radius(use_l_h);
  Eigen::Matrix<double, 3, TargetState::dimension> result =
    Eigen::Matrix<double, 3, TargetState::dimension>::Zero();
  result(0, static_cast<Eigen::Index>(TargetStateComponent::center_x)) = 1.0;
  result(1, static_cast<Eigen::Index>(TargetStateComponent::center_y)) = 1.0;
  result(2, static_cast<Eigen::Index>(TargetStateComponent::center_z)) = 1.0;
  result(0, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = radius * std::sin(angle);
  result(1, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = -radius * std::cos(angle);
  result(0, static_cast<Eigen::Index>(TargetStateComponent::radius)) = -std::cos(angle);
  result(1, static_cast<Eigen::Index>(TargetStateComponent::radius)) = -std::sin(angle);
  if (use_l_h) {
    result(0, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = -std::cos(angle);
    result(1, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = -std::sin(angle);
    result(2, static_cast<Eigen::Index>(TargetStateComponent::height_diff)) = 1.0;
  }
  return result;
}

void Target::constrain_reprojection_state()
{
  if (!reprojection_mode_ || name == ArmorName::outpost) return;
  const auto min_radius = std::max(reprojection_config_.radius_min, 1e-3);
  const auto max_radius = std::max(reprojection_config_.radius_max, min_radius);
  auto state = estimator_.state();
  if (!std::isfinite(state.radius())) state.set_radius(min_radius);
  if (!std::isfinite(state.radius_diff())) state.set_radius_diff(0.0);
  const auto r1 = std::clamp(state.radius(), min_radius, max_radius);
  if (armor_num_ == 4) {
    const auto r2 = std::clamp(state.radius(true), min_radius, max_radius);
    state.set_radius(r1);
    state.set_radius_diff(r2 - r1);
  } else {
    state.set_radius(r1);
  }
  estimator_.set_state(state);
}

Eigen::MatrixXd Target::h_jacobian(const TargetState & state, int id) const
{
  auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = state.radius(use_l_h);
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  // clang-format off
  Eigen::MatrixXd H_armor_xyza = Eigen::MatrixXd::Zero(4, TargetState::dimension);
  H_armor_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::center_x)) = 1;
  H_armor_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::center_y)) = 1;
  H_armor_xyza(2, static_cast<Eigen::Index>(TargetStateComponent::center_z)) = 1;
  H_armor_xyza(3, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = 1;
  H_armor_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = dx_da;
  H_armor_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = dy_da;
  H_armor_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::radius)) = dx_dr;
  H_armor_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::radius)) = dy_dr;
  H_armor_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = dx_dl;
  H_armor_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = dy_dl;
  H_armor_xyza(2, static_cast<Eigen::Index>(TargetStateComponent::height_diff)) = dz_dh;
  // clang-format on

  Eigen::VectorXd armor_xyz = h_armor_xyz(state, id);
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

Eigen::MatrixXd Target::h_jacobian_xyza(const TargetState & state, int id) const
{
  auto angle = tools::limit_rad(state.yaw() + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  auto r = state.radius(use_l_h);
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);
  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;
  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  Eigen::MatrixXd H_xyza = Eigen::MatrixXd::Zero(4, TargetState::dimension);
  H_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::center_x)) = 1;
  H_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::center_y)) = 1;
  H_xyza(2, static_cast<Eigen::Index>(TargetStateComponent::center_z)) = 1;
  H_xyza(3, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = 1;
  H_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = dx_da;
  H_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::yaw)) = dy_da;
  H_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::radius)) = dx_dr;
  H_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::radius)) = dy_dr;
  H_xyza(0, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = dx_dl;
  H_xyza(1, static_cast<Eigen::Index>(TargetStateComponent::radius_diff)) = dy_dl;
  H_xyza(2, static_cast<Eigen::Index>(TargetStateComponent::height_diff)) = dz_dh;
  return H_xyza;
}

void Target::constrain_velocity()
{
  if (!config_.vel_clamp.enable) return;
  auto state = estimator_.state();
  state.set_velocity_x(std::clamp(
    state.velocity_x(), -config_.vel_clamp.max_linear_speed,
    config_.vel_clamp.max_linear_speed));
  state.set_velocity_y(std::clamp(
    state.velocity_y(), -config_.vel_clamp.max_linear_speed,
    config_.vel_clamp.max_linear_speed));
  state.set_velocity_z(std::clamp(
    state.velocity_z(), -config_.vel_clamp.max_linear_speed,
    config_.vel_clamp.max_linear_speed));
  state.set_yaw_rate(std::clamp(
    state.yaw_rate(), -config_.vel_clamp.max_yaw_rate,
    config_.vel_clamp.max_yaw_rate));
  estimator_.set_state(state);
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
