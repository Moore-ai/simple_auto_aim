#include "tracker.hpp"

#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  observation_path_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now())
{
  auto yaml = tools::load(config_path);
  image_width_ = tools::read<int>(yaml, "image_width");
  image_height_ = tools::read<int>(yaml, "image_height");
  min_detect_count_ = tools::read<int>(yaml, "min_detect_count");
  max_temp_lost_count_ = tools::read<int>(yaml, "max_temp_lost_count");
  outpost_max_temp_lost_count_ = tools::read<int>(yaml, "outpost_max_temp_lost_count");
  normal_temp_lost_count_ = max_temp_lost_count_;

  outpost_model_ = yaml["outpost_model"] ? yaml["outpost_model"].as<std::string>() : "current";
  if (outpost_model_ != "current" && outpost_model_ != "v2") {
    tools::logger()->warn("[Tracker] Unknown outpost_model '{}', using current", outpost_model_);
    outpost_model_ = "current";
  }
  const auto outpost_current = yaml["outpost_current"];
  outpost_current_config_.accel_var = tools::read<double>(outpost_current, "accel_var");
  outpost_current_config_.observation_yaw_var = tools::read<double>(outpost_current, "yaw_var");
  outpost_current_config_.observation_pitch_var = tools::read<double>(outpost_current, "pitch_var");
  outpost_current_config_.observation_armor_yaw_base =
    tools::read<double>(outpost_current, "armor_yaw_base");
  outpost_current_config_.velocity_clamp_enabled =
    tools::read<bool>(outpost_current, "velocity_clamp_enabled");
  outpost_current_config_.max_linear_speed =
    tools::read<double>(outpost_current, "max_linear_speed");
  const auto outpost_v2 = yaml["outpost_v2"];
  outpost_v2_config_.radius = tools::read<double>(outpost_v2, "radius");
  outpost_v2_config_.yaw_rate_magnitude = tools::read<double>(outpost_v2, "yaw_rate_magnitude");
  outpost_v2_config_.is_fit_yaw_rate = tools::read<bool>(outpost_v2, "is_fit_yaw_rate");
  outpost_v2_config_.q_xy = tools::read<double>(outpost_v2, "q_xy");
  outpost_v2_config_.q_z = tools::read<double>(outpost_v2, "q_z");
  outpost_v2_config_.q_yaw = tools::read<double>(outpost_v2, "q_yaw");
  outpost_v2_config_.q_yaw_rate = tools::read<double>(outpost_v2, "q_yaw_rate");
  outpost_v2_config_.q_dz = tools::read<double>(outpost_v2, "q_dz");
  outpost_v2_config_.r_yaw = tools::read<double>(outpost_v2, "r_yaw");
  outpost_v2_config_.r_pitch = tools::read<double>(outpost_v2, "r_pitch");
  outpost_v2_config_.r_distance = tools::read<double>(outpost_v2, "r_distance");
  outpost_v2_config_.r_armor_yaw = tools::read<double>(outpost_v2, "r_armor_yaw");
  outpost_v2_config_.frontal_angle_gate = tools::read<double>(outpost_v2, "frontal_angle_gate");
  outpost_v2_config_.yaw_match_gate = tools::read<double>(outpost_v2, "yaw_match_gate");
  outpost_v2_config_.position_match_gate = tools::read<double>(outpost_v2, "position_match_gate");
  outpost_v2_config_.mismatch_reset_count = tools::read<int>(outpost_v2, "mismatch_reset_count");
  outpost_v2_config_.direction_sample_count = tools::read<int>(outpost_v2, "direction_sample_count");
  outpost_v2_config_.direction_compare_interval =
    tools::read<int>(outpost_v2, "direction_compare_interval");
  outpost_v2_config_.direction_stable_threshold =
    tools::read<double>(outpost_v2, "direction_stable_threshold");
  outpost_v2_config_.direction_jump_threshold =
    tools::read<double>(outpost_v2, "direction_jump_threshold");

  // 读取滤波器共用参数（filter 段：过程噪声、P0、radius，EKF/InEKF 共用）
  auto filter_cfg = yaml["filter"];
  filter_config_.process_noise.accel_var = tools::read<double>(filter_cfg, "accel_var");
  filter_config_.process_noise.angular_accel_var = tools::read<double>(filter_cfg, "angular_accel_var");

  auto ekf_obs_cfg = yaml["ekf_obs"];
  filter_config_.ekf.yaw_var = tools::read<double>(ekf_obs_cfg, "yaw_var");
  filter_config_.ekf.pitch_var = tools::read<double>(ekf_obs_cfg, "pitch_var");
  filter_config_.ekf.armor_yaw_base = tools::read<double>(ekf_obs_cfg, "armor_yaw_base");

  // 读取滤波器类型和特有参数
  auto filter_method_str = yaml["filter_method"] ? yaml["filter_method"].as<std::string>() : "ekf";
  if (filter_method_str == "inekf") {
    filter_method_ = FilterMethod::INEKF;
    auto obs_cfg = yaml["inekf_obs"];
    filter_config_.inekf.xy_var = tools::read<double>(obs_cfg, "xy_var");
    filter_config_.inekf.z_var = tools::read<double>(obs_cfg, "z_var");
    filter_config_.inekf.yaw_var = tools::read<double>(obs_cfg, "yaw_var");
    filter_config_.inekf.dist_scale_denom = tools::read<double>(obs_cfg, "dist_scale_denom");
  } else if (filter_method_str == "ukf") {
    filter_method_ = FilterMethod::UKF;
    auto obs_cfg = yaml["ukf_obs"];
    filter_config_.ukf.xy_var = tools::read<double>(obs_cfg, "xy_var");
    filter_config_.ukf.z_var = tools::read<double>(obs_cfg, "z_var");
    filter_config_.ukf.yaw_var = tools::read<double>(obs_cfg, "yaw_var");
    filter_config_.ukf.dist_scale_denom = tools::read<double>(obs_cfg, "dist_scale_denom");
    filter_config_.ukf.sigma_alpha = tools::read<double>(obs_cfg, "sigma_alpha");
    filter_config_.ukf.sigma_beta = tools::read<double>(obs_cfg, "sigma_beta");
    filter_config_.ukf.sigma_kappa = tools::read<double>(obs_cfg, "sigma_kappa");
  } else {
    filter_method_ = FilterMethod::EKF;
  }

  const auto geometry_cache_config = yaml["target_geometry_cache"];
  geometry_cache_enabled_ = geometry_cache_config && geometry_cache_config["enabled"] ?
    geometry_cache_config["enabled"].as<bool>() : false;
  if (geometry_cache_enabled_ && filter_method_ != FilterMethod::EKF) {
    tools::logger()->warn(
      "[Tracker] target_geometry_cache requires EKF; keeping the geometry cache disabled");
    geometry_cache_enabled_ = false;
  }

  observation_path_.configure(
    ObservationPathConfig::from_yaml(yaml, filter_method_), enemy_color_);

  // 读取速度限幅配置
  auto vel_clamp_yaml = yaml["vel_clamp"];
  if (vel_clamp_yaml) {
    filter_config_.vel_clamp.enable = tools::read<bool>(vel_clamp_yaml, "enable");
    if (filter_config_.vel_clamp.enable) {
      filter_config_.vel_clamp.max_linear_speed = std::max(tools::read<double>(vel_clamp_yaml, "max_linear_speed"), 0.1);
      filter_config_.vel_clamp.max_yaw_rate = std::max(tools::read<double>(vel_clamp_yaml, "max_yaw_rate"), 0.1);
    }
  }

  // 读取 P0 和 radius（共用参数，filter 段下）
  auto P0_cfg = filter_cfg["P0"];
  auto radius_cfg = filter_cfg["radius"];

  auto read_P0 = [&](const std::string & key) -> Eigen::VectorXd {
    auto vec = P0_cfg[key].as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
  };

  P0_default_ = read_P0("default");
  P0_balance_ = read_P0("balance");
  auto read_outpost_P0 = [](const YAML::Node & config) -> Eigen::VectorXd {
    auto vec = config["P0"].as<std::vector<double>>();
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size());
  };
  P0_outpost_current_ = read_outpost_P0(outpost_current);
  P0_outpost_v2_ = read_outpost_P0(outpost_v2);
  P0_base_ = read_P0("base");

  radius_default_ = tools::read<double>(radius_cfg, "default");
  radius_base_ = tools::read<double>(radius_cfg, "base");
}

std::string Tracker::state() const { return state_; }

void Tracker::set_enemy_color(Color enemy_color)
{
  if (enemy_color_received_ && enemy_color_ == enemy_color) return;

  enemy_color_ = enemy_color;
  enemy_color_received_ = true;
  observation_path_.set_enemy_color(enemy_color);
  state_ = "lost";
  detect_count_ = 0;
  temp_lost_count_ = 0;
  tools::logger()->info("[Tracker] Enemy color switched to {}", COLORS[enemy_color_]);
}

std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  DetectionResult detections{armors, {}};
  auto targets = track(detections, t, use_enemy_color);
  armors = std::move(detections.armors);
  return targets;
}

std::list<Target> Tracker::track(
  DetectionResult & detections, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  debug_data_ = {};
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }
  // 过滤我方装甲板
  auto & armors = detections.armors;
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 过滤前哨站顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, OUTPOST_MOUNT_PITCH);
  // });

  // 优先选择靠近画面中心的装甲板
  sort_armors(armors);

  bool found;
  if (state_ == "lost") {
    found = set_target(detections, t);
  } else {
    found = update_target(detections, t);
  }

  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 收敛效果检测：
  if (state_ != "lost" && target_.has_bad_nis_convergence()) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  if (state_ == "lost") return {};

  update_debug_data(found);

  std::list<Target> targets = {target_};
  return targets;
}

void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

bool Tracker::set_target(
  DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  auto & armors = detections.armors;
  if (armors.empty()) return false;

  auto solve_armor = [&](Armor & candidate) {
    if (!solver_.solve(candidate)) return false;
    if (
      candidate.name == ArmorName::outpost &&
      observation_path_.outpost_distance_optimizer_enabled()) {
      solver_.optimize_outpost_distance(candidate, detections.lightbars);
    }
    return true;
  };

  auto & armor = armors.front();
  if (!solve_armor(armor)) {
    observation_path_.record_predict_only();
    return false;
  }

  const auto geometry_prior = geometry_prior_for(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    target_ = Target(
      armor, t, radius_default_, 2, P0_balance_, filter_config_, filter_method_,
      observation_path_.uses_reprojection(), observation_path_.reprojection_config(),
      geometry_prior);
  } else if (armor.name == ArmorName::outpost) {
    if (outpost_model_ == "v2") {
      std::vector<Armor> outpost_armors;
      for (auto & candidate : armors) {
        if (candidate.name != ArmorName::outpost || candidate.type != armor.type) continue;
        if (solve_armor(candidate)) outpost_armors.push_back(candidate);
      }
      target_ = Target::make_outpost_v2(outpost_armors, t, P0_outpost_v2_, outpost_v2_config_);
    } else {
      target_ = Target::make_outpost(armor, t, P0_outpost_current_, outpost_current_config_);
    }
  } else if (armor.name == ArmorName::base) {
    target_ = Target(
      armor, t, radius_base_, 3, P0_base_, filter_config_, filter_method_,
      observation_path_.uses_reprojection(), observation_path_.reprojection_config(),
      geometry_prior);
  } else {
    target_ = Target(
      armor, t, radius_default_, 4, P0_default_, filter_config_, filter_method_,
      observation_path_.uses_reprojection(), observation_path_.reprojection_config(),
      geometry_prior);
  }

  observation_path_.record_initialization();
  return true;
}

bool Tracker::update_target(DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  const auto found = observation_path_.update(target_, detections, t);
  if (found) update_geometry_cache();
  return found;
}

std::optional<TargetGeometryPrior> Tracker::geometry_prior_for(const Armor & armor) const
{
  if (!geometry_cache_enabled_) return std::nullopt;

  const auto it = geometry_cache_.find(armor.name);
  if (it == geometry_cache_.end()) return std::nullopt;
  const auto & entry = it->second;
  if (entry.armor_type != armor.type) return std::nullopt;
  if (!std::isfinite(entry.geometry.radius) || entry.geometry.radius <= 0.0 ||
      !std::isfinite(entry.geometry.radius_diff) ||
      !std::isfinite(entry.geometry.height_diff)) {
    return std::nullopt;
  }

  return entry.geometry;
}

void Tracker::update_geometry_cache()
{
  if (!geometry_cache_enabled_ || !target_.geometry_cache_ready()) return;

  const auto state = target_.state();
  if (!state.all_finite()) return;

  geometry_cache_.insert_or_assign(
    target_.name,
    GeometryCacheEntry{
      target_.armor_type, {state.radius(), state.radius_diff(), state.height_diff()}});
}

void Tracker::update_debug_data(bool has_current_observation)
{
  debug_data_.locked_armor = has_current_observation ? target_.locked_armor() : std::nullopt;
  debug_data_.predicted_world_armors = target_.armor_pose_list();
  debug_data_.ekf_converged = target_.convergened();
  if (target_.name == ArmorName::outpost) {
    debug_data_.outpost_snapshot = target_.outpost_snapshot();
  } else {
    debug_data_.target_state = target_.state();
    debug_data_.outpost_snapshot = std::nullopt;
  }
  debug_data_.armor_type = target_.armor_type;
  debug_data_.predicted_image_armors.reserve(debug_data_.predicted_world_armors.size());
  for (const auto & pose : debug_data_.predicted_world_armors) {
    debug_data_.predicted_image_armors.push_back(
      solver_.reproject_armor(pose.center, pose.yaw, pose.pitch, target_.armor_type));
  }
}

void Tracker::sort_armors(std::list<Armor> & armors) const
{
  const cv::Point2f img_center(
    static_cast<float>(image_width_) / 2, static_cast<float>(image_height_) / 2);
  armors.sort([img_center](const Armor & a, const Armor & b) {
    return cv::norm(a.center - img_center) < cv::norm(b.center - img_center);
  });
}

}  // namespace auto_aim
