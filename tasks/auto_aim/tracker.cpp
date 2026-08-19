#include "tracker.hpp"

#include <cmath>
#include <tuple>

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
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = tools::load(config_path);
  enemy_color_ = (tools::read<std::string>(yaml, "enemy_color") == "red") ? Color::red : Color::blue;
  min_detect_count_ = tools::read<int>(yaml, "min_detect_count");
  max_temp_lost_count_ = tools::read<int>(yaml, "max_temp_lost_count");
  outpost_max_temp_lost_count_ = tools::read<int>(yaml, "outpost_max_temp_lost_count");
  normal_temp_lost_count_ = max_temp_lost_count_;

  // 读取优先级配置
  use_priority_ = yaml["use_priority"] ? yaml["use_priority"].as<bool>() : false;
  if (use_priority_ && yaml["priority_list"]) {
    auto list = yaml["priority_list"].as<std::vector<std::string>>();
    for (size_t i = 0; i < list.size(); i++) {
      bool matched = false;
      for (size_t j = 0; j < ARMOR_NAMES.size(); j++) {
        if (ARMOR_NAMES[j] == list[i]) {
          auto key = static_cast<ArmorName>(j);
          if (priority_map_.find(key) != priority_map_.end()) {
            tools::logger()->warn("[Tracker] Duplicate priority entry: {} at index {}", list[i], i);
          }
          priority_map_[key] = static_cast<ArmorPriority>(i + 1);
          matched = true;
          break;
        }
      }
      if (!matched) {
        tools::logger()->warn("[Tracker] Unknown armor name in priority_list: {}", list[i]);
      }
    }
  }

  // 读取滤波器共用参数（filter 段：过程噪声、P0、radius，EKF/InEKF 共用）
  auto filter_cfg = yaml["filter"];
  filter_config_.process_noise.accel_var = tools::read<double>(filter_cfg, "accel_var");
  filter_config_.process_noise.angular_accel_var = tools::read<double>(filter_cfg, "angular_accel_var");
  filter_config_.process_noise.outpost_accel_var = tools::read<double>(filter_cfg, "outpost_accel_var");
  filter_config_.process_noise.outpost_angular_accel_var = tools::read<double>(filter_cfg, "outpost_angular_accel_var");

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
    auto obs_cfg = yaml["ekf_obs"];
    filter_config_.ekf.yaw_var = tools::read<double>(obs_cfg, "yaw_var");
    filter_config_.ekf.pitch_var = tools::read<double>(obs_cfg, "pitch_var");
    filter_config_.ekf.armor_yaw_base = tools::read<double>(obs_cfg, "armor_yaw_base");
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
  P0_outpost_ = read_P0("outpost");
  P0_base_ = read_P0("base");

  radius_default_ = tools::read<double>(radius_cfg, "default");
  radius_outpost_ = tools::read<double>(radius_cfg, "outpost");
  radius_base_ = tools::read<double>(radius_cfg, "base");
}

std::string Tracker::state() const { return state_; }

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
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  sort_armors(armors);

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
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

  std::list<Target> targets = {target_};
  return targets;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  DetectionResult detections{armors, {}};
  auto result = track(detection_queue, detections, t, use_enemy_color);
  armors = std::move(detections.armors);
  return result;
}

std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue,
  DetectionResult & detections, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 对全向感知队列中的目标也应用 tracker 的优先级方案，保证比较一致性
  if (use_priority_ && !temp_target.armors.empty()) {
    assign_priorities(temp_target.armors);
  }

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  auto & armors = detections.armors;
  sort_armors(armors);

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(detections, t);
  }

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
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

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
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

bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  if (!solver_.solve(armor)) {
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
    target_ = Target(
      armor, t, radius_outpost_, 3, P0_outpost_, filter_config_, filter_method_,
      observation_path_.uses_reprojection(), observation_path_.reprojection_config(),
      geometry_prior);
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

void Tracker::sort_armors(std::list<Armor> & armors) const
{
  if (use_priority_) {
    assign_priorities(armors);
    cv::Point2f img_center((float)1440 / 2, (float)1080 / 2);  // TODO
    armors.sort([img_center](const Armor & a, const Armor & b) {
      if (a.priority != b.priority) return a.priority < b.priority;
      return cv::norm(a.center - img_center) < cv::norm(b.center - img_center);
    });
  } else {
    armors.sort([](const Armor & a, const Armor & b) {
      cv::Point2f img_center((float)1440 / 2, (float)1080 / 2);  // TODO
      auto distance_1 = cv::norm(a.center - img_center);
      auto distance_2 = cv::norm(b.center - img_center);
      return distance_1 < distance_2;
    });
    for (auto & armor : armors) armor.priority = ArmorPriority::fifth;
  }
}

void Tracker::assign_priorities(std::list<Armor> & armors) const
{
  for (auto & armor : armors) {
    auto it = priority_map_.find(armor.name);
    armor.priority = (it != priority_map_.end()) ? it->second : static_cast<ArmorPriority>(priority_map_.size() + 1);
  }
}

}  // namespace auto_aim
