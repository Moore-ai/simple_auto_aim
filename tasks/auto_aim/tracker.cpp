#include "tracker.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <set>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"
#include "reprojection.hpp"

namespace auto_aim
{
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  use_reprojection_ = false;
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

  auto observation_mode = yaml["observation_mode"] ? yaml["observation_mode"].as<std::string>() : "pnp";
  debug_info_.observation_mode = "pnp";
  if (observation_mode == "reprojection") {
    if (filter_method_ == FilterMethod::EKF) {
      use_reprojection_ = true;
      debug_info_.observation_mode = "reprojection";
      auto observation_cfg = yaml["reprojection_obs"];
      if (observation_cfg) {
        auto read_if_present = [&observation_cfg](const std::string & key, double & value) {
          if (observation_cfg[key]) value = observation_cfg[key].as<double>();
        };
        read_if_present("angle_var", reprojection_config_.angle_var);
        read_if_present("pixel_var", reprojection_config_.pixel_var);
        read_if_present("length_var", reprojection_config_.length_var);
        read_if_present(
          "r_sigma_px_by_length_ratio", reprojection_config_.r_sigma_px_by_length_ratio);
        read_if_present(
          "r_sigma_length_by_length_ratio", reprojection_config_.r_sigma_length_by_length_ratio);
        read_if_present("r_sigma_angle", reprojection_config_.r_sigma_angle);
        read_if_present(
          "r_sigma_armor_lights_depth_diff", reprojection_config_.r_sigma_armor_lights_depth_diff);
        read_if_present("armor_match_gate", reprojection_config_.armor_match_gate);
        read_if_present(
          "armor_match_gate_not_all_init", reprojection_config_.armor_match_gate_not_all_init);
        read_if_present(
          "armor_match_w_center_err", reprojection_config_.armor_match_w_center_err);
        read_if_present("armor_match_w_angle_err", reprojection_config_.armor_match_w_angle_err);
        read_if_present(
          "armor_match_w_side_length_err", reprojection_config_.armor_match_w_side_length_err);
        read_if_present(
          "light_match_length_ratio_gate", reprojection_config_.light_match_length_ratio_gate);
        read_if_present("light_match_angle_gate", reprojection_config_.light_match_angle_gate);
        read_if_present(
          "light_match_pos_gate_by_length_ratio",
          reprojection_config_.light_match_pos_gate_by_length_ratio);
        read_if_present("radius_min", reprojection_config_.radius_min);
        read_if_present("radius_max", reprojection_config_.radius_max);

        const bool has_adaptive_noise =
          observation_cfg["r_sigma_px_by_length_ratio"] &&
          observation_cfg["r_sigma_length_by_length_ratio"] &&
          observation_cfg["r_sigma_angle"];
        reprojection_config_.use_adaptive_noise = has_adaptive_noise;
      }
      tools::logger()->info("[Tracker] Image reprojection observation enabled");
    } else {
      tools::logger()->warn(
        "[Tracker] observation_mode=reprojection requires EKF; falling back to PnP");
    }
  }

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
  }

  else {
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
  if (
    state_ != "lost" &&
    std::accumulate(
      target_.filter().recent_nis_failures.begin(), target_.filter().recent_nis_failures.end(), 0) >=
    (0.4 * target_.filter().window_size)) {
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
    debug_info_.last_update_source = "predict_only";
    debug_info_.predict_only_count++;
    return false;
  }

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    target_ = Target(
      armor, t, radius_default_, 2, P0_balance_, filter_config_, filter_method_,
      use_reprojection_, reprojection_config_);
  }

  else if (armor.name == ArmorName::outpost) {
    target_ = Target(
      armor, t, radius_outpost_, 3, P0_outpost_, filter_config_, filter_method_,
      use_reprojection_, reprojection_config_);
  }

  else if (armor.name == ArmorName::base) {
    target_ = Target(
      armor, t, radius_base_, 3, P0_base_, filter_config_, filter_method_,
      use_reprojection_, reprojection_config_);
  }

  else {
    target_ = Target(
      armor, t, radius_default_, 4, P0_default_, filter_config_, filter_method_,
      use_reprojection_, reprojection_config_);
  }

  debug_info_.last_update_source = "pnp_init";
  debug_info_.matched_armor_count = 1;
  debug_info_.matched_light_count = 0;
  debug_info_.uvl_observation_count = 0;
  debug_info_.diff_observation_count = 0;
  debug_info_.last_match_cost = 0.0;
  debug_info_.last_nis = 0.0;
  return true;
}

bool Tracker::update_target_reprojection_enhanced(
  DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);
  debug_info_.last_update_source = "predict_only";
  debug_info_.matched_armor_count = 0;
  debug_info_.matched_light_count = 0;
  debug_info_.uvl_observation_count = 0;
  debug_info_.diff_observation_count = 0;
  debug_info_.rejected_armor_count = 0;
  debug_info_.rejected_light_count = 0;
  debug_info_.last_match_cost = 0.0;
  debug_info_.last_nis = 0.0;

  const auto predicted_armors = target_.armor_xyza_list();
  const auto armor_count = static_cast<int>(predicted_armors.size());
  if (armor_count == 0) {
    debug_info_.predict_only_count++;
    return false;
  }

  std::vector<int> visible_ids(armor_count);
  std::iota(visible_ids.begin(), visible_ids.end(), 0);
  std::sort(visible_ids.begin(), visible_ids.end(), [&](int lhs, int rhs) {
    return solver_.armor_visibility_score(
             predicted_armors[lhs].head<3>(), predicted_armors[lhs][3], target_.name) >
      solver_.armor_visibility_score(
        predicted_armors[rhs].head<3>(), predicted_armors[rhs][3], target_.name);
  });
  if (visible_ids.size() > 3) visible_ids.resize(3);

  std::vector<Armor *> candidates;
  for (auto & armor : detections.armors) {
    if (armor.name == target_.name && armor.type == target_.armor_type && armor.points.size() == 4) {
      candidates.push_back(&armor);
    }
  }

  auto polygon_center = [](const std::vector<cv::Point2f> & points) {
    cv::Point2f center;
    for (const auto & point : points) center += point;
    return center * (1.0F / static_cast<float>(points.size()));
  };
  auto edge_angle = [](const cv::Point2f & lhs, const cv::Point2f & rhs) {
    const auto delta = rhs - lhs;
    return std::atan2(static_cast<double>(delta.y), static_cast<double>(delta.x));
  };

  constexpr double invalid_cost = 1e12;
  std::vector<std::vector<double>> armor_cost(
    candidates.size(), std::vector<double>(visible_ids.size(), invalid_cost));
  for (std::size_t obs = 0; obs < candidates.size(); ++obs) {
    const auto & measured = candidates[obs]->points;
    const auto measured_center = polygon_center(measured);
    bool in_gate = false;
    for (std::size_t candidate = 0; candidate < visible_ids.size(); ++candidate) {
      const auto id = visible_ids[candidate];
      const auto projected = solver_.reproject_armor(
        predicted_armors[id].head<3>(), predicted_armors[id][3],
        target_.armor_type, target_.name);
      if (projected.size() != 4) continue;

      const auto projected_center = polygon_center(projected);
      const auto center_error = cv::norm(measured_center - projected_center);
      double angle_error = 0.0;
      double measured_perimeter = 0.0;
      double projected_perimeter = 0.0;
      for (int edge = 0; edge < 4; ++edge) {
        const auto next = (edge + 1) % 4;
        angle_error += std::abs(tools::limit_rad(
          edge_angle(measured[edge], measured[next]) -
          edge_angle(projected[edge], projected[next])));
        measured_perimeter += cv::norm(measured[edge] - measured[next]);
        projected_perimeter += cv::norm(projected[edge] - projected[next]);
      }
      const auto side_length_error = std::abs(projected_perimeter - measured_perimeter) /
        std::max(projected_perimeter, 1e-6);
      const auto cost = reprojection_config_.armor_match_w_center_err * center_error +
        reprojection_config_.armor_match_w_angle_err * angle_error +
        reprojection_config_.armor_match_w_side_length_err * side_length_error;
      const auto gate = target_.all_armor_ids_seen()
        ? reprojection_config_.armor_match_gate
        : reprojection_config_.armor_match_gate_not_all_init;
      if (std::isfinite(cost) && cost < gate) {
        armor_cost[obs][candidate] = cost;
        in_gate = true;
      }
    }
    if (!in_gate) debug_info_.rejected_armor_count++;
  }

  std::vector<bool> used_observations(candidates.size(), false);
  std::vector<bool> used_predictions(visible_ids.size(), false);
  std::vector<std::pair<int, Armor *>> matched_armors;
  while (true) {
    double best_cost = invalid_cost;
    int best_observation = -1;
    int best_prediction = -1;
    for (std::size_t obs = 0; obs < candidates.size(); ++obs) {
      if (used_observations[obs]) continue;
      for (std::size_t candidate = 0; candidate < visible_ids.size(); ++candidate) {
        if (!used_predictions[candidate] && armor_cost[obs][candidate] < best_cost) {
          best_cost = armor_cost[obs][candidate];
          best_observation = static_cast<int>(obs);
          best_prediction = static_cast<int>(candidate);
        }
      }
    }
    if (best_observation < 0) break;
    used_observations[best_observation] = true;
    used_predictions[best_prediction] = true;
    matched_armors.emplace_back(visible_ids[best_prediction], candidates[best_observation]);
    debug_info_.last_match_cost = best_cost;
  }

  std::vector<ReprojectionMeasurement> measurements;
  std::vector<ReprojectionArmorMeasurement> armor_measurements;
  std::set<std::pair<int, bool>> used_sides;
  for (const auto & [id, armor] : matched_armors) {
    ReprojectionArmorMeasurement armor_measurement{id, *armor};
    armor_measurements.push_back(armor_measurement);

    Lightbar left, right;
    left.top = armor->points[0];
    left.bottom = armor->points[3];
    right.top = armor->points[1];
    right.bottom = armor->points[2];
    measurements.push_back({id, false, left});
    measurements.push_back({id, true, right});
    used_sides.emplace(id, false);
    used_sides.emplace(id, true);
  }

  if (!matched_armors.empty() && target_.name != ArmorName::base) {
    const auto closest_id = visible_ids.front();
    struct PredictedLight
    {
      int id;
      bool right;
      cv::Point2f top;
      cv::Point2f bottom;
    };
    std::vector<PredictedLight> visible_lights;
    auto add_light = [&](int id, bool right) {
      if (id < 0 || id >= armor_count || used_sides.count({id, right})) return;
      const auto projected = solver_.reproject_armor(
        predicted_armors[id].head<3>(), predicted_armors[id][3],
        target_.armor_type, target_.name);
      if (projected.size() != 4) return;
      visible_lights.push_back({id, right, right ? projected[1] : projected[0],
                                right ? projected[2] : projected[3]});
    };
    add_light((closest_id + armor_count - 1) % armor_count, true);
    add_light((closest_id + 1) % armor_count, false);
    add_light(closest_id, false);
    add_light(closest_id, true);

    std::vector<const Lightbar *> light_candidates;
    for (const auto & lightbar : detections.lightbars) {
      if (lightbar.color == enemy_color_ && lightbar.top != lightbar.bottom) {
        light_candidates.push_back(&lightbar);
      }
    }
    std::vector<std::vector<double>> light_cost(
      light_candidates.size(), std::vector<double>(visible_lights.size(), invalid_cost));
    for (std::size_t obs = 0; obs < light_candidates.size(); ++obs) {
      const auto observed = uvl_from_endpoints(
        light_candidates[obs]->top, light_candidates[obs]->bottom);
      bool in_gate = false;
      for (std::size_t candidate = 0; candidate < visible_lights.size(); ++candidate) {
        const auto & predicted = visible_lights[candidate];
        const auto predicted_uvl = uvl_from_endpoints(predicted.top, predicted.bottom);
        const auto length_ratio = observed[3] / std::max(predicted_uvl[3], 1.0);
        const auto angle_error = std::abs(tools::limit_rad(observed[0] - predicted_uvl[0]));
        const auto position_error = cv::norm(light_candidates[obs]->top - predicted.top) +
          cv::norm(light_candidates[obs]->bottom - predicted.bottom);
        if (std::abs(length_ratio - 1.0) > reprojection_config_.light_match_length_ratio_gate ||
            angle_error > reprojection_config_.light_match_angle_gate ||
            position_error > predicted_uvl[3] *
              reprojection_config_.light_match_pos_gate_by_length_ratio) {
          continue;
        }
        light_cost[obs][candidate] = position_error;
        in_gate = true;
      }
      if (!in_gate) debug_info_.rejected_light_count++;
    }

    std::vector<bool> used_lights(light_candidates.size(), false);
    std::vector<bool> used_visible_lights(visible_lights.size(), false);
    while (true) {
      double best_cost = invalid_cost;
      int best_observation = -1;
      int best_prediction = -1;
      for (std::size_t obs = 0; obs < light_candidates.size(); ++obs) {
        if (used_lights[obs]) continue;
        for (std::size_t candidate = 0; candidate < visible_lights.size(); ++candidate) {
          if (!used_visible_lights[candidate] && light_cost[obs][candidate] < best_cost) {
            best_cost = light_cost[obs][candidate];
            best_observation = static_cast<int>(obs);
            best_prediction = static_cast<int>(candidate);
          }
        }
      }
      if (best_observation < 0) break;
      used_lights[best_observation] = true;
      used_visible_lights[best_prediction] = true;
      const auto & prediction = visible_lights[best_prediction];
      measurements.push_back({
        prediction.id, prediction.right, *light_candidates[best_observation]});
      used_sides.emplace(prediction.id, prediction.right);
      debug_info_.last_match_cost = best_cost;
    }
  }

  debug_info_.matched_armor_count = matched_armors.size();
  debug_info_.matched_light_count = measurements.size() - matched_armors.size() * 2;
  debug_info_.uvl_observation_count = measurements.size();
  const auto observed_depth_diff = armor_measurements.size() == 1 ?
    solver_.armor_lights_depth_diff(armor_measurements.front().armor) : std::nullopt;
  const auto max_depth_difference = target_.armor_type == ArmorType::big ? 0.23 : 0.135;
  debug_info_.diff_observation_count = observed_depth_diff.has_value() &&
    std::abs(*observed_depth_diff) <= max_depth_difference;

  if (target_.update_reprojection(
      measurements, armor_measurements, solver_, reprojection_config_)) {
    debug_info_.last_update_source = "reprojection";
    debug_info_.last_nis = target_.filter().last_nis;
    tools::logger()->debug(
      "[Tracker] reprojection update: armors={}, lights={}, uvl={}, diff={}, nis={:.3f}",
      debug_info_.matched_armor_count, debug_info_.matched_light_count,
      debug_info_.uvl_observation_count, debug_info_.diff_observation_count,
      debug_info_.last_nis);
    return true;
  }

  Armor * fallback = matched_armors.empty() ?
    (candidates.empty() ? nullptr : candidates.front()) : matched_armors.front().second;
  if (fallback && solver_.solve(*fallback)) {
    target_.update(*fallback);
    debug_info_.last_update_source = "pnp_fallback";
    debug_info_.pnp_fallback_count++;
    tools::logger()->debug("[Tracker] reprojection fallback to PnP");
    return true;
  }

  debug_info_.predict_only_count++;
  tools::logger()->debug(
    "[Tracker] reprojection update unavailable: armors={}, lights={}",
    debug_info_.matched_armor_count, debug_info_.matched_light_count);
  return false;
}

bool Tracker::update_target(DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  if (use_reprojection_) {
    return update_target_reprojection_enhanced(detections, t);
  }

  target_.predict(t);
  auto & armors = detections.armors;
  debug_info_.last_update_source = "predict_only";
  debug_info_.matched_armor_count = 0;
  debug_info_.matched_light_count = 0;
  debug_info_.uvl_observation_count = 0;
  debug_info_.diff_observation_count = 0;
  debug_info_.rejected_armor_count = 0;
  debug_info_.rejected_light_count = 0;
  debug_info_.last_match_cost = 0.0;
  debug_info_.last_nis = 0.0;
  bool found = false;
  for (auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    if (!solver_.solve(armor)) continue;
    found = true;
    debug_info_.matched_armor_count++;
    target_.update(armor);
  }
  if (found) {
    debug_info_.last_update_source = "pnp";
    debug_info_.last_nis = target_.filter().last_nis;
  } else {
    debug_info_.predict_only_count++;
  }
  return found;
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
