#include "observation_path.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <yaml-cpp/yaml.h>

#include "solver.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
namespace
{

template<typename T>
void read_if_present(const YAML::Node & node, const std::string & key, T & value)
{
  if (node && node[key]) value = node[key].as<T>();
}

void load_reprojection_config(
  const YAML::Node & yaml, ReprojectionObservationConfig & config)
{
  const auto observation_cfg = yaml["reprojection_obs"];
  if (!observation_cfg) return;

  read_if_present(observation_cfg, "angle_var", config.angle_var);
  read_if_present(observation_cfg, "pixel_var", config.pixel_var);
  read_if_present(observation_cfg, "length_var", config.length_var);
  read_if_present(
    observation_cfg, "r_sigma_px_by_length_ratio", config.r_sigma_px_by_length_ratio);
  read_if_present(
    observation_cfg, "r_sigma_length_by_length_ratio", config.r_sigma_length_by_length_ratio);
  read_if_present(observation_cfg, "r_sigma_angle", config.r_sigma_angle);
  read_if_present(
    observation_cfg, "r_sigma_armor_lights_depth_diff", config.r_sigma_armor_lights_depth_diff);
  read_if_present(observation_cfg, "armor_match_gate", config.armor_match_gate);
  read_if_present(
    observation_cfg, "armor_match_gate_not_all_init", config.armor_match_gate_not_all_init);
  read_if_present(observation_cfg, "armor_match_w_center_err", config.armor_match_w_center_err);
  read_if_present(observation_cfg, "armor_match_w_angle_err", config.armor_match_w_angle_err);
  read_if_present(
    observation_cfg, "armor_match_w_side_length_err", config.armor_match_w_side_length_err);
  read_if_present(
    observation_cfg, "light_match_length_ratio_gate", config.light_match_length_ratio_gate);
  read_if_present(observation_cfg, "light_match_angle_gate", config.light_match_angle_gate);
  read_if_present(
    observation_cfg, "light_match_pos_gate_by_length_ratio",
    config.light_match_pos_gate_by_length_ratio);
  read_if_present(observation_cfg, "radius_min", config.radius_min);
  read_if_present(observation_cfg, "radius_max", config.radius_max);

  config.use_adaptive_noise =
    observation_cfg["r_sigma_px_by_length_ratio"] &&
    observation_cfg["r_sigma_length_by_length_ratio"] &&
    observation_cfg["r_sigma_angle"];
}

}  // namespace

ObservationPathConfig ObservationPathConfig::from_yaml(
  const YAML::Node & yaml, FilterMethod filter_method)
{
  ObservationPathConfig config;
  const auto observation_mode =
    yaml["observation_mode"] ? yaml["observation_mode"].as<std::string>() : "pnp";
  const auto pnp_config = yaml["pnp"];
  const auto pnp_assist_requested = pnp_config && pnp_config["enable_lightbar_assist"] ?
    pnp_config["enable_lightbar_assist"].as<bool>() : false;
  const auto pnp_yaw_refinement_requested = pnp_config && pnp_config["enable_yaw_refinement"] ?
    pnp_config["enable_yaw_refinement"].as<bool>() : false;
  const auto outpost_distance_optimizer_requested =
    pnp_config && pnp_config["enable_outpost_distance_optimizer"] ?
      pnp_config["enable_outpost_distance_optimizer"].as<bool>() : false;

  if (observation_mode == "reprojection") {
    if (filter_method == FilterMethod::EKF) {
      config.mode = ObservationMode::REPROJECTION;
      load_reprojection_config(yaml, config.reprojection);
      tools::logger()->info("[ObservationPath] Image reprojection observation enabled");
    } else {
      tools::logger()->warn(
        "[ObservationPath] observation_mode=reprojection requires EKF; falling back to PnP");
    }
  } else if (observation_mode == "pnp") {
    config.yaw_refinement_enabled = pnp_yaw_refinement_requested;
    config.outpost_distance_optimizer_enabled = outpost_distance_optimizer_requested;
    if (pnp_assist_requested && filter_method == FilterMethod::EKF) {
      config.lightbar_assist_enabled = true;
      load_reprojection_config(yaml, config.reprojection);
      tools::logger()->info("[ObservationPath] PnP lightbar assist enabled");
    } else if (pnp_assist_requested) {
      tools::logger()->warn(
        "[ObservationPath] pnp.enable_lightbar_assist requires EKF; keeping original PnP path");
    }
  }
  return config;
}

ObservationPath::ObservationPath(Solver & solver)
: solver_{solver}, geometry_{solver}, matcher_{geometry_}
{}

void ObservationPath::configure(const ObservationPathConfig & config, Color enemy_color)
{
  config_ = config;
  enemy_color_ = enemy_color;
  debug_info_.observation_mode = config_.mode == ObservationMode::REPROJECTION ?
    "reprojection" : "pnp";
  debug_info_.lightbar_assist_enabled = config_.lightbar_assist_enabled;
  debug_info_.yaw_refinement_enabled = config_.yaw_refinement_enabled;
  debug_info_.outpost_distance_optimizer_enabled = config_.outpost_distance_optimizer_enabled;
}

bool ObservationPath::uses_reprojection() const
{
  return config_.mode == ObservationMode::REPROJECTION;
}

bool ObservationPath::lightbar_assist_enabled() const
{
  return config_.lightbar_assist_enabled;
}

bool ObservationPath::outpost_distance_optimizer_enabled() const
{
  return config_.outpost_distance_optimizer_enabled;
}

const ReprojectionObservationConfig & ObservationPath::reprojection_config() const
{
  return config_.reprojection;
}

const ObservationPathDebugInfo & ObservationPath::debug_info() const
{
  return debug_info_;
}

void ObservationPath::record_initialization()
{
  debug_info_.last_update_source = "pnp_init";
  debug_info_.matched_armor_count = 1;
  debug_info_.matched_light_count = 0;
  debug_info_.uvl_observation_count = 0;
  debug_info_.diff_observation_count = 0;
  debug_info_.rejected_armor_count = 0;
  debug_info_.rejected_light_count = 0;
  debug_info_.last_match_cost = 0.0;
  debug_info_.last_nis = 0.0;
}

void ObservationPath::record_predict_only()
{
  debug_info_.last_update_source = "predict_only";
  debug_info_.predict_only_count++;
}

void ObservationPath::reset_frame_debug_info()
{
  debug_info_.last_update_source = "predict_only";
  debug_info_.matched_armor_count = 0;
  debug_info_.matched_light_count = 0;
  debug_info_.uvl_observation_count = 0;
  debug_info_.diff_observation_count = 0;
  debug_info_.rejected_armor_count = 0;
  debug_info_.rejected_light_count = 0;
  debug_info_.last_match_cost = 0.0;
  debug_info_.last_nis = 0.0;
}

bool ObservationPath::update(
  Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  if (target.name == ArmorName::outpost) return update_pnp(target, detections, t);
  if (uses_reprojection()) return update_reprojection(target, detections, t);
  return update_pnp(target, detections, t);
}

bool ObservationPath::update_reprojection(
  Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  target.predict(t);
  reset_frame_debug_info();

  const auto predicted_armors = target.armor_xyza_list();
  const auto armor_count = static_cast<int>(predicted_armors.size());
  if (armor_count == 0) {
    debug_info_.predict_only_count++;
    return false;
  }

  std::vector<int> visible_ids(armor_count);
  std::iota(visible_ids.begin(), visible_ids.end(), 0);
  std::sort(visible_ids.begin(), visible_ids.end(), [&](int lhs, int rhs) {
    return solver_.armor_visibility_score(
             predicted_armors[lhs].head<3>(), predicted_armors[lhs][3], target.name) >
      solver_.armor_visibility_score(
        predicted_armors[rhs].head<3>(), predicted_armors[rhs][3], target.name);
  });
  if (visible_ids.size() > 3) visible_ids.resize(3);

  ObservationMatchStats match_stats;
  const auto matched_armors = matcher_.match_armors(
    detections, target.name, target.armor_type, predicted_armors, visible_ids,
    config_.reprojection, target.all_armor_ids_seen(), match_stats);
  debug_info_.rejected_armor_count += match_stats.rejected_armor_count;
  debug_info_.last_match_cost = match_stats.last_match_cost;

  std::vector<ReprojectionMeasurement> measurements;
  std::vector<ReprojectionArmorMeasurement> armor_measurements;
  for (const auto & match : matched_armors) {
    armor_measurements.push_back({match.prediction_id, *match.observation});

    Lightbar left, right;
    left.top = match.observation->points[0];
    left.bottom = match.observation->points[3];
    right.top = match.observation->points[1];
    right.bottom = match.observation->points[2];
    measurements.push_back({match.prediction_id, false, left});
    measurements.push_back({match.prediction_id, true, right});
  }

  if (!matched_armors.empty() && target.name != ArmorName::base) {
    std::vector<int> occupied_ids;
    occupied_ids.reserve(matched_armors.size());
    for (const auto & match : matched_armors) {
      occupied_ids.push_back(match.prediction_id);
    }
    const auto independent_matches = matcher_.match_lightbars(
      detections, target.name, target.armor_type, predicted_armors, visible_ids, occupied_ids,
      config_.reprojection, enemy_color_, match_stats);
    for (const auto & match : independent_matches) {
      measurements.push_back({match.prediction_id, match.right, *match.observation});
    }
  }

  debug_info_.rejected_light_count += match_stats.rejected_light_count;
  debug_info_.last_match_cost = match_stats.last_match_cost;

  debug_info_.matched_armor_count = matched_armors.size();
  debug_info_.matched_light_count = measurements.size() - matched_armors.size() * 2;
  debug_info_.uvl_observation_count = measurements.size();
  const auto observed_depth_diff = armor_measurements.size() == 1 ?
    solver_.armor_lights_depth_diff(armor_measurements.front().armor) : std::nullopt;
  const auto max_depth_difference = target.armor_type == ArmorType::big ? 0.23 : 0.135;
  debug_info_.diff_observation_count = observed_depth_diff.has_value() &&
    std::abs(*observed_depth_diff) <= max_depth_difference;

  if (target.update_reprojection(
      measurements, armor_measurements, solver_, config_.reprojection)) {
    debug_info_.last_update_source = "reprojection";
    debug_info_.last_nis = target.last_nis();
    tools::logger()->debug(
      "[ObservationPath] reprojection update: armors={}, lights={}, uvl={}, diff={}, nis={:.3f}",
      debug_info_.matched_armor_count, debug_info_.matched_light_count,
      debug_info_.uvl_observation_count, debug_info_.diff_observation_count,
      debug_info_.last_nis);
    return true;
  }

  Armor * fallback = matched_armors.empty() ? nullptr : matched_armors.front().observation;
  if (!fallback) {
    for (auto & armor : detections.armors) {
      if (armor.name == target.name && armor.type == target.armor_type && armor.points.size() == 4) {
        fallback = &armor;
        break;
      }
    }
  }
  if (fallback && solver_.solve(*fallback)) {
    target.update(*fallback);
    debug_info_.last_update_source = "pnp_fallback";
    debug_info_.pnp_fallback_count++;
    tools::logger()->debug("[ObservationPath] reprojection fallback to PnP");
    return true;
  }

  debug_info_.predict_only_count++;
  tools::logger()->debug(
    "[ObservationPath] reprojection update unavailable: armors={}, lights={}",
    debug_info_.matched_armor_count, debug_info_.matched_light_count);
  return false;
}

bool ObservationPath::update_pnp(
  Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t)
{
  target.predict(t);
  reset_frame_debug_info();
  bool found = false;
  std::vector<int> occupied_ids;
  std::vector<Armor> outpost_armors;
  for (auto & armor : detections.armors) {
    if (armor.name != target.name || armor.type != target.armor_type) continue;
    if (!solver_.solve(armor)) continue;
    if (config_.outpost_distance_optimizer_enabled && armor.name == ArmorName::outpost) {
      solver_.optimize_outpost_distance(armor, detections.lightbars);
    }
    if (config_.yaw_refinement_enabled) {
      const auto predicted_armors = target.armor_xyza_list();
      if (!predicted_armors.empty()) {
        const auto matched = std::min_element(
          predicted_armors.begin(), predicted_armors.end(), [&](const auto & lhs, const auto & rhs) {
            return std::abs(tools::limit_rad(armor.ypr_in_world[0] - lhs[3])) <
              std::abs(tools::limit_rad(armor.ypr_in_world[0] - rhs[3]));
          });
        solver_.refine_yaw_with_prediction(
          armor, (*matched)[3], static_cast<int>(predicted_armors.size()));
      }
    }
    debug_info_.matched_armor_count++;
    if (target.name == ArmorName::outpost) {
      outpost_armors.push_back(armor);
    } else {
      found = true;
      target.update(armor);
      occupied_ids.push_back(target.last_id);
    }
  }

  if (target.name == ArmorName::outpost && !outpost_armors.empty()) {
    found = target.update_outpost(outpost_armors);
    if (found) occupied_ids.push_back(target.last_id);
  }

  if (found && target.name != ArmorName::outpost && lightbar_assist_enabled()) {
    const auto predicted_armors = target.armor_xyza_list();
    std::vector<int> visible_ids(predicted_armors.size());
    std::iota(visible_ids.begin(), visible_ids.end(), 0);
    std::sort(visible_ids.begin(), visible_ids.end(), [&](int lhs, int rhs) {
      return solver_.armor_visibility_score(
               predicted_armors[lhs].head<3>(), predicted_armors[lhs][3], target.name) >
        solver_.armor_visibility_score(
          predicted_armors[rhs].head<3>(), predicted_armors[rhs][3], target.name);
    });
    if (visible_ids.size() > 3) visible_ids.resize(3);

    ObservationMatchStats match_stats;
    const auto matches = matcher_.match_lightbars(
      detections, target.name, target.armor_type, predicted_armors, visible_ids, occupied_ids,
      config_.reprojection, enemy_color_, match_stats);
    std::vector<ReprojectionMeasurement> measurements;
    measurements.reserve(matches.size());
    for (const auto & match : matches) {
      measurements.push_back({match.prediction_id, match.right, *match.observation});
    }
    debug_info_.rejected_light_count += match_stats.rejected_light_count;
    debug_info_.last_match_cost = match_stats.last_match_cost;
    debug_info_.matched_light_count = measurements.size();
    debug_info_.uvl_observation_count = measurements.size();
    if (!measurements.empty() &&
        target.update_lightbar_assist(measurements, solver_, config_.reprojection)) {
      debug_info_.last_update_source = "pnp_lightbar_assist";
      debug_info_.last_nis = target.last_nis();
      debug_info_.lightbar_assist_update_count++;
    } else if (!measurements.empty()) {
      debug_info_.lightbar_assist_failed_count++;
    }
  }

  if (found) {
    if (debug_info_.last_update_source != "pnp_lightbar_assist") {
      debug_info_.last_update_source = "pnp";
    }
    debug_info_.last_nis = target.last_nis();
  } else {
    debug_info_.predict_only_count++;
  }
  return found;
}

}  // namespace auto_aim
