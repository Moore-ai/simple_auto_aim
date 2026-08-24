#include "observation_path.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

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

ObservationPath::ObservationPath(Solver & solver) : solver_{solver}, geometry_{solver} {}

void ObservationPath::configure(const ObservationPathConfig & config, Color enemy_color)
{
  config_ = config;
  enemy_color_ = enemy_color;
  debug_info_.observation_mode = config_.mode == ObservationMode::REPROJECTION ?
    "reprojection" : "pnp";
  debug_info_.lightbar_assist_enabled = config_.lightbar_assist_enabled;
  debug_info_.yaw_refinement_enabled = config_.yaw_refinement_enabled;
}

bool ObservationPath::uses_reprojection() const
{
  return config_.mode == ObservationMode::REPROJECTION;
}

bool ObservationPath::lightbar_assist_enabled() const
{
  return config_.lightbar_assist_enabled;
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

std::vector<ReprojectionMeasurement> ObservationPath::match_independent_lightbars(
  Target & target, DetectionResult & detections,
  const std::vector<Eigen::Vector4d> & predicted_armors,
  const std::vector<int> & visible_ids, const std::vector<int> & occupied_ids)
{
  std::vector<ReprojectionMeasurement> measurements;
  if (target.name == ArmorName::base || predicted_armors.empty() || visible_ids.empty()) {
    return measurements;
  }

  const auto armor_count = static_cast<int>(predicted_armors.size());
  std::set<std::pair<int, bool>> used_sides;
  for (const auto id : occupied_ids) {
    if (id >= 0 && id < armor_count) {
      used_sides.emplace(id, false);
      used_sides.emplace(id, true);
    }
  }

  struct PredictedLight
  {
    int id;
    bool right;
    PredictedLightbarObservation observation;
  };
  std::vector<PredictedLight> visible_lights;
  const auto closest_id = visible_ids.front();
  auto add_light = [&](int id, bool right) {
    if (id < 0 || id >= armor_count || used_sides.count({id, right})) return;
    const auto projected = geometry_.project_lightbar(
      predicted_armors[id].head<3>(), predicted_armors[id][3], right,
      target.armor_type, target.name);
    if (!projected.valid) return;
    visible_lights.push_back({id, right, projected});
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
  constexpr double invalid_cost = 1e12;
  std::vector<std::vector<double>> light_cost(
    light_candidates.size(), std::vector<double>(visible_lights.size(), invalid_cost));
  for (std::size_t obs = 0; obs < light_candidates.size(); ++obs) {
    bool in_gate = false;
    for (std::size_t candidate = 0; candidate < visible_lights.size(); ++candidate) {
      const auto & predicted = visible_lights[candidate];
      const auto cost = geometry_.lightbar_match_cost(
        *light_candidates[obs], predicted.observation, config_.reprojection);
      if (!cost) continue;
      light_cost[obs][candidate] = *cost;
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
    measurements.push_back({prediction.id, prediction.right, *light_candidates[best_observation]});
    debug_info_.last_match_cost = best_cost;
  }
  return measurements;
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

  std::vector<Armor *> candidates;
  for (auto & armor : detections.armors) {
    if (armor.name == target.name && armor.type == target.armor_type && armor.points.size() == 4) {
      candidates.push_back(&armor);
    }
  }

  constexpr double invalid_cost = 1e12;
  std::vector<std::vector<double>> armor_cost(
    candidates.size(), std::vector<double>(visible_ids.size(), invalid_cost));
  for (std::size_t obs = 0; obs < candidates.size(); ++obs) {
    const auto & measured = candidates[obs]->points;
    bool in_gate = false;
    for (std::size_t candidate = 0; candidate < visible_ids.size(); ++candidate) {
      const auto id = visible_ids[candidate];
      const auto cost = geometry_.armor_match_cost(
        measured,
        predicted_armors[id].head<3>(), predicted_armors[id][3],
        target.armor_type, target.name, config_.reprojection, target.all_armor_ids_seen());
      if (cost) {
        armor_cost[obs][candidate] = *cost;
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
  for (const auto & [id, armor] : matched_armors) {
    armor_measurements.push_back({id, *armor});

    Lightbar left, right;
    left.top = armor->points[0];
    left.bottom = armor->points[3];
    right.top = armor->points[1];
    right.bottom = armor->points[2];
    measurements.push_back({id, false, left});
    measurements.push_back({id, true, right});
  }

  if (!matched_armors.empty() && target.name != ArmorName::base) {
    std::vector<int> occupied_ids;
    occupied_ids.reserve(matched_armors.size());
    for (const auto & [id, armor] : matched_armors) {
      (void)armor;
      occupied_ids.push_back(id);
    }
    const auto independent_measurements = match_independent_lightbars(
      target, detections, predicted_armors, visible_ids, occupied_ids);
    measurements.insert(
      measurements.end(), independent_measurements.begin(), independent_measurements.end());
  }

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

  Armor * fallback = matched_armors.empty() ?
    (candidates.empty() ? nullptr : candidates.front()) : matched_armors.front().second;
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
  for (auto & armor : detections.armors) {
    if (armor.name != target.name || armor.type != target.armor_type) continue;
    if (!solver_.solve(armor)) continue;
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
    found = true;
    debug_info_.matched_armor_count++;
    target.update(armor);
    occupied_ids.push_back(target.last_id);
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

    const auto measurements = match_independent_lightbars(
      target, detections, predicted_armors, visible_ids, occupied_ids);
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
