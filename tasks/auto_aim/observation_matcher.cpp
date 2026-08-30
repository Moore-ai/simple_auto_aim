#include "observation_matcher.hpp"

#include <set>

namespace auto_aim
{
namespace
{
constexpr double invalid_cost = 1e12;

struct Assignment
{
  std::size_t observation = 0;
  std::size_t prediction = 0;
  double cost = 0.0;
};

std::vector<Assignment> greedy_assign(const std::vector<std::vector<double>> & costs)
{
  if (costs.empty() || costs.front().empty()) return {};

  std::vector<bool> used_observations(costs.size(), false);
  std::vector<bool> used_predictions(costs.front().size(), false);
  std::vector<Assignment> result;
  while (true) {
    Assignment best{0, 0, invalid_cost};
    bool found = false;
    for (std::size_t observation = 0; observation < costs.size(); ++observation) {
      if (used_observations[observation]) continue;
      for (std::size_t prediction = 0; prediction < costs[observation].size(); ++prediction) {
        if (!used_predictions[prediction] && costs[observation][prediction] < best.cost) {
          best = {observation, prediction, costs[observation][prediction]};
          found = true;
        }
      }
    }
    if (!found) break;
    used_observations[best.observation] = true;
    used_predictions[best.prediction] = true;
    result.push_back(best);
  }
  return result;
}
}  // namespace

std::vector<ArmorObservationMatch> ObservationMatcher::match_armors(
  DetectionResult & detections, ArmorName target_name, ArmorType armor_type,
  const std::vector<Eigen::Vector4d> & predicted_armors, const std::vector<int> & visible_ids,
  const ReprojectionObservationConfig & config, bool all_armor_ids_seen,
  ObservationMatchStats & stats) const
{
  std::vector<Armor *> candidates;
  for (auto & armor : detections.armors) {
    if (armor.name == target_name && armor.type == armor_type && armor.points.size() == 4) {
      candidates.push_back(&armor);
    }
  }

  std::vector<std::vector<double>> costs(
    candidates.size(), std::vector<double>(visible_ids.size(), invalid_cost));
  for (std::size_t observation = 0; observation < candidates.size(); ++observation) {
    bool in_gate = false;
    for (std::size_t prediction = 0; prediction < visible_ids.size(); ++prediction) {
      const auto id = visible_ids[prediction];
      const auto cost = geometry_.armor_match_cost(
        candidates[observation]->points, predicted_armors[id].head<3>(), predicted_armors[id][3],
        armor_type, target_name, config, all_armor_ids_seen);
      if (cost) {
        costs[observation][prediction] = *cost;
        in_gate = true;
      }
    }
    if (!in_gate) stats.rejected_armor_count++;
  }

  std::vector<ArmorObservationMatch> result;
  for (const auto & assignment : greedy_assign(costs)) {
    stats.last_match_cost = assignment.cost;
    result.push_back({visible_ids[assignment.prediction], candidates[assignment.observation]});
  }
  return result;
}

std::vector<LightbarObservationMatch> ObservationMatcher::match_lightbars(
  const DetectionResult & detections, ArmorName target_name, ArmorType armor_type,
  const std::vector<Eigen::Vector4d> & predicted_armors, const std::vector<int> & visible_ids,
  const std::vector<int> & occupied_ids, const ReprojectionObservationConfig & config,
  Color enemy_color, ObservationMatchStats & stats) const
{
  std::vector<LightbarObservationMatch> result;
  if (target_name == ArmorName::base || predicted_armors.empty() || visible_ids.empty()) {
    return result;
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
      predicted_armors[id].head<3>(), predicted_armors[id][3], right, armor_type, target_name);
    if (projected.valid) visible_lights.push_back({id, right, projected});
  };
  add_light((closest_id + armor_count - 1) % armor_count, true);
  add_light((closest_id + 1) % armor_count, false);
  add_light(closest_id, false);
  add_light(closest_id, true);

  std::vector<const Lightbar *> candidates;
  for (const auto & lightbar : detections.lightbars) {
    if (lightbar.color == enemy_color && lightbar.top != lightbar.bottom) {
      candidates.push_back(&lightbar);
    }
  }

  std::vector<std::vector<double>> costs(
    candidates.size(), std::vector<double>(visible_lights.size(), invalid_cost));
  for (std::size_t observation = 0; observation < candidates.size(); ++observation) {
    bool in_gate = false;
    for (std::size_t prediction = 0; prediction < visible_lights.size(); ++prediction) {
      const auto cost = geometry_.lightbar_match_cost(
        *candidates[observation], visible_lights[prediction].observation, config);
      if (cost) {
        costs[observation][prediction] = *cost;
        in_gate = true;
      }
    }
    if (!in_gate) stats.rejected_light_count++;
  }

  for (const auto & assignment : greedy_assign(costs)) {
    stats.last_match_cost = assignment.cost;
    const auto & prediction = visible_lights[assignment.prediction];
    result.push_back({prediction.id, prediction.right, candidates[assignment.observation]});
  }
  return result;
}

}  // namespace auto_aim
