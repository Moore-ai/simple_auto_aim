#ifndef AUTO_AIM__OBSERVATION_MATCHER_HPP
#define AUTO_AIM__OBSERVATION_MATCHER_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <vector>

#include "armor.hpp"
#include "observation_geometry.hpp"

namespace auto_aim
{

struct ArmorObservationMatch
{
  int prediction_id = -1;
  Armor * observation = nullptr;
};

struct LightbarObservationMatch
{
  int prediction_id = -1;
  bool right = false;
  const Lightbar * observation = nullptr;
};

struct ObservationMatchStats
{
  std::size_t rejected_armor_count = 0;
  std::size_t rejected_light_count = 0;
  double last_match_cost = 0.0;
};

// Data-association seam for image observations. Projection and filtering stay in
// ObservationGeometry; this module owns candidate selection and one-to-one assignment.
class ObservationMatcher
{
public:
  explicit ObservationMatcher(const ObservationGeometry & geometry) : geometry_{geometry} {}

  std::vector<ArmorObservationMatch> match_armors(
    DetectionResult & detections, ArmorName target_name, ArmorType armor_type,
    const std::vector<Eigen::Vector4d> & predicted_armors, const std::vector<int> & visible_ids,
    const ReprojectionObservationConfig & config, bool all_armor_ids_seen,
    ObservationMatchStats & stats) const;

  std::vector<LightbarObservationMatch> match_lightbars(
    const DetectionResult & detections, ArmorName target_name, ArmorType armor_type,
    const std::vector<Eigen::Vector4d> & predicted_armors, const std::vector<int> & visible_ids,
    const std::vector<int> & occupied_ids, const ReprojectionObservationConfig & config,
    Color enemy_color, ObservationMatchStats & stats) const;

private:
  const ObservationGeometry & geometry_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OBSERVATION_MATCHER_HPP
