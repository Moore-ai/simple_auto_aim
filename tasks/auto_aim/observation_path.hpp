#ifndef AUTO_AIM__OBSERVATION_PATH_HPP
#define AUTO_AIM__OBSERVATION_PATH_HPP

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "armor.hpp"
#include "observation_geometry.hpp"
#include "target.hpp"

namespace YAML
{
class Node;
}

namespace auto_aim
{

class Solver;

enum class ObservationMode { PNP, REPROJECTION };

struct ObservationPathConfig
{
  ObservationMode mode = ObservationMode::PNP;
  bool lightbar_assist_enabled = false;
  bool yaw_refinement_enabled = false;
  bool outpost_distance_optimizer_enabled = false;
  ReprojectionObservationConfig reprojection;

  static ObservationPathConfig from_yaml(const YAML::Node & yaml, FilterMethod filter_method);
};

struct ObservationPathDebugInfo
{
  std::string observation_mode{"pnp"};
  bool lightbar_assist_enabled = false;
  bool yaw_refinement_enabled = false;
  bool outpost_distance_optimizer_enabled = false;
  std::string last_update_source{"none"};
  std::size_t matched_armor_count = 0;
  std::size_t matched_light_count = 0;
  std::size_t uvl_observation_count = 0;
  std::size_t diff_observation_count = 0;
  std::size_t rejected_armor_count = 0;
  std::size_t rejected_light_count = 0;
  std::uint64_t pnp_fallback_count = 0;
  std::uint64_t predict_only_count = 0;
  std::uint64_t lightbar_assist_update_count = 0;
  std::uint64_t lightbar_assist_failed_count = 0;
  double last_match_cost = 0.0;
  double last_nis = 0.0;
};

class ObservationPath
{
public:
  explicit ObservationPath(Solver & solver);

  void configure(const ObservationPathConfig & config, Color enemy_color);
  void set_enemy_color(Color enemy_color) { enemy_color_ = enemy_color; }

  bool uses_reprojection() const;
  bool lightbar_assist_enabled() const;
  bool outpost_distance_optimizer_enabled() const;
  const ReprojectionObservationConfig & reprojection_config() const;
  const ObservationPathDebugInfo & debug_info() const;

  void record_initialization();
  void record_predict_only();

  bool update(
    Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t);

private:
  Solver & solver_;
  ObservationGeometry geometry_;
  Color enemy_color_{Color::blue};
  ObservationPathConfig config_;
  ObservationPathDebugInfo debug_info_;

  void reset_frame_debug_info();

  bool update_pnp(
    Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t);
  bool update_reprojection(
    Target & target, DetectionResult & detections, std::chrono::steady_clock::time_point t);

  std::vector<ReprojectionMeasurement> match_independent_lightbars(
    Target & target, DetectionResult & detections,
    const std::vector<Eigen::Vector4d> & predicted_armors,
    const std::vector<int> & visible_ids, const std::vector<int> & occupied_ids);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OBSERVATION_PATH_HPP
