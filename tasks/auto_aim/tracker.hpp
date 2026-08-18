#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "armor.hpp"
#include "observation_path.hpp"
#include "solver.hpp"
#include "target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{

using TrackerDebugInfo = ObservationPathDebugInfo;

class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;
  bool reprojection_enabled() const { return observation_path_.uses_reprojection(); }
  const TrackerDebugInfo & debug_info() const { return observation_path_.debug_info(); }

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::list<Target> track(
    DetectionResult & detections, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue,
    DetectionResult & detections, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;
  ObservationPath observation_path_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  // 优先级配置
  bool use_priority_;
  std::unordered_map<ArmorName, ArmorPriority> priority_map_;

  // 滤波器配置（一次性读取，传递给 Target）
  FilterConfig filter_config_;
  FilterMethod filter_method_;
  // EKF 初始协方差 P0（按兵种）
  Eigen::VectorXd P0_default_, P0_balance_, P0_outpost_, P0_base_;
  // EKF 初始半径（按兵种）
  double radius_default_, radius_outpost_, radius_base_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(DetectionResult & detections, std::chrono::steady_clock::time_point t);

  void sort_armors(std::list<Armor> & armors) const;

  void assign_priorities(std::list<Armor> & armors) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP
