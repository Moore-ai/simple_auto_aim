#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <optional>

#include "armor_selection_hysteresis.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tinympc/types.hpp"
#include "tools/adaptive_delay_controller.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control{false};
  bool fire{false};
  float target_yaw{0.0F};
  float target_pitch{0.0F};
  float yaw{0.0F};
  float yaw_vel{0.0F};
  float yaw_acc{0.0F};
  float pitch{0.0F};
  float pitch_vel{0.0F};
  float pitch_acc{0.0F};
  float distance{-1.0F};
  Eigen::Vector4d debug_xyza{Eigen::Vector4d::Zero()};
  double debug_armor_pitch{0.0};
  double fly_time{0.0};
  bool debug_valid{false};
  bool anti_spin_active{false};
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  enum class WaitArmorHeight { low, high };

  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double extra_delay_{0.015};
  double speed_hysteresis_{0.0};
  bool decision_speed_enable_{true};
  bool high_speed_state_{false};
  bool anti_spin_enable_{false};
  WaitArmorHeight anti_spin_wait_armor_{WaitArmorHeight::low};
  bool fly_time_iteration_enabled_{false};
  int fly_time_iteration_max_iteration_{3};
  double fly_time_iteration_convergence_threshold_{1e-3};
  bool armor_selection_hysteresis_enabled_{false};
  ArmorSelectionHysteresis armor_selector_{{}};

  double rho_;
  int max_iter_;
  double bullet_speed_min_, bullet_speed_max_, bullet_speed_default_;

  struct ManeuverAdaptConfig
  {
    bool enable = false;
    double nis_threshold = 10.0;
    double damping_factor = 0.7;
    double ema_alpha = 0.3;
  };

  tools::AdaptiveDelayController aimd_ctrl_;
  bool aimd_enabled_{false};
  bool last_fire_advice_{false};

  ManeuverAdaptConfig maneuver_;
  double nis_avg_{0.0};

  TinySolver * yaw_solver_{nullptr};
  TinySolver * pitch_solver_{nullptr};
  bool yaw_nonconvergence_logged_{false};
  bool pitch_nonconvergence_logged_{false};

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  int select_armor(const Target & target);
  void update_high_speed_state(const TargetState & state);
  bool anti_spin_waits_long_axis(const TargetState & state) const;
  double anti_spin_armor_height(const TargetState & state) const;
  Eigen::Vector3d anti_spin_aim_point(const Target & target) const;
  bool anti_spin_fire_ready(const Target & target) const;
  Eigen::Matrix<double, 2, 1> aim(
    const Target & target, double bullet_speed, int selected_armor, bool anti_spin);
  Trajectory get_trajectory(
    Target & target, double yaw0, double bullet_speed, int selected_armor, bool anti_spin);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
