#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tinympc/tiny_api.hpp"
#include "tools/adaptive_delay_controller.hpp"

namespace auto_aim
{
constexpr double DT = 0.01;
constexpr int HALF_HORIZON = 50;
constexpr int HORIZON = HALF_HORIZON * 2;

using Trajectory = Eigen::Matrix<double, 4, HORIZON>;  // yaw, yaw_vel, pitch, pitch_vel

struct Plan
{
  bool control;
  bool fire;
  float target_yaw;
  float target_pitch;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
  Eigen::Vector4d debug_xyza{Eigen::Vector4d::Zero()};
  double fly_time{0.0};
  bool debug_valid{false};
};

class Planner
{
public:
  Eigen::Vector4d debug_xyza;
  Planner(const std::string & config_path);

  Plan plan(Target target, double bullet_speed);
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  double yaw_offset_;
  double pitch_offset_;
  double fire_thresh_;
  double low_speed_delay_time_, high_speed_delay_time_, decision_speed_;
  double extra_delay_{0.015};
  double speed_hysteresis_{0.0};
  bool decision_speed_enable_{true};
  bool high_speed_state_{false};

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

  TinySolver * yaw_solver_;
  TinySolver * pitch_solver_;

  void setup_yaw_solver(const std::string & config_path);
  void setup_pitch_solver(const std::string & config_path);

  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP
