#ifndef AUTO_BUFF_V2__BUFF_PLANNER_HPP
#define AUTO_BUFF_V2__BUFF_PLANNER_HPP

#include <optional>
#include <string>

#include "io/gimbal/gimbal.hpp"
#include "rune_model.hpp"
#include "tools/ballistic_solver.hpp"

namespace auto_buff_v2
{
struct BuffPlan
{
  bool control = false;
  bool fire = false;
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  float distance = -1;
};

class BuffPlanner
{
public:
  struct Config
  {
    double shoot_delay = 0.04;
    double rune_idle_duration = 0.4;
    double rune_shoot_duration = 0.2;
    double yaw_tolerance = 0.07;
    double pitch_tolerance = 0.04;
    double yaw_offset = 0;
    double pitch_offset = 0;
    double bullet_speed_min = 10;
    double bullet_speed_max = 25;
    double bullet_speed_default = 23.4;
    std::string ballistic_model = "njust";
    tools::BallisticSolverConfig ballistic_config;
  };

  explicit BuffPlanner(Config config);
  explicit BuffPlanner(const std::string & config_path);
  BuffPlan plan(std::optional<RuneState> target, double bullet_speed,
                const io::GimbalState & gimbal, Timestamp now);

private:
  Config config_;
  std::unique_ptr<tools::BallisticSolver> ballistic_solver_;
  std::optional<Timestamp> attack_start_;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__BUFF_PLANNER_HPP
