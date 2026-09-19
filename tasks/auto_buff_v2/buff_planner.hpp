#ifndef AUTO_BUFF_V2__BUFF_PLANNER_HPP
#define AUTO_BUFF_V2__BUFF_PLANNER_HPP

#include <cstdint>
#include <optional>
#include <string>

#include "io/gimbal/gimbal.hpp"
#include "rune_model.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/ballistic_solver.hpp"

namespace auto_buff_v2
{
struct BuffTrackingRequest
{
  auto_aim::Trajectory trajectory;
  double yaw0 = 0;
  double distance = -1;
  double fly_time = 0;
  Eigen::Vector3d rune_center = Eigen::Vector3d::Zero();
  Eigen::Vector3d aimpoint = Eigen::Vector3d::Zero();
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
  std::optional<BuffTrackingRequest> prepare(
    std::uint64_t target_generation, const std::optional<RuneState> & target, double bullet_speed,
    Timestamp now);
  bool fire_advice(const BuffTrackingRequest & request, const auto_aim::Plan & plan,
                   const io::GimbalState & gimbal, Timestamp now) const;

private:
  Config config_;
  std::unique_ptr<tools::BallisticSolver> ballistic_solver_;
  std::optional<Timestamp> attack_start_;
  std::optional<std::uint64_t> attack_generation_;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__BUFF_PLANNER_HPP
