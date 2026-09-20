#ifndef AUTO_BUFF_V2__BUFF_PLANNER_HPP
#define AUTO_BUFF_V2__BUFF_PLANNER_HPP

#include <cstdint>
#include <optional>

#include "io/gimbal/gimbal.hpp"
#include "buff_config.hpp"
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
  using Config = BuffConfig::Planner;

  explicit BuffPlanner(Config config);
  std::optional<BuffTrackingRequest> prepare(
    std::uint64_t target_generation, const std::optional<RuneEstimate> & target,
    double bullet_speed,
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
