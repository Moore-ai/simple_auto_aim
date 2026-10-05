#include "njust_buff_planner.hpp"

#include "../aimers/rune_aimer_factory.hpp"

namespace auto_buff_v2
{
NjustBuffPlanner::NjustBuffPlanner(BuffConfig::Planner config)
: aimer_(make_rune_aimer(config))
{
}

auto_aim::Plan NjustBuffPlanner::plan(
  std::uint64_t generation, const std::optional<RuneEstimate> & target, double speed,
  const io::GimbalState & gimbal, Timestamp now)
{
  const auto aimed = aimer_->aim(generation, target, speed, gimbal, now);
  if (!aimed) return {};
  auto result = aimed->plan;
  result.fire = aimer_->fire_advice(
    target->center, result.debug_xyza.head<3>(), result, gimbal, now);
  result.njust_aim_debug = aimer_->njust_debug();
  return result;
}
}  // namespace auto_buff_v2
