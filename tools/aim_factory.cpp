#include "aim_factory.hpp"

#include <memory>

#include "tools/logger.hpp"

std::function<auto_aim::Plan(std::optional<auto_aim::Target>, double, std::chrono::steady_clock::time_point)>
create_aim_fn(const std::string & config_path)
{
  tools::logger()->info("[AimFactory] Using MPC planner");
  auto planner = std::make_shared<auto_aim::Planner>(config_path);
  return [planner](
           std::optional<auto_aim::Target> target, double bullet_speed,
           std::chrono::steady_clock::time_point) -> auto_aim::Plan {
    return planner->plan(target, bullet_speed);
  };
}
