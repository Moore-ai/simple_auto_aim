#include "aim_factory.hpp"

#include <yaml-cpp/yaml.h>

#include <memory>

#include "tasks/auto_aim/aimer.hpp"
#include "tools/logger.hpp"

std::function<auto_aim::Plan(std::optional<auto_aim::Target>, double, std::chrono::steady_clock::time_point)>
create_aim_fn(const std::string & config_path)
{
  auto config = YAML::LoadFile(config_path);
  auto aim_method = config["aim_method"] ? config["aim_method"].as<std::string>() : "mpc";
  tools::logger()->info("[AimFactory] Using {} aimer", aim_method);

  if (aim_method == "aimer") {
    auto aimer = std::make_shared<auto_aim::Aimer>(config_path);
    return [aimer](
             std::optional<auto_aim::Target> target, double bullet_speed,
             std::chrono::steady_clock::time_point timestamp) -> auto_aim::Plan {
      if (!target) {
        return {false, false, 0, 0, 0, 0, 0, 0, 0, 0};
      }
      std::list<auto_aim::Target> targets;
      targets.push_back(*target);
      auto cmd = aimer->aim(targets, timestamp, bullet_speed);
      return {cmd.control,           cmd.shoot,
              static_cast<float>(cmd.yaw),    static_cast<float>(cmd.pitch),
              static_cast<float>(cmd.yaw),    0,
              0,                              static_cast<float>(cmd.pitch),
              0,                              0};
    };
  }

  // Default: MPC planner
  auto planner = std::make_shared<auto_aim::Planner>(config_path);
  return [planner](
           std::optional<auto_aim::Target> target, double bullet_speed,
           std::chrono::steady_clock::time_point) -> auto_aim::Plan {
    return planner->plan(target, bullet_speed);
  };
}
