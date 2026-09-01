#ifndef SRC__AIM_FACTORY_HPP
#define SRC__AIM_FACTORY_HPP

#include <chrono>
#include <functional>
#include <optional>
#include <string>

#include "tasks/auto_aim/planner/planner.hpp"  // auto_aim::Plan
#include "tasks/auto_aim/target.hpp"

/// 创建 MPC 轨迹规划器。
///
/// 返回的 std::function 统一签名：
///   Plan(std::optional<Target>, double bullet_speed, steady_clock::time_point)
std::function<auto_aim::Plan(std::optional<auto_aim::Target>, double, std::chrono::steady_clock::time_point)>
create_aim_fn(const std::string & config_path);

#endif  // SRC__AIM_FACTORY_HPP
