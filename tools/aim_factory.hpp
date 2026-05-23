#ifndef SRC__AIM_FACTORY_HPP
#define SRC__AIM_FACTORY_HPP

#include <chrono>
#include <functional>
#include <optional>
#include <string>

#include "tasks/auto_aim/planner/planner.hpp"  // auto_aim::Plan
#include "tasks/auto_aim/target.hpp"

/// 根据配置文件的 aim_method 字段创建对应的瞄准决策器
/// "mpc" → Planner（轨迹规划器，默认）
/// "aimer" → Aimer（传统决策器）
///
/// 返回的 std::function 统一签名：
///   Plan(std::optional<Target>, double bullet_speed, steady_clock::time_point)
std::function<auto_aim::Plan(std::optional<auto_aim::Target>, double, std::chrono::steady_clock::time_point)>
create_aim_fn(const std::string & config_path);

#endif  // SRC__AIM_FACTORY_HPP
