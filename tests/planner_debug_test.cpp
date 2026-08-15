#include <cassert>
#include <cmath>
#include <optional>

#include "tasks/auto_aim/planner/planner.hpp"

int main()
{
  auto_aim::Planner planner("configs/demo.yaml");
  auto_aim::Target target(3.0, 5.0, 0.2, 0.1);

  const auto plan = planner.plan(target, 22.0);
  assert(plan.control);
  assert(plan.debug_valid);
  assert(std::isfinite(plan.fly_time));
  assert(plan.fly_time > 0.0);
  assert(plan.debug_xyza.array().isFinite().all());

  const auto idle_plan = planner.plan(std::optional<auto_aim::Target>{}, 22.0);
  assert(!idle_plan.control);
  assert(!idle_plan.fire);
  assert(!idle_plan.debug_valid);
  return 0;
}
