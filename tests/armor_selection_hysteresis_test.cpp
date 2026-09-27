#include <cassert>
#include <vector>

#include "tasks/auto_aim/planner/armor_selection_hysteresis.hpp"

int main()
{
  auto_aim::ArmorSelectionHysteresis selector({true, 3});

  assert(selector.select({1.00, 1.01}) == 0);
  assert(selector.select({1.04, 1.00}) == 0);
  assert(selector.select({1.04, 1.00}) == 0);
  assert(selector.select({1.04, 1.00}) == 1);

  // Even a small advantage switches after three consecutive frames.
  assert(selector.select({0.98, 0.99}) == 1);
  assert(selector.select({0.98, 0.99}) == 1);
  assert(selector.select({0.98, 0.99}) == 0);

  // A frame led by the locked armor breaks the candidate's streak.
  assert(selector.select({0.99, 1.00}) == 0);
  assert(selector.select({1.01, 1.00}) == 0);
  assert(selector.select({0.99, 1.00}) == 0);
  assert(selector.select({1.01, 1.00}) == 0);
  assert(selector.select({1.01, 1.00}) == 0);
  assert(selector.select({1.01, 1.00}) == 1);

  // Losing the locked candidate must switch immediately.
  assert(selector.select({1.00}) == 0);

  auto_aim::ArmorSelectionHysteresis disabled({false, 3});
  assert(disabled.select({1.00, 1.01}) == 0);
  assert(disabled.select({1.01, 1.00}) == 1);
  return 0;
}
