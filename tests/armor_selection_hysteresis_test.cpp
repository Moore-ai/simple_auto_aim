#include <cassert>
#include <vector>

#include "tasks/auto_aim/planner/armor_selection_hysteresis.hpp"

int main()
{
  auto_aim::ArmorSelectionHysteresis selector({true, 0.02, 3});

  assert(selector.select({1.00, 1.01}) == 0);
  assert(selector.select({1.04, 1.00}) == 0);
  assert(selector.select({1.04, 1.00}) == 0);
  assert(selector.select({1.04, 1.00}) == 1);

  // A small advantage is noise inside the hysteresis band, not a switch request.
  assert(selector.select({0.98, 0.99}) == 1);

  // Losing the locked candidate must switch immediately.
  assert(selector.select({1.00}) == 0);

  auto_aim::ArmorSelectionHysteresis disabled({false, 0.02, 3});
  assert(disabled.select({1.00, 1.01}) == 0);
  assert(disabled.select({1.01, 1.00}) == 1);
  return 0;
}
