#include <type_traits>

#include "tasks/auto_aim/tracker.hpp"

static_assert(std::is_same_v<
              decltype(std::declval<const auto_aim::Tracker>().debug_data()),
              const auto_aim::TrackerDebugData &>);

int main() { return 0; }
