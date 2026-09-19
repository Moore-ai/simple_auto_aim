#include <list>
#include <optional>
#include <type_traits>
#include <utility>

#include "tools/processed_frame.hpp"

using ProcessedFrame = tools::ProcessedFrame;

static_assert(std::is_same_v<decltype(std::declval<ProcessedFrame>().targets),
                             std::list<auto_aim::Target>>);
static_assert(std::is_same_v<decltype(std::declval<ProcessedFrame>().buff_target),
                             std::optional<auto_buff_v2::RuneState>>);

int main() { return 0; }
