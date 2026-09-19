#include <type_traits>
#include <utility>

#include "tasks/auto_aim/frame_runtime.hpp"

static_assert(std::is_constructible_v<
              auto_aim::FrameRuntime, io::Camera &, io::Gimbal &, auto_aim::Solver &,
              auto_aim::Tracker &, tools::DetectionBackend &>);
static_assert(std::is_same_v<
              decltype(std::declval<auto_aim::FrameRuntime>().next(
                std::declval<tools::ProcessedFrame &>())),
              bool>);

int main() { return 0; }
