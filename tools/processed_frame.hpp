#ifndef TOOLS__PROCESSED_FRAME_HPP
#define TOOLS__PROCESSED_FRAME_HPP

#include <list>
#include <optional>

#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_buff_v2/rune_model.hpp"
#include "tools/frame_snapshot.hpp"

namespace tools
{

struct ProcessedFrame
{
  FrameSnapshot snapshot;
  std::list<auto_aim::Target> targets;
  std::optional<auto_buff_v2::RuneState> buff_target;
};

}  // namespace tools

#endif  // TOOLS__PROCESSED_FRAME_HPP
