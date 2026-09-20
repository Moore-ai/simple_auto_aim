#ifndef AUTO_BUFF_V2__FRAME_RUNTIME_HPP
#define AUTO_BUFF_V2__FRAME_RUNTIME_HPP

#include "io/gimbal/gimbal.hpp"
#include "buff_frame_processor.hpp"
#include "rune_model.hpp"
#include "tools/frame_facts.hpp"
#include "tools/processed_frame.hpp"

namespace auto_buff_v2
{
class FrameRuntime
{
public:
  FrameRuntime(io::Camera & camera, io::Gimbal & gimbal, RuneModel & model,
               BuffConfig::Detector detector_config)
  : frames_(camera, gimbal), processor_(model, std::move(detector_config))
  {}

  bool next(tools::ProcessedFrame & result)
  {
    tools::FrameFacts facts;
    if (!frames_.next(facts)) return false;
    result = processor_.process(facts);
    result.snapshot = frames_.complete(
      facts, result.buff_target.has_value(), {}, {}, std::move(result.snapshot.buff_debug));
    return true;
  }

private:
  tools::FrameCapture frames_;
  BuffFrameProcessor processor_;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__FRAME_RUNTIME_HPP
