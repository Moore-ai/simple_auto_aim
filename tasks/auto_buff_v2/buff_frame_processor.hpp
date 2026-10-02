#ifndef AUTO_BUFF_V2__BUFF_FRAME_PROCESSOR_HPP
#define AUTO_BUFF_V2__BUFF_FRAME_PROCESSOR_HPP

#include <memory>

#include "buff_config.hpp"
#include "detectors/rune_detector.hpp"
#include "rune_model.hpp"
#include "tools/frame_facts.hpp"
#include "tools/processed_frame.hpp"

namespace auto_buff_v2
{
// 将共享帧事实转换为打符专用的处理帧。
class BuffFrameProcessor
{
public:
  BuffFrameProcessor(RuneModel & model, BuffConfig::Detector detector_config);

  tools::ProcessedFrame process(const tools::FrameFacts & facts);

private:
  RuneModel & model_;
  std::unique_ptr<RuneDetector> detector_;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__BUFF_FRAME_PROCESSOR_HPP
