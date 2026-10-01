#ifndef TOOLS__BUFF_DETECTOR_DEBUG_HPP
#define TOOLS__BUFF_DETECTOR_DEBUG_HPP

#include <memory>
#include <string>

#include "tasks/auto_buff_v2/rune_detector.hpp"
#include "tools/frame_facts.hpp"
#include <foxglove/channel.hpp>
#include <nlohmann/json.hpp>

namespace tools
{
enum class BuffDetectorDebugTarget { icon, bullseye };

struct BuffDetectorDebugFrame
{
  FrameSnapshot snapshot;
  auto_buff_v2::RuneDetectorMeasurements measurements;
  BuffDetectorDebugTarget target;
};

// Only the standalone debug programs select objects and collect contour measurements.
class BuffDetectorDebug
{
public:
  BuffDetectorDebug(auto_buff_v2::BuffConfig::Detector config, BuffDetectorDebugTarget target);
  BuffDetectorDebugFrame process(const FrameFacts & facts);

private:
  auto_buff_v2::RuneDetector detector_;
  BuffDetectorDebugTarget target_;
};

namespace detail
{
nlohmann::json buff_detector_values(const auto_buff_v2::RuneDetectorMeasurements & measurements);
foxglove::FoxgloveResult<foxglove::RawChannel> create_buff_detector_channel();
cv::Mat buff_detector_debug_image(const BuffDetectorDebugFrame & frame, bool draw_roi = true);
}

class BuffDetectorDebugVisualizer
{
public:
  explicit BuffDetectorDebugVisualizer(const std::string & config_path);
  ~BuffDetectorDebugVisualizer();
  void publish(BuffDetectorDebugFrame frame);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

int run_buff_detector_debug(int argc, char * argv[], BuffDetectorDebugTarget target);
}  // namespace tools

#endif  // TOOLS__BUFF_DETECTOR_DEBUG_HPP
