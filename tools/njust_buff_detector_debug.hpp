#ifndef TOOLS__NJUST_BUFF_DETECTOR_DEBUG_HPP
#define TOOLS__NJUST_BUFF_DETECTOR_DEBUG_HPP

#include <memory>
#include <string>

#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tasks/auto_buff_v2/detectors/njust_rune_detector.hpp"
#include "tools/frame_facts.hpp"
#include <foxglove/channel.hpp>
#include <nlohmann/json.hpp>

namespace tools
{
enum class NjustBuffDetectorDebugTarget { icon, bullseye };

struct NjustBuffDetectorDebugFrame
{
  FrameSnapshot snapshot;
  auto_buff_v2::NjustDetectorMeasurements measurements;
  NjustBuffDetectorDebugTarget target;
};

// Njust-only R icon and bullseye diagnostics; requires buff_v2.detector.type: njust.
class NjustBuffDetectorDebug
{
public:
  NjustBuffDetectorDebug(
    auto_buff_v2::BuffConfig::Detector config, NjustBuffDetectorDebugTarget target);
  NjustBuffDetectorDebugFrame process(const FrameFacts & facts);

private:
  auto_buff_v2::NjustRuneDetector detector_;
  NjustBuffDetectorDebugTarget target_;
};

namespace detail
{
nlohmann::json njust_buff_detector_values(
  const auto_buff_v2::NjustDetectorMeasurements & measurements);
foxglove::FoxgloveResult<foxglove::RawChannel> create_njust_buff_detector_channel();
cv::Mat njust_buff_detector_debug_image(
  const NjustBuffDetectorDebugFrame & frame, bool draw_roi = true);
}

class NjustBuffDetectorDebugVisualizer
{
public:
  explicit NjustBuffDetectorDebugVisualizer(const std::string & config_path);
  ~NjustBuffDetectorDebugVisualizer();
  void publish(NjustBuffDetectorDebugFrame frame);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

int run_njust_buff_detector_debug(int argc, char * argv[], NjustBuffDetectorDebugTarget target);
}  // namespace tools

#endif  // TOOLS__NJUST_BUFF_DETECTOR_DEBUG_HPP
