#ifndef TOOLS__FOXGLOVE_VISUALIZER_HPP
#define TOOLS__FOXGLOVE_VISUALIZER_HPP

#include <memory>

#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/tracker.hpp"

namespace tools
{
class FoxgloveVisualizer
{
public:
  FoxgloveVisualizer();
  ~FoxgloveVisualizer();

  FoxgloveVisualizer(const FoxgloveVisualizer &) = delete;
  FoxgloveVisualizer & operator=(const FoxgloveVisualizer &) = delete;

  void publish(
    const io::GimbalState & serial_receive, const io::GimbalCommand & serial_send,
    const cv::Mat & image_raw, cv::Mat image, const auto_aim::TrackerDebugData & target_data);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tools

#endif  // TOOLS__FOXGLOVE_VISUALIZER_HPP
