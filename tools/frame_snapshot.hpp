#ifndef TOOLS__FRAME_SNAPSHOT_HPP
#define TOOLS__FRAME_SNAPSHOT_HPP

#include <chrono>
#include <optional>
#include <utility>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/tracker.hpp"

namespace tools
{

// Data captured while processing one camera frame.
// Consumers receive one snapshot so rendering and serialization cannot mix frame values.
struct FrameSnapshot
{
  using Timestamp = std::chrono::steady_clock::time_point;

  Timestamp timestamp{};
  cv::Mat image;
  Eigen::Quaterniond gimbal_orientation{Eigen::Quaterniond::Identity()};
  io::GimbalState gimbal_state{};
  io::GimbalCommand gimbal_command{};
  auto_aim::DetectionResult detections;
  auto_aim::TrackerDebugData tracker;
  std::optional<std::vector<cv::Point2f>> anti_spin_hit_armor;

  static FrameSnapshot capture(
    Timestamp timestamp, const cv::Mat & image, const Eigen::Quaterniond & gimbal_orientation,
    const io::GimbalState & gimbal_state, const io::GimbalCommand & gimbal_command,
    auto_aim::DetectionResult detections, const auto_aim::TrackerDebugData & tracker)
  {
    return {
      timestamp,
      image.clone(),
      gimbal_orientation,
      gimbal_state,
      gimbal_command,
      std::move(detections),
      tracker};
  }
};

}  // namespace tools

#endif  // TOOLS__FRAME_SNAPSHOT_HPP
