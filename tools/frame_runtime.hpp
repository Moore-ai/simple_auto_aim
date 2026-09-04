#ifndef TOOLS__FRAME_RUNTIME_HPP
#define TOOLS__FRAME_RUNTIME_HPP

#include <cstdint>
#include <list>
#include <utility>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/detect_factory.hpp"
#include "tools/frame_snapshot.hpp"

namespace tools
{

struct ProcessedFrame
{
  FrameSnapshot snapshot;
  std::list<auto_aim::Target> targets;
};

// Owns the ordering and timestamp contract for one vehicle camera frame.
class FrameRuntime
{
public:
  FrameRuntime(
    io::Camera & camera, io::Gimbal & gimbal, auto_aim::Solver & solver,
    auto_aim::Tracker & tracker, DetectionBackend & detector)
  : camera_{camera}, gimbal_{gimbal}, solver_{solver}, tracker_{tracker}, detector_{detector}
  {}

  bool next(ProcessedFrame & result)
  {
    cv::Mat image;
    FrameSnapshot::Timestamp timestamp;
    if (!camera_.read(image, timestamp)) return false;

    const auto gimbal_state = gimbal_.state();
    if (const auto color = io::infantry_enemy_color(gimbal_state.mode)) {
      tracker_.set_enemy_color(
        *color == io::InfantryEnemyColor::red ? auto_aim::Color::red : auto_aim::Color::blue);
    }

    const auto orientation = gimbal_.q(timestamp);
    solver_.set_R_gimbal2world(orientation);
    auto detections = detector_.detect(image, -1);
    auto tracking_detections = detections;
    auto targets = tracker_.track(tracking_detections, timestamp);
    const auto has_target = !targets.empty();
    if (has_target && !had_target_) ++target_generation_;
    had_target_ = has_target;

    result.snapshot = FrameSnapshot::capture(
      timestamp, image, orientation, gimbal_state, gimbal_.command(), std::move(detections),
      tracker_.debug_data());
    result.snapshot.target_generation = target_generation_;
    result.targets = std::move(targets);
    return true;
  }

private:
  io::Camera & camera_;
  io::Gimbal & gimbal_;
  auto_aim::Solver & solver_;
  auto_aim::Tracker & tracker_;
  DetectionBackend & detector_;
  std::uint64_t target_generation_ = 0;
  bool had_target_ = false;
};

}  // namespace tools

#endif  // TOOLS__FRAME_RUNTIME_HPP
