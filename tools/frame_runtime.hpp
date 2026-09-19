#ifndef TOOLS__FRAME_RUNTIME_HPP
#define TOOLS__FRAME_RUNTIME_HPP

#include <utility>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tools/detect_factory.hpp"
#include "tools/frame_facts.hpp"
#include "tools/processed_frame.hpp"

namespace tools
{

// Owns the ordering and timestamp contract for one vehicle camera frame.
class FrameRuntime
{
public:
  FrameRuntime(
    io::Camera & camera, io::Gimbal & gimbal, auto_aim::Solver & solver,
    auto_aim::Tracker & tracker, DetectionBackend & detector)
  : frames_{camera, gimbal}, solver_{solver}, tracker_{tracker}, detector_{detector}
  {}

  bool next(ProcessedFrame & result)
  {
    FrameFacts facts;
    if (!frames_.next(facts)) return false;

    if (const auto color = io::infantry_enemy_color(facts.received.state.mode)) {
      tracker_.set_enemy_color(
        *color == io::InfantryEnemyColor::red ? auto_aim::Color::red : auto_aim::Color::blue);
    }

    solver_.set_R_gimbal2world(facts.gimbal_orientation);
    auto detections = detector_.detect(facts.image, -1);
    auto tracking_detections = detections;
    auto targets = tracker_.track(tracking_detections, facts.timestamp);

    result.snapshot = frames_.complete(
      facts, !targets.empty(), std::move(detections), tracker_.debug_data());
    result.targets = std::move(targets);
    result.buff_target.reset();
    return true;
  }

private:
  FrameCapture frames_;
  auto_aim::Solver & solver_;
  auto_aim::Tracker & tracker_;
  DetectionBackend & detector_;
};

}  // namespace tools

#endif  // TOOLS__FRAME_RUNTIME_HPP
