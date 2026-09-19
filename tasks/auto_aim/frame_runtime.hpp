#ifndef AUTO_AIM__FRAME_RUNTIME_HPP
#define AUTO_AIM__FRAME_RUNTIME_HPP

#include <utility>

#include "solver.hpp"
#include "tracker.hpp"
#include "tools/detect_factory.hpp"
#include "tools/frame_facts.hpp"
#include "tools/processed_frame.hpp"

namespace auto_aim
{
// Owns the ordering and timestamp contract for one auto-aim camera frame.
class FrameRuntime
{
public:
  FrameRuntime(
    io::Camera & camera, io::Gimbal & gimbal, Solver & solver, Tracker & tracker,
    tools::DetectionBackend & detector)
  : frames_{camera, gimbal}, solver_{solver}, tracker_{tracker}, detector_{detector}
  {}

  bool next(tools::ProcessedFrame & result)
  {
    tools::FrameFacts facts;
    if (!frames_.next(facts)) return false;

    if (const auto color = io::infantry_enemy_color(facts.received.state.mode)) {
      tracker_.set_enemy_color(
        *color == io::InfantryEnemyColor::red ? Color::red : Color::blue);
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
  tools::FrameCapture frames_;
  Solver & solver_;
  Tracker & tracker_;
  tools::DetectionBackend & detector_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__FRAME_RUNTIME_HPP
