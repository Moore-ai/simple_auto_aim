#ifndef TOOLS__FRAME_FACTS_HPP
#define TOOLS__FRAME_FACTS_HPP

#include <cstdint>
#include <optional>
#include <utility>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/frame_snapshot.hpp"

namespace tools
{

// Inputs observed while processing one camera frame.
struct FrameFacts
{
  FrameSnapshot::Timestamp timestamp;
  cv::Mat image;
  Eigen::Quaterniond q_gimbal2world;
  io::GimbalStatePacket received;
  std::optional<io::InfantryEnemyColor> target_color_override;

  std::optional<io::InfantryEnemyColor> enemy_color() const
  {
    return target_color_override ? target_color_override :
                                   io::infantry_enemy_color(received.state.mode);
  }

  FrameSnapshot snapshot(
    const io::GimbalCommandPacket & sent, auto_aim::DetectionResult detections,
    const auto_aim::TrackerDebugData & tracker, BuffDebugData buff_debug = {}) const
  {
    return FrameSnapshot::capture(
      timestamp, image, q_gimbal2world, received.state, sent.command, std::move(detections),
      tracker, sent.packet, received.packet, std::move(buff_debug));
  }
};

// Owns the capture order and target-generation contract shared by every mode.
class FrameCapture
{
public:
  FrameCapture(io::Camera & camera, io::Gimbal & gimbal,
               std::optional<io::InfantryEnemyColor> target_color_override = std::nullopt)
  : camera_(camera), gimbal_(gimbal), target_color_override_(target_color_override)
  {}

  bool next(FrameFacts & facts)
  {
    if (!camera_.read(facts.image, facts.timestamp)) return false;
    facts.received = gimbal_.state_with_packet();
    facts.target_color_override = target_color_override_;
    facts.q_gimbal2world = gimbal_.q_gimbal2world(facts.timestamp);
    return true;
  }

  FrameSnapshot complete(
    const FrameFacts & facts, bool has_target, auto_aim::DetectionResult detections,
    const auto_aim::TrackerDebugData & tracker, BuffDebugData buff_debug = {})
  {
    if (has_target && !had_target_) ++target_generation_;
    had_target_ = has_target;
    auto snapshot = facts.snapshot(gimbal_.command_with_packet(), std::move(detections), tracker,
                                   std::move(buff_debug));
    snapshot.target_generation = target_generation_;
    return snapshot;
  }

private:
  io::Camera & camera_;
  io::Gimbal & gimbal_;
  std::optional<io::InfantryEnemyColor> target_color_override_;
  std::uint64_t target_generation_ = 0;
  bool had_target_ = false;
};

}  // namespace tools

#endif  // TOOLS__FRAME_FACTS_HPP
