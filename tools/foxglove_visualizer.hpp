#ifndef TOOLS__FOXGLOVE_VISUALIZER_HPP
#define TOOLS__FOXGLOVE_VISUALIZER_HPP

#include <array>
#include <cstdint>
#include <list>
#include <memory>
#include <optional>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <foxglove/channel.hpp>
#include <foxglove/schemas.hpp>
#include <nlohmann/json.hpp>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/frame_snapshot.hpp"

namespace tools
{
namespace detail
{
enum class FoxgloveTargetTopic { normal, outpost_current, outpost_v2 };

cv::Mat prepare_image_for_publish(const cv::Mat & image);
foxglove::schemas::CubePrimitive armor_cube(
  const Eigen::Vector3d & center, double yaw, double pitch, auto_aim::ArmorType armor_type);
void draw_aim_overlay(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors,
  const auto_aim::Armor * locked_armor,
  const std::vector<cv::Point2f> * anti_spin_hit_armor);
std::optional<std::vector<cv::Point2f>> anti_spin_hit_armor(
  const auto_aim::Plan & plan, std::uint64_t plan_target_generation,
  std::uint64_t current_target_generation, auto_aim::ArmorType armor_type,
  const auto_aim::Solver & solver);
FoxgloveTargetTopic target_topic(const auto_aim::TrackerDebugData & target_data);
const char * target_topic_name(FoxgloveTargetTopic topic);
foxglove::FoxgloveResult<foxglove::RawChannel> create_target_values_channel(
  FoxgloveTargetTopic topic);
nlohmann::json angular_acceleration_values(const io::GimbalCommand & command);
foxglove::FoxgloveResult<foxglove::RawChannel> create_angular_acceleration_channel();
nlohmann::json command_packet_values(
  const std::array<uint8_t, io::kInfantryCommandPacketSize> & packet);
nlohmann::json feedback_packet_values(
  const std::array<uint8_t, io::kInfantryFeedbackPacketSize> & packet);
nlohmann::json angular_error_values(
  const auto_aim::Plan & plan, const io::GimbalState & gimbal_state);
foxglove::FoxgloveResult<foxglove::RawChannel> create_angular_error_channel();
foxglove::schemas::SceneUpdate target_scene_update(
  const auto_aim::TrackerDebugData & target_data);
nlohmann::json target_values(const auto_aim::TrackerDebugData & target_data);
}

class FoxgloveVisualizer
{
public:
  explicit FoxgloveVisualizer(auto_aim::Solver & solver);
  ~FoxgloveVisualizer();

  FoxgloveVisualizer(const FoxgloveVisualizer &) = delete;
  FoxgloveVisualizer & operator=(const FoxgloveVisualizer &) = delete;

  void update_plan(std::uint64_t target_generation, const auto_aim::Plan & plan);
  void publish(const FrameSnapshot & frame);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tools

#endif  // TOOLS__FOXGLOVE_VISUALIZER_HPP
