#ifndef TOOLS__FOXGLOVE_VISUALIZER_HPP
#define TOOLS__FOXGLOVE_VISUALIZER_HPP

#include <list>
#include <memory>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <foxglove/schemas.hpp>
#include <nlohmann/json.hpp>

#include "tools/frame_snapshot.hpp"

namespace tools
{
namespace detail
{
enum class FoxgloveTargetTopic { normal, outpost_current, outpost_v2 };

cv::Mat prepare_image_for_publish(const cv::Mat & image);
foxglove::schemas::CubePrimitive armor_cube(
  const Eigen::Vector3d & center, double yaw, double pitch, auto_aim::ArmorType armor_type);
void draw_detected_armors(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors, auto_aim::Color target_color);
FoxgloveTargetTopic target_topic(const auto_aim::TrackerDebugData & target_data);
const char * target_topic_name(FoxgloveTargetTopic topic);
foxglove::schemas::SceneUpdate target_scene_update(
  const auto_aim::TrackerDebugData & target_data);
nlohmann::json target_values(const auto_aim::TrackerDebugData & target_data);
}

class FoxgloveVisualizer
{
public:
  FoxgloveVisualizer();
  ~FoxgloveVisualizer();

  FoxgloveVisualizer(const FoxgloveVisualizer &) = delete;
  FoxgloveVisualizer & operator=(const FoxgloveVisualizer &) = delete;

  void publish(const FrameSnapshot & frame);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tools

#endif  // TOOLS__FOXGLOVE_VISUALIZER_HPP
