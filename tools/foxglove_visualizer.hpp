#ifndef TOOLS__FOXGLOVE_VISUALIZER_HPP
#define TOOLS__FOXGLOVE_VISUALIZER_HPP

#include <list>
#include <memory>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include <foxglove/schemas.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/tracker.hpp"

namespace tools
{
namespace detail
{
cv::Mat prepare_image_for_publish(const cv::Mat & image);
foxglove::schemas::CubePrimitive armor_cube(
  const Eigen::Vector3d & center, double yaw, double pitch, auto_aim::ArmorType armor_type);
void draw_detected_armors(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors, auto_aim::Color target_color);
}

class FoxgloveVisualizer
{
public:
  FoxgloveVisualizer();
  ~FoxgloveVisualizer();

  FoxgloveVisualizer(const FoxgloveVisualizer &) = delete;
  FoxgloveVisualizer & operator=(const FoxgloveVisualizer &) = delete;

  void publish(
    const io::GimbalState & serial_receive, const io::GimbalCommand & serial_send,
    const cv::Mat & image_raw, cv::Mat image,
    const std::list<auto_aim::Armor> & detected_armors,
    auto_aim::Color target_color,
    const auto_aim::TrackerDebugData & target_data);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace tools

#endif  // TOOLS__FOXGLOVE_VISUALIZER_HPP
