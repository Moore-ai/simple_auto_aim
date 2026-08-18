#ifndef TOOLS__WEB_DEBUG_HPP
#define TOOLS__WEB_DEBUG_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/planner/planner.hpp"

namespace tools
{

struct WebDebugPaths
{
  std::string shm_name{"/sp_vision_25_frame"};
  std::string data_path{"/dev/shm/sp_vision_25_data.json"};
  std::string log_path{"/dev/shm/sp_vision_25_log.json"};
};

struct WebDebugContext
{
  uint64_t frame_id{0};
  double elapsed_seconds{0.0};
  double latency_ms{0.0};
  io::GimbalMode mode{io::GimbalMode::IDLE};
  io::GimbalState gimbal{};
  std::vector<auto_aim::Armor> armors;
  std::vector<auto_aim::Lightbar> lightbars;
  std::optional<auto_aim::Target> target;
  nlohmann::json tracker_json{nullptr};
  double target_velocity_norm{0.0};
  double target_yaw_rate{0.0};
  auto_aim::Plan plan{};
  std::vector<std::vector<cv::Point2f>> projected_target_armors;
  std::vector<cv::Point2f> projected_aim_armor;
  std::optional<std::pair<cv::Point2f, cv::Point2f>> target_velocity_arrow;
  std::optional<std::pair<cv::Point2f, cv::Point2f>> target_yaw_rate_arrow;
};

class WebDebug
{
public:
  explicit WebDebug(WebDebugPaths paths = {});
  ~WebDebug();

  WebDebug(const WebDebug &) = delete;
  WebDebug & operator=(const WebDebug &) = delete;

  static nlohmann::json make_log_json(const WebDebugContext & context);
  nlohmann::json data_json() const;
  void publish(const WebDebugContext & context, cv::Mat & image);
  void publish(const WebDebugContext & context, cv::Mat && image)
  {
    publish(context, image);
  }
  void draw(cv::Mat & image, const WebDebugContext & context) const;

private:
  static constexpr size_t kMaxPoints = 100;
  static constexpr size_t kSharedMemorySize = 2 * 1024 * 1024;

  WebDebugPaths paths_;
  nlohmann::json data_;
  int shm_fd_{-1};
  uint8_t * shm_data_{nullptr};

  void append_data(const WebDebugContext & context);
  void write_json_atomically(const std::string & path, const nlohmann::json & json) const;
  void write_frame(const cv::Mat & image);
};

}  // namespace tools

#endif  // TOOLS__WEB_DEBUG_HPP
