#ifndef AUTO_BUFF_V2__FRAME_RUNTIME_HPP
#define AUTO_BUFF_V2__FRAME_RUNTIME_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "rune_detector.hpp"
#include "rune_model.hpp"
#include "tools/processed_frame.hpp"

namespace auto_buff_v2
{
class FrameRuntime
{
public:
  FrameRuntime(io::Camera & camera, io::Gimbal & gimbal, RuneModel & model,
               const std::string & config_path)
  : camera_(camera), gimbal_(gimbal), model_(model)
  {
    const auto yaml = YAML::LoadFile(config_path);
    const auto intrinsics = yaml["camera_matrix"].as<std::vector<double>>();
    detector_.config.fx = intrinsics[0];
    detector_.config.fy = intrinsics[4];
    const auto buff = yaml["buff_v2"];
    if (buff) {
      detector_.config.min_distance =
        buff["min_distance"].as<double>(detector_.config.min_distance);
      detector_.config.max_distance =
        buff["max_distance"].as<double>(detector_.config.max_distance);
      detector_.config.active_threshold =
        buff["active_threshold"].as<double>(detector_.config.active_threshold);
      detector_.config.match_threshold =
        buff["match_threshold"].as<double>(detector_.config.match_threshold);
    }
  }

  bool next(tools::ProcessedFrame & result)
  {
    cv::Mat image;
    Timestamp timestamp;
    if (!camera_.read(image, timestamp)) return false;
    const auto received = gimbal_.state_with_packet();
    if (const auto color = io::infantry_enemy_color(received.state.mode))
      detector_.config.enemy_red = *color == io::InfantryEnemyColor::red;
    const auto orientation = gimbal_.q(timestamp);
    model_.update_transform(orientation);
    const auto elements = detector_.detect(image);
    model_.update(elements, timestamp);
    const auto target = model_.state();
    const bool has_target = target.has_value();
    if (has_target && !had_target_) ++target_generation_;
    had_target_ = has_target;
    cv::Mat debug = image.clone();
    for (const auto & icon : elements.icons)
      cv::circle(debug, icon.center, 5, {0, 255, 255}, 2);
    for (const auto & bull : elements.bullseyes) {
      cv::circle(debug, bull.center, 5, bull.active ? cv::Scalar(0, 0, 255) :
                                                    cv::Scalar(0, 255, 0), 2);
      for (const auto & corner : bull.corners)
        cv::circle(debug, corner, 3, {255, 255, 0}, 1);
    }
    const auto sent = gimbal_.command_with_packet();
    result.snapshot = tools::FrameSnapshot::capture(
      timestamp, debug, orientation, received.state, sent.command, {}, {}, sent.packet,
      received.packet);
    result.snapshot.target_generation = target_generation_;
    result.targets.clear();
    result.buff_target = target;
    return true;
  }

private:
  io::Camera & camera_;
  io::Gimbal & gimbal_;
  RuneModel & model_;
  RuneDetector detector_;
  std::uint64_t target_generation_ = 0;
  bool had_target_ = false;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__FRAME_RUNTIME_HPP
