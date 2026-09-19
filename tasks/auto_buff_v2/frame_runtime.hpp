#ifndef AUTO_BUFF_V2__FRAME_RUNTIME_HPP
#define AUTO_BUFF_V2__FRAME_RUNTIME_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
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
    tools::BuffDebugData buff_debug;
    for (const auto & icon : elements.icons)
      buff_debug.detections.push_back({icon.center, {}, fmt::format("R: {:.3f}", icon.score)});
    for (const auto & bull : elements.bullseyes) {
      buff_debug.detections.push_back(
        {bull.center, bull.active ? std::vector<cv::Point2f>{} :
                                     std::vector<cv::Point2f>(bull.corners.begin(), bull.corners.end()),
         fmt::format("B: {:.3f}", bull.score)});
    }
    std::array<cv::Point2f, 5> blades;
    bool complete_polygon = true;
    for (const auto & feature : model_.reprojected_features()) {
      buff_debug.reprojected_features.push_back(feature.point);
      if (feature.id == 0)
        buff_debug.icon = feature.point;
      else if (feature.id >= 1 && feature.id <= 5)
        blades[static_cast<std::size_t>(feature.id - 1)] = feature.point;
      else
        complete_polygon = false;
    }
    complete_polygon = complete_polygon && buff_debug.reprojected_features.size() == 6;
    if (complete_polygon) buff_debug.blade_polygon = blades;
    buff_debug.info_anchor = model_.reprojected_center();
    if (target) {
      if (target->sine_valid) {
        buff_debug.info = fmt::format(
          "spd_{}(t)={:+.2f}{:+.2f}*sin({:+.2f}{:+.2f}t), e={:.3f}", target->update_count,
          target->sine_v, target->sine_a, target->sine_phase, target->sine_omega,
          target->prediction_cost);
      } else if (target->use_prediction_speed) {
        buff_debug.info = fmt::format("spd_{}(t)={:+.2f}, e={:.3f}", target->update_count,
                                      target->rotation_speed, target->prediction_cost);
      } else {
        buff_debug.info = fmt::format("theta_ekf={:+.2f}", target->rotation_angle);
      }
    }
    const auto sent = gimbal_.command_with_packet();
    result.snapshot = tools::FrameSnapshot::capture(
      timestamp, image, orientation, received.state, sent.command, {}, {}, sent.packet,
      received.packet, std::move(buff_debug));
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
