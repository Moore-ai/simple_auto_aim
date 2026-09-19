#ifndef AUTO_BUFF_V2__FRAME_RUNTIME_HPP
#define AUTO_BUFF_V2__FRAME_RUNTIME_HPP

#include <optional>
#include <vector>

#include <fmt/format.h>
#include <opencv2/imgproc.hpp>
#include "io/gimbal/gimbal.hpp"
#include "rune_detector.hpp"
#include "rune_model.hpp"
#include "tools/frame_facts.hpp"
#include "tools/processed_frame.hpp"

namespace auto_buff_v2
{
class FrameRuntime
{
public:
  FrameRuntime(io::Camera & camera, io::Gimbal & gimbal, RuneModel & model,
               BuffConfig::Detector detector_config)
  : frames_(camera, gimbal), model_(model)
  {
    static_cast<BuffConfig::Detector &>(detector_.config) = std::move(detector_config);
  }

  bool next(tools::ProcessedFrame & result)
  {
    tools::FrameFacts facts;
    if (!frames_.next(facts)) return false;
    if (const auto color = io::infantry_enemy_color(facts.received.state.mode))
      detector_.config.enemy_red = *color == io::InfantryEnemyColor::red;
    model_.update_transform(facts.gimbal_orientation);
    const auto elements = detector_.detect(facts.image);
    model_.update(elements, facts.timestamp);
    const auto target = model_.state();
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
    result.snapshot =
      frames_.complete(facts, target.has_value(), {}, {}, std::move(buff_debug));
    result.targets.clear();
    result.buff_target = target;
    return true;
  }

private:
  tools::FrameCapture frames_;
  RuneModel & model_;
  RuneDetector detector_;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__FRAME_RUNTIME_HPP
