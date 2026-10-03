#include "buff_frame_processor.hpp"

#include <vector>

#include <fmt/format.h>
#include <opencv2/imgproc.hpp>

#include "detectors/rune_detector_factory.hpp"

namespace auto_buff_v2
{
BuffFrameProcessor::BuffFrameProcessor(RuneModel & model, BuffConfig::Detector detector_config)
: model_(model), detector_(make_rune_detector(detector_config))
{
}

tools::ProcessedFrame BuffFrameProcessor::process(const tools::FrameFacts & facts)
{
  if (const auto color = facts.enemy_color())
    detector_->set_enemy_red(*color == io::InfantryEnemyColor::red);
  model_.set_q_gimbal2world(facts.q_gimbal2world);
  tools::BuffDebugData buff_debug;
  const auto elements = detector_->detect(facts.image);
  model_.update(elements, facts.timestamp);
  const auto target = model_.state();

  buff_debug.is_buff_mode = true;
  buff_debug.candidates = elements.candidates;
  buff_debug.climber_contours = elements.climber_contours;
  buff_debug.bullseye_count = elements.bullseyes.size();
  buff_debug.icon_count = elements.icons.size();
  for (const auto & icon : elements.icons)
    buff_debug.detections.push_back({icon.center, {}, fmt::format("R: {:.3f}", icon.score)});
  for (const auto & bull : elements.bullseyes) {
    buff_debug.detections.push_back(
      {bull.center, bull.active ? std::vector<cv::Point2f>{} :
                                   std::vector<cv::Point2f>(
                                     bull.corners.begin(), bull.corners.end()),
       fmt::format("B: {:.3f} {}", bull.score, bull.active ? "active" : "inactive")});
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
        target->motion_fit_cost);
    } else if (target->has_fitted_motion) {
      buff_debug.info = fmt::format("spd_{}(t)={:+.2f}, e={:.3f}", target->update_count,
                                    target->rotation_speed, target->motion_fit_cost);
    } else {
      buff_debug.info = fmt::format("theta_ekf={:+.2f}", target->rotation_angle);
    }
  } else {
    buff_debug.info = model_.diagnostic();
    buff_debug.info_anchor = cv::Point2f(10, 50);
  }

  tools::ProcessedFrame result;
  result.snapshot = facts.snapshot({}, {}, {}, std::move(buff_debug));
  result.buff_target = target;
  return result;
}
}  // namespace auto_buff_v2
