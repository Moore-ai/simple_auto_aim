#include <cassert>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "tools/buff_detector_debug.hpp"

int main()
{
  auto_buff_v2::BuffConfig::Detector detector;
  detector.min_distance = 2;
  detector.max_distance = 5;
  tools::FrameFacts facts{};
  facts.timestamp = std::chrono::steady_clock::now();
  facts.q_gimbal2world = Eigen::Quaterniond::Identity();
  facts.target_color_override = io::InfantryEnemyColor::blue;
  // Both objects in one frame must produce disjoint debug observations.
  auto mixed_detector = detector;
  mixed_detector.fx = mixed_detector.fy = 5000;
  cv::Mat mixed = cv::Mat::zeros(700, 900, CV_8UC3);
  cv::circle(mixed, {600, 350}, 170, {255, 0, 0}, cv::FILLED);
  cv::putText(mixed, "R", {120, 260}, cv::FONT_HERSHEY_SIMPLEX, 3, {255, 0, 0}, 12,
              cv::LINE_8);
  auto_buff_v2::RuneDetector joint_detector;
  static_cast<auto_buff_v2::BuffConfig::Detector &>(joint_detector.config) = mixed_detector;
  const auto joint = joint_detector.detect(mixed);
  assert(joint.icons.size() == 1 && joint.bullseyes.size() == 1);
  tools::BuffDetectorDebug icon_processor(mixed_detector, tools::BuffDetectorDebugTarget::icon);
  tools::BuffDetectorDebug bull_processor(mixed_detector, tools::BuffDetectorDebugTarget::bullseye);
  auto mixed_facts = facts;
  mixed_facts.image = mixed;
  const auto icon_frame = icon_processor.process(mixed_facts);
  const auto bull_frame = bull_processor.process(mixed_facts);
  const auto & icon_debug = icon_frame.snapshot.buff_debug;
  const auto & bull_debug = bull_frame.snapshot.buff_debug;
  assert(icon_debug.icon_count == 1 && icon_debug.bullseye_count == 0);
  assert(bull_debug.icon_count == 0 && bull_debug.bullseye_count == 1);
  assert(icon_debug.detections.size() == 1);
  assert(bull_debug.detections.size() == 1);
  assert(icon_debug.detections.front().label.rfind("R: ", 0) == 0);
  assert(bull_debug.detections.front().label.rfind("B: ", 0) == 0);
  assert(icon_frame.measurements.candidates.size() == 1);
  assert(bull_frame.measurements.candidates.size() == 1);
  assert(cv::norm(
    icon_frame.measurements.candidates.front().center - joint.icons.front().center) < 2);
  assert(cv::norm(
    bull_frame.measurements.candidates.front().center - joint.bullseyes.front().center) < 2);
  for (const auto * debug : {&icon_debug, &bull_debug}) {
    assert(debug->reprojected_features.empty());
    assert(!debug->blade_polygon && !debug->icon && !debug->info_anchor);
    assert(debug->info.empty());
  }
  const auto icon_image = tools::detail::buff_detector_debug_image(icon_frame);
  const auto bull_image = tools::detail::buff_detector_debug_image(bull_frame);
  const cv::Rect bull_roi(420, 170, 360, 360);
  const cv::Rect icon_roi(80, 140, 160, 160);
  assert(cv::norm(icon_image(bull_roi), mixed(bull_roi), cv::NORM_INF) == 0);
  assert(cv::norm(bull_image(icon_roi), mixed(icon_roi), cv::NORM_INF) == 0);
  for (const auto * frame : {&icon_frame, &bull_frame}) {
    const auto marked = tools::detail::buff_detector_debug_image(*frame);
    const auto point = frame->snapshot.buff_debug.detections.front().point;
    const auto pixel = marked.at<cv::Vec3b>(cvRound(point.y) - 5, cvRound(point.x));
    assert(pixel[0] == 0 && pixel[1] > 0 && pixel[2] == 0);
    assert(cv::norm(frame->snapshot.image, mixed, cv::NORM_INF) == 0);
  }

  cv::Mat bull_only = mixed.clone();
  bull_only(cv::Rect(80, 140, 160, 160)).setTo(cv::Scalar{});
  mixed_facts.image = bull_only;
  const auto no_icon = icon_processor.process(mixed_facts);
  assert(no_icon.snapshot.buff_debug.detections.empty());
  assert(no_icon.measurements.candidates.empty());
  cv::Mat icon_only = mixed.clone();
  icon_only(cv::Rect(420, 170, 360, 360)).setTo(cv::Scalar{});
  mixed_facts.image = icon_only;
  const auto no_bull = bull_processor.process(mixed_facts);
  assert(no_bull.snapshot.buff_debug.detections.empty());
  assert(no_bull.measurements.candidates.empty());

  auto_buff_v2::RuneDetectorMeasurements detector_debug;
  detector_debug.min_radius = 1.35;
  detector_debug.max_radius = 16.5;
  detector_debug.min_icon_area = 0.28;
  detector_debug.max_icon_area = 43.2;
  detector_debug.candidates.push_back({{20, 30}, 20, 1256, false, false});
  const auto detector_values = tools::detail::buff_detector_values(detector_debug);
  assert(detector_values.at("min_radius") == 1.35);
  assert(detector_values.at("max_radius") == 16.5);
  assert(detector_values.at("min_icon_area") == 0.28);
  assert(detector_values.at("max_icon_area") == 43.2);
  assert(detector_values.at("candidates").at(0).at("radius") == 20);
  assert(detector_values.at("candidates").at(0).at("area") == 1256);
  assert(detector_values.at("candidates").at(0).at("center_x") == 20);
  detector_debug.candidates.clear();
  assert(tools::detail::buff_detector_values(detector_debug).at("candidates").empty());
  auto detector_channel = tools::detail::create_buff_detector_channel();
  assert(detector_channel.has_value());
  assert(detector_channel.value().topic() == "/buff_v2/detector");
  const auto detector_schema = detector_channel.value().schema();
  assert(detector_schema);
  const auto detector_schema_json = nlohmann::json::parse(
    reinterpret_cast<const char *>(detector_schema->data),
    reinterpret_cast<const char *>(detector_schema->data) + detector_schema->data_len);
  assert(detector_schema_json.at("properties").at("candidates").at("items")
           .at("properties").at("radius").at("type") == "number");
}
