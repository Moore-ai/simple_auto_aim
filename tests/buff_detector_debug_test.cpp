#include <cassert>
#include <chrono>
#include <cmath>

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
  // The R scoring crop extends five pixels beyond the dilated R contour.
  // Its top edge is y=185 for this fixture; it must be drawn in observation green.
  const auto roi_pixel = icon_image.at<cv::Vec3b>(185, 150);
  assert(roi_pixel == cv::Vec3b(0, 255, 0));
  const auto no_roi_image = tools::detail::buff_detector_debug_image(icon_frame, false);
  assert(no_roi_image.at<cv::Vec3b>(185, 150) == mixed.at<cv::Vec3b>(185, 150));
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
  assert(no_icon.measurements.icon_rois.empty());
  cv::Mat icon_only = mixed.clone();
  icon_only(cv::Rect(420, 170, 360, 360)).setTo(cv::Scalar{});
  mixed_facts.image = icon_only;
  const auto no_bull = bull_processor.process(mixed_facts);
  assert(no_bull.snapshot.buff_debug.detections.empty());
  assert(no_bull.measurements.candidates.empty());

  // Small bright fragments and non-R contours must never appear as R observations.
  cv::Mat cluttered = mixed.clone();
  for (int i = 0; i < 12; ++i)
    cv::circle(cluttered, {20 + i * 30, 550}, 2, {255, 0, 0}, cv::FILLED);
  // This disk passes the R area gate but has no R skeleton features.
  cv::circle(cluttered, {300, 400}, 20, {255, 0, 0}, cv::FILLED);
  cv::ellipse(cluttered, {200, 450}, {45, 8}, 0, 0, 360, {255, 0, 0}, cv::FILLED);
  mixed_facts.image = cluttered;
  const auto clean_icon = icon_processor.process(mixed_facts);
  assert(clean_icon.snapshot.buff_debug.icon_count == 1);
  assert(clean_icon.measurements.candidates.size() == 1);
  assert(cv::norm(clean_icon.measurements.candidates.front().center -
                  joint.icons.front().center) < 2);
  const auto clean_values = tools::detail::buff_detector_values(clean_icon.measurements);
  assert(clean_values.at("candidates").size() == 1);
  const auto clean_image = tools::detail::buff_detector_debug_image(clean_icon);
  const cv::Rect clutter_roi(0, 360, 360, 240);
  assert(cv::norm(clean_image(clutter_roi), cluttered(clutter_roi), cv::NORM_INF) == 0);
  assert(cv::norm(clean_image(bull_roi), cluttered(bull_roi), cv::NORM_INF) == 0);

  // Even an area-eligible shape alone must yield an empty R observation array.
  mixed_facts.image = cv::Mat::zeros(700, 900, CV_8UC3);
  cv::circle(mixed_facts.image, {300, 400}, 20, {255, 0, 0}, cv::FILLED);
  const auto not_an_icon = icon_processor.process(mixed_facts);
  assert(not_an_icon.snapshot.buff_debug.detections.empty());
  assert(not_an_icon.measurements.candidates.empty());
  assert(not_an_icon.measurements.min_icon_area > 0);

  // Scores rejected by the threshold must remain available for threshold tuning.
  auto rejecting_detector = mixed_detector;
  rejecting_detector.match_threshold = 1.1;
  tools::BuffDetectorDebug rejecting_processor(
    rejecting_detector, tools::BuffDetectorDebugTarget::icon);
  mixed_facts.image = icon_only;
  const auto rejected = rejecting_processor.process(mixed_facts);
  assert(rejected.snapshot.buff_debug.detections.empty());
  const auto rejected_values = tools::detail::buff_detector_values(rejected.measurements);
  assert(rejected_values.contains("match_threshold"));
  assert(rejected_values.at("match_threshold") == 1.1);
  assert(rejected_values.at("icon_scores").size() == 1);
  const auto & rejected_score = rejected_values.at("icon_scores").at(0);
  assert(rejected_score.at("score").get<double>() >= mixed_detector.match_threshold);
  assert(rejected_score.at("score").get<double>() < 1.1);
  assert(rejected_score.contains("endpoints"));
  assert(rejected_score.at("endpoints").get<int>() >= 1);
  assert(rejected_score.at("lower_endpoints").get<int>() >= 1);
  assert(rejected_score.at("branches").get<int>() >= 8);
  assert(rejected_score.at("branches").get<int>() <= 50);
  assert(rejected_score.at("holes") == 1);
  assert(std::abs(rejected_score.at("center_x").get<double>() -
                  joint.icons.front().center.x) < 2);
  const auto accepted_values = tools::detail::buff_detector_values(icon_frame.measurements);
  assert(accepted_values.at("icon_scores").size() == 1);
  assert(accepted_values.at("icon_scores").at(0).at("score") == rejected_score.at("score"));
  for (const auto * name : {"endpoints", "lower_endpoints", "branches", "holes"})
    assert(accepted_values.at("icon_scores").at(0).at(name) == rejected_score.at(name));
  const auto disk_values = tools::detail::buff_detector_values(not_an_icon.measurements);
  assert(disk_values.at("icon_scores").size() == 1);
  for (const auto * name : {"endpoints", "lower_endpoints", "branches", "holes"})
    assert(disk_values.at("icon_scores").at(0).at(name) == 0);
  const auto empty_scores = tools::detail::buff_detector_values(no_icon.measurements);
  assert(empty_scores.at("icon_scores").empty());
  assert(empty_scores.at("match_threshold") == mixed_detector.match_threshold);

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
  assert(detector_schema_json.at("properties").at("match_threshold").at("type") == "number");
  assert(detector_schema_json.at("properties").at("icon_scores").at("items")
           .at("properties").at("score").at("type") == "number");
  for (const auto * name : {"endpoints", "lower_endpoints", "branches", "holes"})
    assert(detector_schema_json.at("properties").at("icon_scores").at("items")
             .at("properties").at(name).at("type") == "integer");
}
