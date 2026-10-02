#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <opencv2/imgproc.hpp>

#include "tasks/auto_buff_v2/detectors/njust_rune_detector.hpp"
#include "tasks/auto_buff_v2/detectors/rune_detector_factory.hpp"

int main()
{
  cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::circle(image, {200, 200}, 42, {255, 0, 0}, cv::FILLED);
  static_assert(std::is_abstract_v<auto_buff_v2::RuneDetector>);
  auto_buff_v2::NjustRuneDetector detector;
  detector.config.fx = 1000;
  detector.config.fy = 1000;
  detector.config.min_distance = 2;
  detector.config.max_distance = 5;
  detector.config.enemy_red = false;
  const auto blue = detector.detect(image);
  assert(blue.bullseyes.size() == 1);
  assert(cv::norm(blue.bullseyes.front().center - cv::Point2f(200, 200)) < 2);

  cv::Mat cluttered = image.clone();
  cv::rectangle(cluttered, {250, 195}, {258, 205}, {0, 0, 255}, cv::FILLED);
  const auto clutter = detector.detect(cluttered);
  assert(clutter.bullseyes.size() == 1);
  assert(std::abs(clutter.bullseyes.front().score - blue.bullseyes.front().score) < 1e-6);

  cv::Mat spokes = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::circle(spokes, {200, 200}, 42, {90, 0, 0}, cv::FILLED);
  for (int p = 0; p < 4; ++p) {
    const double angle = p * 3.14159265358979323846 / 2;
    const cv::Point end(200 + 40 * std::cos(angle), 200 + 40 * std::sin(angle));
    cv::line(spokes, {200, 200}, end, {255, 200, 0}, 8);
  }
  const auto cross = detector.detect(spokes);
  assert(cross.bullseyes.size() == 1);
  assert(cross.bullseyes.front().score >= detector.config.active_threshold);
  for (const auto & tip : cross.bullseyes.front().corners)
    assert(cv::norm(tip - cross.bullseyes.front().center) > 35);

  cv::Mat uneven = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::circle(uneven, {200, 200}, 42, {90, 0, 0}, cv::FILLED);
  for (int p = 0; p < 4; ++p) {
    const double angle = p * 3.14159265358979323846 / 2;
    const int length = p == 0 ? 20 : 40;
    const cv::Point end(200 + length * std::cos(angle), 200 + length * std::sin(angle));
    cv::line(uneven, {200, 200}, end, {255, 200, 0}, 8);
  }
  const auto uneven_result = detector.detect(uneven);
  assert(uneven_result.bullseyes.size() == 1);
  assert(uneven_result.bullseyes.front().score >= detector.config.active_threshold);
  double shortest = 100;
  double longest = 0;
  for (const auto & tip : uneven_result.bullseyes.front().corners)
  {
    const double length = cv::norm(tip - uneven_result.bullseyes.front().center);
    shortest = std::min(shortest, length);
    longest = std::max(longest, length);
  }
  assert(shortest < 30);
  assert(longest > 35);

  cv::Mat perspective = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::ellipse(perspective, {200, 200}, {80, 45}, 0, 0, 360, {90, 0, 0}, cv::FILLED);
  cv::line(perspective, {122, 200}, {278, 200}, {255, 200, 0}, 8);
  cv::line(perspective, {200, 157}, {200, 243}, {255, 200, 0}, 8);
  const auto perspective_result = detector.detect(perspective);
  assert(perspective_result.bullseyes.size() == 1);
  double perspective_longest = 0;
  for (const auto & tip : perspective_result.bullseyes.front().corners)
    perspective_longest = std::max(
      perspective_longest, cv::norm(tip - perspective_result.bullseyes.front().center));
  assert(perspective_longest > 75);

  cv::Mat icon_image = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::putText(icon_image, "R", {120, 260}, cv::FONT_HERSHEY_SIMPLEX, 3, {255, 0, 0}, 12,
              cv::LINE_8);
  cv::Mat icon_binary;
  cv::inRange(icon_image, cv::Scalar(200, 0, 0), cv::Scalar(255, 0, 0), icon_binary);
  cv::dilate(icon_binary, icon_binary, cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3}));
  std::vector<std::vector<cv::Point>> icon_contours;
  cv::findContours(icon_binary, icon_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  assert(icon_contours.size() == 1);
  const cv::Point2f expected_icon_center = cv::minAreaRect(icon_contours.front()).center;
  detector.config.fx = 5000;
  detector.config.fy = 5000;
  const auto icon_result = detector.detect(icon_image);
  assert(icon_result.icons.size() == 1);
  assert(cv::norm(icon_result.icons.front().center - expected_icon_center) < 0.1);

  // Diagnostics must retain measurements even when the size gates reject the R.
  detector.measurements.emplace();
  const auto & debug = *detector.measurements;
  detector.config.fx = 100;
  detector.config.fy = 100;
  const auto rejected_icon = detector.detect(icon_image);
  assert(rejected_icon.icons.empty());
  assert(debug.candidates.size() == 1);
  assert(debug.candidates.front().area > debug.max_icon_area);
  assert(debug.candidates.front().radius > debug.max_radius);
  assert(!debug.candidates.front().icon_area_pass);
  assert(std::abs(debug.min_radius - 1.35) < 1e-6);
  assert(std::abs(debug.max_radius - 16.5) < 1e-6);
  assert(std::abs(debug.min_icon_area - 0.09 * 3.14159265358979323846) < 1e-6);
  assert(std::abs(debug.max_icon_area - 13.75 * 3.14159265358979323846) < 1e-6);
  assert(std::abs(debug.candidates.front().radius -
                  std::sqrt(debug.candidates.front().area / 3.14159265358979323846)) < 1e-6);
  detector.detect(cv::Mat::zeros(400, 400, CV_8UC3));
  assert(debug.candidates.empty());
  assert(debug.max_radius > 0);
  detector.detect(cv::Mat{});
  assert(debug.candidates.empty());
  assert(debug.max_radius == 0);

  detector.config.enemy_red = true;
  assert(detector.detect(image).bullseyes.empty());

  auto_buff_v2::BuffConfig::Detector factory_config;
  factory_config.fx = factory_config.fy = 1000;
  factory_config.parameters["njust"]["min_distance"] = 2;
  auto selected = auto_buff_v2::make_rune_detector(factory_config);
  selected->set_enemy_red(false);
  assert(selected->detect(image).bullseyes.size() == 1);
  selected->set_enemy_red(true);
  assert(selected->detect(image).bullseyes.empty());
  factory_config.parameters["njust"]["min_distance"] = 20;
  factory_config.parameters["njust"]["max_distance"] = 25;
  auto far_detector = auto_buff_v2::make_rune_detector(factory_config);
  assert(far_detector->detect(image).bullseyes.empty());
  factory_config.type = "unknown";
  bool rejected = false;
  try {
    auto_buff_v2::make_rune_detector(factory_config);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);

}
