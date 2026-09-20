#include <cassert>
#include <cmath>

#include <opencv2/imgproc.hpp>

#include "tasks/auto_buff_v2/rune_detector.hpp"

int main()
{
  cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::circle(image, {200, 200}, 42, {255, 0, 0}, cv::FILLED);
  auto_buff_v2::RuneDetector detector;
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

  detector.config.enemy_red = true;
  assert(detector.detect(image).bullseyes.empty());
}
