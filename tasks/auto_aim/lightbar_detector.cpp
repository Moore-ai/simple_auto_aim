#include "lightbar_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{

LightbarDetector::LightbarDetector(const std::string & config_path)
{
  const auto yaml = YAML::LoadFile(config_path);
  threshold_ = yaml["threshold"].as<double>();
  max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3;
  min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
  max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
  min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
}

std::list<Lightbar> LightbarDetector::detect(const cv::Mat & bgr_img) const
{
  cv::Mat gray_img;
  cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

  cv::Mat binary_img;
  cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  std::list<Lightbar> lightbars;
  std::size_t lightbar_id = 0;
  for (const auto & contour : contours) {
    auto lightbar = Lightbar(cv::minAreaRect(contour), lightbar_id);
    if (!check_geometry(lightbar)) continue;
    lightbar.color = get_color(bgr_img, contour);
    lightbars.emplace_back(std::move(lightbar));
    ++lightbar_id;
  }
  lightbars.sort([](const Lightbar & lhs, const Lightbar & rhs) {
    return lhs.center.x < rhs.center.x;
  });
  return lightbars;
}

bool LightbarDetector::check_geometry(const Lightbar & lightbar) const
{
  return lightbar.angle_error < max_angle_error_ &&
    lightbar.ratio > min_lightbar_ratio_ && lightbar.ratio < max_lightbar_ratio_ &&
    lightbar.length > min_lightbar_length_;
}

Color LightbarDetector::get_color(
  const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const
{
  int red_sum = 0;
  int blue_sum = 0;
  for (const auto & point : contour) {
    red_sum += bgr_img.at<cv::Vec3b>(point)[2];
    blue_sum += bgr_img.at<cv::Vec3b>(point)[0];
  }
  return blue_sum > red_sum ? Color::blue : Color::red;
}

}  // namespace auto_aim
