#include <cassert>
#include <cmath>
#include <list>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "tools/foxglove_visualizer.hpp"

int main()
{
  cv::Mat input(1, 3, CV_8UC3);
  input.at<cv::Vec3b>(0, 0) = {1, 2, 3};
  input.at<cv::Vec3b>(0, 1) = {4, 5, 6};
  input.at<cv::Vec3b>(0, 2) = {7, 8, 9};

  const auto output = tools::detail::prepare_image_for_publish(input);
  assert(cv::norm(input, output, cv::NORM_INF) == 0.0);

  const std::vector<cv::Point2f> red_points = {{10, 10}, {30, 10}, {30, 30}, {10, 30}};
  const std::vector<cv::Point2f> blue_points = {{60, 10}, {80, 10}, {80, 30}, {60, 30}};
  auto red_armor = auto_aim::Armor(1, 0.9F, {10, 10, 20, 20}, red_points);
  auto blue_armor = auto_aim::Armor(0, 0.8F, {60, 10, 20, 20}, blue_points);
  std::list<auto_aim::Armor> armors = {red_armor, blue_armor};

  cv::Mat detection_image = cv::Mat::zeros(40, 90, CV_8UC3);
  tools::detail::draw_detected_armors(detection_image, armors, auto_aim::red);
  const auto channel_sum = cv::sum(detection_image);
  assert(channel_sum[2] > 0.0);
  assert(channel_sum[0] == 0.0);

  const double yaw = 0.3;
  const double pitch = CV_PI / 12.0;
  const auto cube = tools::detail::armor_cube({-0.3, 0.0, 0.0}, yaw, pitch, auto_aim::big);
  assert(cube.pose.has_value());
  assert(cube.pose->orientation.has_value());
  const auto & orientation = *cube.pose->orientation;
  const Eigen::Quaterniond visualization_rotation{
    orientation.w, orientation.x, orientation.y, orientation.z};
  const Eigen::Quaterniond expected_rotation{
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())};
  assert(visualization_rotation.angularDistance(expected_rotation) < 1e-12);

  assert(cube.size.has_value());
  assert(std::abs(cube.size->x - 0.020) < 1e-12);
  assert(std::abs(cube.size->y - 0.230) < 1e-12);
  assert(std::abs(cube.size->z - 0.130) < 1e-12);
  return 0;
}
