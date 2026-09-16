#include <cassert>
#include <cmath>
#include <fstream>

#include <opencv2/calib3d.hpp>

#include "tasks/auto_buff_v2/rune_model.hpp"

int main()
{
  const char * path = "/tmp/buff_v2_pnp_test.yaml";
  std::ofstream(path)
    << "camera_matrix: [1000, 0, 320, 0, 1000, 240, 0, 0, 1]\n"
    << "distort_coeffs: [0, 0, 0, 0, 0]\n"
    << "R_gimbal2imubody: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
    << "camera2gimbal_mode: matrix\n"
    << "R_camera2gimbal: [0, 0, 1, -1, 0, 0, 0, -1, 0]\n"
    << "t_camera2gimbal: [0, 0, 0]\n";

  cv::Mat camera = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
  cv::Mat rotation = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  const std::vector<cv::Point3f> object = {
    {-0.1f, 0, 0}, {0, 0, 0.85f}, {0, 0.15f, 0.7f},
    {0, 0, 0.55f}, {0, -0.15f, 0.7f}};
  std::vector<cv::Point2f> pixel;
  cv::projectPoints(object, rvec, cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), pixel);
  auto_buff_v2::RuneElements elements;
  elements.icons.push_back({pixel[0], 1});
  elements.bullseyes.push_back({pixel[1], {pixel[2], pixel[3], pixel[4], pixel[1]}, false, 0.5});
  auto_buff_v2::RuneModel model(path, false);
  model.update_transform(Eigen::Quaterniond::Identity());
  const auto now = std::chrono::steady_clock::now();
  assert(model.update(elements, now));
  const auto state = model.state();
  assert(state);
  assert((state->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);
  assert(std::abs(state->rotation_angle) < 0.03);

  const double seed_angle = 0.3;
  cv::Mat tilted = (cv::Mat_<double>(3, 3) <<
    1, 0, 0, 0, std::cos(seed_angle), -std::sin(seed_angle),
    0, std::sin(seed_angle), std::cos(seed_angle));
  cv::Mat tilted_rvec;
  cv::Rodrigues(rotation * tilted, tilted_rvec);
  std::vector<cv::Point2f> tilted_pixel;
  cv::projectPoints(object, tilted_rvec, cv::Vec3d(0, 0, 3), camera,
                    cv::Mat::zeros(1, 5, CV_64F), tilted_pixel);
  auto_buff_v2::RuneElements tilted_elements;
  tilted_elements.icons.push_back({tilted_pixel[0], 1});
  tilted_elements.bullseyes.push_back(
    {tilted_pixel[1], {tilted_pixel[2], tilted_pixel[3], tilted_pixel[4], tilted_pixel[1]},
     false, 0.5});
  auto_buff_v2::RuneModel tilted_model(path, false);
  tilted_model.update_transform(Eigen::Quaterniond::Identity());
  assert(tilted_model.update(tilted_elements, now));
  assert(std::abs(tilted_model.state()->rotation_angle - seed_angle) < 0.03);

  auto_buff_v2::RuneModel big(path, true);
  big.update_transform(Eigen::Quaterniond::Identity());
  assert(big.update(elements, now));
  for (int i = 1; i <= 120; ++i) {
    const double t = 0.05 * i;
    const double angle = t - 0.4 * (std::cos(2 * t + 0.4) - std::cos(0.4));
    std::vector<cv::Point2f> blade;
    cv::projectPoints(std::vector<cv::Point3f>{{0, static_cast<float>(-0.7 * std::sin(angle)),
                                                static_cast<float>(0.7 * std::cos(angle))}},
                      rvec, cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), blade);
    elements.bullseyes.front().center = blade.front();
    const auto stamp = now + std::chrono::milliseconds(50 * i);
    model.update(elements, stamp);
    big.update(elements, stamp);
  }
  assert(model.state() && !model.state()->sine_valid);
  assert(big.state() && big.state()->sine_valid);
}
