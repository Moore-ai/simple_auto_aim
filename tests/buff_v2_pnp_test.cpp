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
  std::vector<cv::Point2f> bull_center;
  cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.7f}}, rvec, cv::Vec3d(0, 0, 3),
                    camera, cv::Mat::zeros(1, 5, CV_64F), bull_center);
  auto_buff_v2::RuneElements elements;
  elements.icons.push_back({pixel[0], 1});
  elements.bullseyes.push_back(
    {bull_center[0], {pixel[2], pixel[3], pixel[4], pixel[1]}, false, 0.5});
  auto_buff_v2::RuneModel model(path, false);
  model.update_transform(Eigen::Quaterniond::Identity());
  const auto now = std::chrono::steady_clock::now();
  assert(model.update(elements, now));
  const auto state = model.state();
  assert(state);
  assert((state->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);
  assert(std::abs(state->rotation_angle) < 0.03);
  assert(!state->inactive[0]);

  auto_buff_v2::RuneElements candidates = elements;
  candidates.icons.insert(candidates.icons.begin(),
                          {pixel[0] + cv::Point2f(12, 0), 0.5});
  auto_buff_v2::RuneModel selected(path, false);
  selected.update_transform(Eigen::Quaterniond::Identity());
  assert(selected.update(candidates, now));
  assert((selected.state()->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);

  constexpr double kBladeAngle = 2 * 3.14159265358979323846 / 5;
  const auto rotate_blade = [=](double y, double z) {
    return cv::Point3f(0, static_cast<float>(std::cos(kBladeAngle) * y -
                                               std::sin(kBladeAngle) * z),
                       static_cast<float>(std::sin(kBladeAngle) * y +
                                          std::cos(kBladeAngle) * z));
  };
  const std::vector<cv::Point3f> second_object = {
    rotate_blade(0, 0.7), rotate_blade(0, 0.85),
    rotate_blade(0.15, 0.7), rotate_blade(0, 0.55),
    rotate_blade(-0.15, 0.7)};
  std::vector<cv::Point2f> second_pixel;
  cv::projectPoints(second_object, rvec, cv::Vec3d(0, 0, 3), camera,
                    cv::Mat::zeros(1, 5, CV_64F), second_pixel);
  auto_buff_v2::RuneElements layout = candidates;
  layout.bullseyes.insert(layout.bullseyes.begin(),
                          {second_pixel[0], {second_pixel[2], second_pixel[3],
                                             second_pixel[4], second_pixel[1]}, false, 0.5});
  auto_buff_v2::RuneModel layout_model(path, false);
  layout_model.update_transform(Eigen::Quaterniond::Identity());
  assert(layout_model.update(layout, now));
  assert((layout_model.state()->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);
  assert(std::abs(layout_model.state()->rotation_angle) < 0.03);

  auto_buff_v2::RuneElements wrong_center = elements;
  wrong_center.bullseyes.front().center += cv::Point2f(35, 0);
  auto_buff_v2::RuneModel gated(path, false);
  gated.update_transform(Eigen::Quaterniond::Identity());
  assert(!gated.update(wrong_center, now));

  auto_buff_v2::RuneElements degenerate = elements;
  degenerate.bullseyes.front().corners[0] =
    degenerate.bullseyes.front().corners[2];
  auto_buff_v2::RuneModel rejected(path, false);
  rejected.update_transform(Eigen::Quaterniond::Identity());
  assert(!rejected.update(degenerate, now));

  auto_buff_v2::RuneElements noisy = elements;
  noisy.bullseyes.front().corners[3] += cv::Point2f(3, 0);
  const std::vector<cv::Point2f> noisy_points = {
    noisy.icons.front().center, noisy.bullseyes.front().corners[3],
    noisy.bullseyes.front().corners[0], noisy.bullseyes.front().corners[1],
    noisy.bullseyes.front().corners[2]};
  cv::Vec3d noisy_rvec, noisy_tvec;
  assert(cv::solvePnP(object, noisy_points, camera, cv::Mat::zeros(1, 5, CV_64F),
                      noisy_rvec, noisy_tvec, false, cv::SOLVEPNP_EPNP));
  assert(cv::solvePnP(object, noisy_points, camera, cv::Mat::zeros(1, 5, CV_64F),
                      noisy_rvec, noisy_tvec, true, cv::SOLVEPNP_ITERATIVE));
  const Eigen::Vector3d raw_center(noisy_tvec[2], -noisy_tvec[0], -noisy_tvec[1]);
  auto_buff_v2::RuneModel corrected_seed(path, false);
  corrected_seed.update_transform(Eigen::Quaterniond::Identity());
  assert(corrected_seed.update(noisy, now));
  assert((corrected_seed.state()->center - raw_center).norm() > 0.002);

  const double seed_angle = 0.3;
  cv::Mat tilted = (cv::Mat_<double>(3, 3) <<
    1, 0, 0, 0, std::cos(seed_angle), -std::sin(seed_angle),
    0, std::sin(seed_angle), std::cos(seed_angle));
  cv::Mat tilted_rvec;
  cv::Rodrigues(rotation * tilted, tilted_rvec);
  std::vector<cv::Point2f> tilted_pixel;
  cv::projectPoints(object, tilted_rvec, cv::Vec3d(0, 0, 3), camera,
                    cv::Mat::zeros(1, 5, CV_64F), tilted_pixel);
  std::vector<cv::Point2f> tilted_center;
  cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.7f}}, tilted_rvec,
                    cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), tilted_center);
  auto_buff_v2::RuneElements tilted_elements;
  tilted_elements.icons.push_back({tilted_pixel[0], 1});
  tilted_elements.bullseyes.push_back(
    {tilted_center[0], {tilted_pixel[2], tilted_pixel[3], tilted_pixel[4], tilted_pixel[1]},
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

  auto_buff_v2::RuneElements empty;
  for (int i = 121; i <= 153; ++i)
    big.update(empty, now + std::chrono::milliseconds(50 * i));
  assert(!big.state());
  assert(big.update(tilted_elements, now + std::chrono::milliseconds(7700)));
  assert(std::abs(big.state()->rotation_angle - seed_angle) < 0.03);
}
