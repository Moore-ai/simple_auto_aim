#include "sp_rune_solver.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include "tools/math_tools.hpp"

namespace auto_buff_v2
{
SPRuneSolver::SPRuneSolver(BuffConfig::Camera camera, SPRuneConfig config)
: camera_(std::move(camera)), config_(std::move(config))
{
  config_.validate();
}

void SPRuneSolver::set_q_gimbal2world(const Eigen::Quaterniond & q)
{
  attitude_valid_ = q.coeffs().allFinite() && q.norm() > 1e-12;
  if (attitude_valid_) q_gimbal2world_ = q.normalized();
}

std::vector<SPRuneObservation> SPRuneSolver::solve_all(const RuneElements & elements) const
{
  std::vector<SPRuneObservation> result;
  for (const auto & bull : elements.bullseyes) {
    if (bull.active || elements.icons.empty()) continue;
    const auto icon = std::min_element(elements.icons.begin(), elements.icons.end(),
      [&](const auto & a, const auto & b) {
        return cv::norm(a.center - bull.center) < cv::norm(b.center - bull.center);
      });
    if (auto observation = solve_one(bull, *icon)) result.push_back(*observation);
  }
  return result;
}

std::optional<SPRuneObservation> SPRuneSolver::solve(const RuneElements & elements) const
{
  auto observations = solve_all(elements);
  if (observations.empty()) return std::nullopt;
  return *std::max_element(observations.begin(), observations.end(),
    [](const auto & a, const auto & b) { return a.score < b.score; });
}

std::optional<SPRuneObservation> SPRuneSolver::solve_one(
  const RuneBullseye & bull, const RuneIcon & icon) const
{
  if (!attitude_valid_) return std::nullopt;
  auto corners = bull.corners;
  const auto top = std::max_element(
    corners.begin(), corners.end(), [&](const auto & a, const auto & b) {
    return cv::norm(a - icon.center) < cv::norm(b - icon.center);
  });
  const auto bottom = std::min_element(
    corners.begin(), corners.end(), [&](const auto & a, const auto & b) {
    return cv::norm(a - icon.center) < cv::norm(b - icon.center);
  });
  if (top == bottom || cv::norm(*top - *bottom) < 1e-6) return std::nullopt;
  std::vector<cv::Point2f> sides;
  for (auto it = corners.begin(); it != corners.end(); ++it)
    if (it != top && it != bottom) sides.push_back(*it);
  const auto up = *top - *bottom;
  const cv::Point2f right(up.y, -up.x);
  if ((sides[0] - bull.center).dot(right) < (sides[1] - bull.center).dot(right))
    std::swap(sides[0], sides[1]);
  std::vector<cv::Point3f> object{
    {0, 0, static_cast<float>(config_.target_radius + config_.target_top)},
    {0, static_cast<float>(config_.target_left), static_cast<float>(config_.target_radius)},
    {0, 0, static_cast<float>(config_.target_radius - config_.target_bottom)},
    {0, -static_cast<float>(config_.target_right), static_cast<float>(config_.target_radius)}};
  std::vector<cv::Point2f> image{*top, sides[0], *bottom, sides[1]};
  if (std::abs(cv::contourArea(image)) < 1) return std::nullopt;
  cv::Vec3d rvec, tvec;
  cv::Mat rmat;
  Eigen::Matrix3d R_buff2camera;
  try {
    if (!cv::solvePnP(object, image, camera_.camera_matrix, camera_.distort_coeffs,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE)) return std::nullopt;
    if (config_.pnp_mode != "4") {
      if (config_.pnp_mode == "5") {
        object.emplace_back(0, 0, config_.r_center_z);
        image.push_back(icon.center);
      }
      if (!cv::solvePnP(object, image, camera_.camera_matrix, camera_.distort_coeffs,
                        rvec, tvec, true, cv::SOLVEPNP_ITERATIVE)) return std::nullopt;
    }
    cv::Rodrigues(rvec, rmat);
    cv::cv2eigen(rmat, R_buff2camera);
  } catch (const cv::Exception &) { return std::nullopt; }
  const Eigen::Vector3d origin(tvec[0], tvec[1], tvec[2]);
  if (!origin.allFinite() || !R_buff2camera.allFinite() ||
      origin.z() <= config_.min_depth || origin.z() >= config_.max_depth) return std::nullopt;
  for (const auto & p : object) {
    const Eigen::Vector3d pc = origin + R_buff2camera * Eigen::Vector3d(p.x, p.y, p.z);
    if (!pc.allFinite() || pc.z() <= config_.min_depth) return std::nullopt;
  }
  const auto to_world = [&](const Eigen::Vector3d & p) -> Eigen::Vector3d {
    return q_gimbal2world_ * (camera_.R_camera2gimbal * p + camera_.t_camera2gimbal);
  };
  // 将去畸变后的图像射线与 x=0 的符面求交，分别保留
  // 几何原点、检测到的 R 标中心和靶心。
  const auto intersect = [&](cv::Point2f pixel) -> std::optional<Eigen::Vector3d> {
    std::vector<cv::Point2f> normalized;
    cv::undistortPoints(std::vector<cv::Point2f>{pixel}, normalized,
                       camera_.camera_matrix, camera_.distort_coeffs);
    const Eigen::Vector3d ray(normalized[0].x, normalized[0].y, 1);
    const Eigen::Vector3d normal = R_buff2camera.col(0);
    const double denominator = normal.dot(ray);
    if (std::abs(denominator) < 1e-6) return std::nullopt;
    const double depth = normal.dot(origin) / denominator;
    if (!(depth > config_.min_depth && depth < config_.max_depth))
      return std::nullopt;
    return to_world(depth * ray);
  };
  const auto detected_r = intersect(icon.center);
  const auto target = intersect(bull.center);
  if (!detected_r || !target) return std::nullopt;
  const Eigen::Matrix3d R_world = q_gimbal2world_.toRotationMatrix() *
    camera_.R_camera2gimbal * R_buff2camera;
  const Eigen::Vector3d blade = to_world(origin +
    R_buff2camera * Eigen::Vector3d(0, 0, config_.target_radius));
  SPRuneObservation result{tools::xyz2ypd(to_world(origin)),
    tools::eulers(R_world, 2, 1, 0), tools::xyz2ypd(blade)};
  result.detected_r_in_world = *detected_r;
  result.target_in_world = *target;
  result.rotation_in_world = R_world;
  const auto v = bull.center - icon.center;
  if (cv::norm(v) < 1e-3) return std::nullopt;
  double angle = std::atan2(-v.y, v.x);
  if (angle < 0) angle += 2 * CV_PI;
  result.slot_id = static_cast<int>(std::round(angle / (kSPSlotAngle))) % 5;
  result.score = bull.score;
  if (!result.ypd_in_world.allFinite() || !result.ypr_in_world.allFinite() ||
      !result.blade_ypd_in_world.allFinite()) return std::nullopt;
  return result;
}

std::optional<cv::Point2f> SPRuneSolver::project(const Eigen::Vector3d & world) const
{
  if (!attitude_valid_) return std::nullopt;
  const Eigen::Vector3d point = camera_.R_camera2gimbal.transpose() *
    (q_gimbal2world_.conjugate() * world - camera_.t_camera2gimbal);
  if (!point.allFinite() || point.z() <= 0.1) return std::nullopt;
  std::vector<cv::Point2f> pixels;
  cv::projectPoints(
    std::vector<cv::Point3f>{{static_cast<float>(point.x()), static_cast<float>(point.y()),
                            static_cast<float>(point.z())}},
    cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_.camera_matrix, camera_.distort_coeffs, pixels);
  return pixels.front();
}
}  // namespace auto_buff_v2
