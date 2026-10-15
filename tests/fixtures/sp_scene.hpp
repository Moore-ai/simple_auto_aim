#pragma once
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "tasks/auto_buff_v2/rune.hpp"
#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tools/math_tools.hpp"
namespace sp_test
{
using namespace auto_buff_v2;
inline BuffConfig::Camera camera()
{
  BuffConfig::Camera c;
  c.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 640, 0, 1000, 480, 0, 0, 1);
  c.distort_coeffs = (cv::Mat_<double>(1, 5) << -0.08, 0.01, 0.001, -0.002, 0);
  c.R_camera2gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  c.t_camera2gimbal = {0.05, -0.02, 0.03};
  return c;
}
inline RuneElements scene(const BuffConfig::Camera & c,
  const Eigen::Quaterniond & q,
  Eigen::Vector3d center,
  double yaw,
  double roll,
  bool two = false,
  double radius = .7,
  double r_center_z = 0)
{
  RuneElements e;
  const Eigen::Matrix3d Rc = c.R_camera2gimbal.transpose() * q.conjugate().toRotationMatrix();
  const Eigen::Vector3d tc = c.R_camera2gimbal.transpose() *
    (q.conjugate() * center - c.t_camera2gimbal);
  for (int b = 0; b < (two ? 2 : 1); ++b) {
    const Eigen::Matrix3d R = Rc * tools::rotation_matrix({yaw, 0, roll + b * 2 * M_PI / 5});
    cv::Mat rm, rv; cv::eigen2cv(R, rm); cv::Rodrigues(rm, rv);
    std::vector<cv::Point2f> px;
    cv::projectPoints(std::vector<cv::Point3f>{{0, 0, static_cast<float>(radius + .115)},
      {0, .150f, static_cast<float>(radius)}, {0, 0, static_cast<float>(radius - .115)},
      {0, -.115f, static_cast<float>(radius)}, {0, 0, static_cast<float>(radius)},
      {0, 0, static_cast<float>(r_center_z)}}, rv,
      cv::Vec3d(tc.x(), tc.y(), tc.z()), c.camera_matrix, c.distort_coeffs, px);
    if (b == 0) e.icons.push_back({px[5], 1});
    e.bullseyes.push_back({px[4], {px[2], px[0], px[3], px[1]}, false, 1.0 - b * .1});
  }
  return e;
}
}
