#ifndef AUTO_AIM__SOLVER_HPP
#define AUTO_AIM__SOLVER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

#include <vector>
#include <optional>

#include "armor.hpp"

namespace auto_aim
{

struct ArmorProjection
{
  std::vector<cv::Point2f> points;
  // Each matrix differentiates one projected corner by [center_x, center_y, center_z, yaw].
  std::vector<Eigen::Matrix<double, 2, 4>> point_jacobian;
  double light_depth_diff = 0.0;
  Eigen::RowVector4d light_depth_diff_jacobian = Eigen::RowVector4d::Zero();
  bool valid = false;
};

class Solver
{
public:
  explicit Solver(const std::string & config_path);

  Eigen::Matrix3d R_gimbal2world() const;

  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  bool solve(Armor & armor) const;

  std::vector<cv::Point2f> reproject_armor(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  ArmorProjection project_armor_with_jacobian(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const;

  std::optional<double> armor_lights_depth_diff(const Armor & armor) const;

  double armor_visibility_score(
    const Eigen::Vector3d & xyz_in_world, double yaw, ArmorName name) const;

  double oupost_reprojection_error(Armor armor, const double & picth);

  std::vector<cv::Point2f> world2pixel(const std::vector<cv::Point3f> & worldPoints);

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_;
  Eigen::Matrix3d R_camera2gimbal_;
  Eigen::Vector3d t_camera2gimbal_;
  Eigen::Matrix3d R_gimbal2world_;

  std::string cost_fn_{"l1"};

  void optimize_yaw(Armor & armor) const;

  double armor_reprojection_error(const Armor & armor, double yaw, const double & inclined) const;
  double SJTU_cost(
    const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
    const double & inclined) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__SOLVER_HPP
