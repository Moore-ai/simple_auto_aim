#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "tools/camera2gimbal_extrinsic.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
constexpr double LIGHTBAR_LENGTH = 56e-3;     // m
constexpr double BIG_ARMOR_WIDTH = 230e-3;    // m
constexpr double SMALL_ARMOR_WIDTH = 135e-3;  // m

const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

namespace
{
Eigen::Matrix3d armor_rotation(double yaw, double pitch)
{
  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_pitch = std::sin(pitch);
  const auto cos_pitch = std::cos(pitch);
  Eigen::Matrix3d result;
  result << cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
    sin_yaw * cos_pitch, cos_yaw, sin_yaw * sin_pitch,
    -sin_pitch, 0.0, cos_pitch;
  return result;
}

Eigen::Matrix3d armor_rotation_derivative(double yaw, double pitch)
{
  const auto sin_yaw = std::sin(yaw);
  const auto cos_yaw = std::cos(yaw);
  const auto sin_pitch = std::sin(pitch);
  const auto cos_pitch = std::cos(pitch);
  Eigen::Matrix3d result;
  result << -sin_yaw * cos_pitch, -cos_yaw, -sin_yaw * sin_pitch,
    cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch,
    0.0, 0.0, 0.0;
  return result;
}

const std::vector<cv::Point3f> & armor_points(ArmorType type)
{
  return type == ArmorType::big ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
}

bool project_distorted(
  const Eigen::Vector3d & point, const cv::Mat & camera_matrix, const cv::Mat & distort_coeffs,
  cv::Point2f & pixel, Eigen::Matrix<double, 2, 3> & jacobian)
{
  if (!point.allFinite() || point.z() <= 1e-9) return false;

  const auto fx = camera_matrix.at<double>(0, 0);
  const auto fy = camera_matrix.at<double>(1, 1);
  const auto cx = camera_matrix.at<double>(0, 2);
  const auto cy = camera_matrix.at<double>(1, 2);
  const auto k1 = distort_coeffs.at<double>(0, 0);
  const auto k2 = distort_coeffs.at<double>(0, 1);
  const auto p1 = distort_coeffs.at<double>(0, 2);
  const auto p2 = distort_coeffs.at<double>(0, 3);
  const auto k3 = distort_coeffs.at<double>(0, 4);

  const auto x = point.x() / point.z();
  const auto y = point.y() / point.z();
  const auto r2 = x * x + y * y;
  const auto r4 = r2 * r2;
  const auto r6 = r4 * r2;
  const auto radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const auto x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const auto y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

  pixel = {static_cast<float>(fx * x_distorted + cx), static_cast<float>(fy * y_distorted + cy)};

  const auto radial_dx = 2.0 * x * (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4);
  const auto radial_dy = 2.0 * y * (k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4);
  Eigen::Matrix2d distorted_jacobian;
  distorted_jacobian <<
    radial + x * radial_dx + 2.0 * p1 * y + 6.0 * p2 * x,
    x * radial_dy + 2.0 * p1 * x + 2.0 * p2 * y,
    y * radial_dx + 2.0 * p1 * y + 2.0 * p2 * x,
    radial + y * radial_dy + 6.0 * p1 * y + 2.0 * p2 * x;

  Eigen::Matrix<double, 2, 3> normalized_jacobian;
  normalized_jacobian <<
    1.0 / point.z(), 0.0, -x / point.z(),
    0.0, 1.0 / point.z(), -y / point.z();
  jacobian =
    (Eigen::DiagonalMatrix<double, 2>(fx, fy) * distorted_jacobian * normalized_jacobian).eval();
  return pixel.x == pixel.x && pixel.y == pixel.y && jacobian.allFinite();
}
}  // namespace

Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  const auto extrinsic = tools::load_camera2gimbal_extrinsic(yaml);
  R_camera2gimbal_ = extrinsic.rotation;
  t_camera2gimbal_ = extrinsic.translation;

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  if (yaml["cost_fn"]) {
    cost_fn_ = yaml["cost_fn"].as<std::string>();
  }
}

Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

//solvePnP（获得姿态）
bool Solver::solve(Armor & armor) const
{
  if (armor.points.size() != 4) return false;
  const auto & object_points = armor_points(armor.type);

  cv::Vec3d rvec, tvec;
  if (!cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE)) return false;

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 平衡不做yaw优化，因为pitch假设不成立
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return true;

  optimize_yaw(armor);
  return armor.xyz_in_world.allFinite() && armor.ypr_in_world.allFinite();
}

bool Solver::refine_yaw_with_prediction(
  Armor & armor, double predicted_yaw, int armor_num) const
{
  if (armor.points.size() != 4 || armor_num < 3 || !armor.xyz_in_world.allFinite()) return false;

  const auto gimbal_yaw = tools::eulers(R_gimbal2world_, 2, 1, 0)[0];
  const auto angle_between_armors = 2.0 * CV_PI / armor_num;
  constexpr double must_see_angle = CV_PI / 4.0;
  constexpr double must_not_see_angle = CV_PI / 2.0;
  double left = std::max(-must_not_see_angle, must_see_angle - angle_between_armors);
  double right = std::min(must_not_see_angle, -must_see_angle + angle_between_armors);
  if (left >= right) return false;

  const auto inclined = tools::limit_rad(predicted_yaw - gimbal_yaw);
  constexpr int iterations = 12;
  const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
  double middle_left = left + (right - left) * (1.0 - phi);
  double middle_right = left + (right - left) * phi;
  auto cost = [&](double relative_yaw) {
    const auto projected = reproject_armor(
      armor.xyz_in_world, tools::limit_rad(relative_yaw + gimbal_yaw), armor.type, armor.name);
    return projected.size() == armor.points.size() ? SJTU_cost(projected, armor.points, inclined) :
                                                     std::numeric_limits<double>::infinity();
  };
  double middle_left_cost = cost(middle_left);
  double middle_right_cost = cost(middle_right);
  for (int i = 0; i < iterations; ++i) {
    if (middle_left_cost < middle_right_cost) {
      right = middle_right;
      middle_right = middle_left;
      middle_right_cost = middle_left_cost;
      middle_left = left + (right - left) * (1.0 - phi);
      middle_left_cost = cost(middle_left);
    } else {
      left = middle_left;
      middle_left = middle_right;
      middle_left_cost = middle_right_cost;
      middle_right = left + (right - left) * phi;
      middle_right_cost = cost(middle_right);
    }
  }
  const auto refined_yaw = tools::limit_rad((left + right) * 0.5 + gimbal_yaw);
  if (!std::isfinite(refined_yaw)) return false;
  armor.ypr_in_world[0] = refined_yaw;
  return true;
}

bool Solver::optimize_outpost_distance(Armor & armor, const std::list<Lightbar> & lightbars) const
{
  if (armor.name != ArmorName::outpost || armor.type != ArmorType::small ||
      armor.points.size() != 4) {
    return false;
  }

  if (!armor.xyz_in_gimbal.allFinite() || !armor.ypr_in_gimbal.allFinite()) return false;
  const Eigen::Matrix3d initial_rotation =
    R_camera2gimbal_.transpose() * tools::rotation_matrix(armor.ypr_in_gimbal);
  const Eigen::Vector3d initial_translation =
    R_camera2gimbal_.transpose() * (armor.xyz_in_gimbal - t_camera2gimbal_);
  cv::Mat initial_rotation_cv;
  cv::eigen2cv(initial_rotation, initial_rotation_cv);
  cv::Vec3d initial_rvec;
  cv::Rodrigues(initial_rotation_cv, initial_rvec);
  const cv::Vec3d initial_tvec{
    initial_translation.x(), initial_translation.y(), initial_translation.z()};

  struct PredictedNeighbor
  {
    std::array<cv::Point3f, 2> object_points;
    std::vector<cv::Point2f> image_points;
  };
  std::vector<PredictedNeighbor> predicted_neighbors;
  constexpr double height_step = 0.102;
  const auto armor_to_center =
    Eigen::AngleAxisd(-OUTPOST_MOUNT_PITCH, Eigen::Vector3d::UnitY()) * Eigen::Vector3d::UnitX();
  const auto vertical = Eigen::AngleAxisd(-CV_PI / 2.0, Eigen::Vector3d::UnitY()) * armor_to_center;
  const auto rotation_center = OUTPOST_RADIUS * armor_to_center;
  const std::array<std::array<double, 2>, 4> relations{{
    {2.0 * height_step, 2.0 * CV_PI / 3.0},
    {-height_step, 2.0 * CV_PI / 3.0},
    {height_step, -2.0 * CV_PI / 3.0},
    {-2.0 * height_step, -2.0 * CV_PI / 3.0}}};
  for (const auto & relation : relations) {
    const Eigen::AngleAxisd rotate_to_neighbor(relation[1], vertical);
    const auto neighbor_center =
      rotation_center + rotate_to_neighbor * (-armor_to_center * OUTPOST_RADIUS) +
      vertical * relation[0];
    const auto x_axis = (rotate_to_neighbor * Eigen::Vector3d::UnitX()).normalized();
    const auto y_axis = vertical.cross(x_axis).normalized();
    const auto z_axis = x_axis.cross(y_axis).normalized();
    const Eigen::Matrix3d rotation = (Eigen::Matrix3d() << x_axis, y_axis, z_axis).finished();

    std::array<Eigen::Vector3d, 4> endpoints{
      neighbor_center +
        rotation * Eigen::Vector3d{0.0, SMALL_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0},
      neighbor_center +
        rotation * Eigen::Vector3d{0.0, SMALL_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0},
      neighbor_center +
        rotation * Eigen::Vector3d{0.0, -SMALL_ARMOR_WIDTH / 2.0, LIGHTBAR_LENGTH / 2.0},
      neighbor_center +
        rotation * Eigen::Vector3d{0.0, -SMALL_ARMOR_WIDTH / 2.0, -LIGHTBAR_LENGTH / 2.0}};
    const auto use_first_side = endpoints[0].norm() < endpoints[2].norm();
    const auto offset = use_first_side ? 0U : 2U;
    PredictedNeighbor neighbor;
    for (std::size_t index = 0; index < 2; ++index) {
      const auto & point = endpoints[offset + index];
      neighbor.object_points[index] = {
        static_cast<float>(point.x()), static_cast<float>(point.y()),
        static_cast<float>(point.z())};
    }
    cv::projectPoints(
      std::vector<cv::Point3f>{neighbor.object_points[0], neighbor.object_points[1]}, initial_rvec,
      initial_tvec, camera_matrix_, distort_coeffs_, neighbor.image_points);
    predicted_neighbors.push_back(neighbor);
  }

  const PredictedNeighbor * best_neighbor = nullptr;
  std::array<cv::Point2f, 2> best_image_points;
  double best_error = std::numeric_limits<double>::infinity();
  for (const auto & lightbar : lightbars) {
    if (lightbar.color != armor.color || lightbar.top == lightbar.bottom) continue;
    for (const auto & neighbor : predicted_neighbors) {
      const auto expected_length = cv::norm(neighbor.image_points[1] - neighbor.image_points[0]);
      const auto observed_length = cv::norm(lightbar.bottom - lightbar.top);
      if (expected_length <= 1e-6 ||
          std::abs(observed_length / expected_length - 1.0) > 0.2) {
        continue;
      }
      const std::array<std::array<cv::Point2f, 2>, 2> orientations{{
        {lightbar.top, lightbar.bottom}, {lightbar.bottom, lightbar.top}}};
      for (const auto & observed : orientations) {
        const auto error = cv::norm(observed[0] - neighbor.image_points[0]) +
          cv::norm(observed[1] - neighbor.image_points[1]);
        if (error < best_error && error < expected_length * 5.0) {
          best_error = error;
          best_neighbor = &neighbor;
          best_image_points = observed;
        }
      }
    }
  }
  if (!best_neighbor) return false;

  std::vector<cv::Point3f> object_points = SMALL_ARMOR_POINTS;
  object_points.insert(
    object_points.end(), best_neighbor->object_points.begin(), best_neighbor->object_points.end());
  std::vector<cv::Point2f> image_points = armor.points;
  image_points.insert(image_points.end(), best_image_points.begin(), best_image_points.end());
  cv::Vec3d optimized_rvec = initial_rvec;
  cv::Vec3d optimized_tvec = initial_tvec;
  if (!cv::solvePnP(
        object_points, image_points, camera_matrix_, distort_coeffs_, optimized_rvec,
        optimized_tvec, true, cv::SOLVEPNP_ITERATIVE) ||
      optimized_tvec[2] <= 0.0) {
    return false;
  }

  Armor optimized = armor;
  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(optimized_tvec, xyz_in_camera);
  optimized.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  optimized.xyz_in_world = R_gimbal2world_ * optimized.xyz_in_gimbal;
  cv::Mat rotation_cv;
  cv::Rodrigues(optimized_rvec, rotation_cv);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rotation_cv, R_armor2camera);
  const auto R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  const auto R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  optimized.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  optimized.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);
  optimized.ypd_in_world = tools::xyz2ypd(optimized.xyz_in_world);
  optimize_yaw(optimized);
  if (!optimized.xyz_in_world.allFinite() || !optimized.ypr_in_world.allFinite()) return false;
  armor = optimized;
  return true;
}

std::optional<double> Solver::armor_lights_depth_diff(const Armor & armor) const
{
  if (armor.points.size() != 4) return std::nullopt;
  const auto & object_points = armor_points(armor.type);
  cv::Vec3d rvec, tvec;
  if (!cv::solvePnP(
      object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
      cv::SOLVEPNP_IPPE)) {
    return std::nullopt;
  }

  cv::Mat rotation_cv;
  cv::Rodrigues(rvec, rotation_cv);
  Eigen::Matrix3d rotation;
  cv::cv2eigen(rotation_cv, rotation);
  const Eigen::Vector3d translation(tvec[0], tvec[1], tvec[2]);
  const auto point_in_camera = [&](std::size_t index) {
    const auto & point = object_points[index];
    return rotation * Eigen::Vector3d(point.x, point.y, point.z) + translation;
  };
  const auto left_center = (point_in_camera(0) + point_in_camera(3)) * 0.5;
  const auto right_center = (point_in_camera(1) + point_in_camera(2)) * 0.5;
  const auto result = left_center.z() - right_center.z();
  return std::isfinite(result) ? std::optional<double>(result) : std::nullopt;
}

double Solver::armor_visibility_score(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorName name) const
{
  return armor_visibility_score(xyz_in_world, yaw, armor_mount_pitch(name));
}

double Solver::armor_visibility_score(
  const Eigen::Vector3d & xyz_in_world, double yaw, double pitch) const
{
  const Eigen::Matrix3d R_armor2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() *
    armor_rotation(yaw, pitch);
  const Eigen::Vector3d t_armor2camera = R_camera2gimbal_.transpose() *
    (R_gimbal2world_.transpose() * xyz_in_world - t_camera2gimbal_);
  const Eigen::Vector3d front_normal = -R_armor2camera.col(0);
  return front_normal.dot(-t_armor2camera);
}

ArmorProjection Solver::project_armor_with_jacobian(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  return project_armor_with_jacobian(xyz_in_world, yaw, armor_mount_pitch(name), type);
}

ArmorProjection Solver::project_armor_with_jacobian(
  const Eigen::Vector3d & xyz_in_world, double yaw, double pitch, ArmorType type) const
{
  ArmorProjection result;
  const auto & points = armor_points(type);
  result.points.reserve(points.size());
  result.point_jacobian.reserve(points.size());

  const auto R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  const auto t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;
  const auto R_armor2world = armor_rotation(yaw, pitch);
  const auto dR_armor2world = armor_rotation_derivative(yaw, pitch);

  std::vector<Eigen::Vector3d> camera_points;
  std::vector<Eigen::Matrix<double, 3, 4>> camera_jacobians;
  camera_points.reserve(points.size());
  camera_jacobians.reserve(points.size());

  for (const auto & object_point : points) {
    const Eigen::Vector3d object(object_point.x, object_point.y, object_point.z);
    const auto world_point = xyz_in_world + R_armor2world * object;
    const auto camera_point = R_world2camera * world_point + t_world2camera;
    Eigen::Matrix<double, 3, 4> camera_jacobian;
    camera_jacobian.leftCols<3>() = R_world2camera;
    camera_jacobian.col(3) = R_world2camera * dR_armor2world * object;
    camera_points.push_back(camera_point);
    camera_jacobians.push_back(camera_jacobian);

    cv::Point2f pixel;
    Eigen::Matrix<double, 2, 3> pixel_jacobian;
    if (!project_distorted(camera_point, camera_matrix_, distort_coeffs_, pixel, pixel_jacobian)) {
      return result;
    }
    result.points.push_back(pixel);
    result.point_jacobian.push_back(pixel_jacobian * camera_jacobian);
  }

  const auto left_center = (camera_points[0] + camera_points[3]) * 0.5;
  const auto right_center = (camera_points[1] + camera_points[2]) * 0.5;
  result.light_depth_diff = left_center.z() - right_center.z();
  result.light_depth_diff_jacobian =
    (camera_jacobians[0].row(2) + camera_jacobians[3].row(2) -
     camera_jacobians[1].row(2) - camera_jacobians[2].row(2)) * 0.5;
  result.valid = result.light_depth_diff == result.light_depth_diff &&
    result.light_depth_diff_jacobian.allFinite();
  return result;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  return project_armor_with_jacobian(xyz_in_world, yaw, type, name).points;
}

std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, double pitch, ArmorType type) const
{
  return project_armor_with_jacobian(xyz_in_world, yaw, pitch, type).points;
}

double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // solve
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

void Solver::optimize_yaw(Armor & armor) const
{
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // degree
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    // i - p 构成线段。过程：先移动起点，再补长度，再旋转
    Eigen::Vector2d ref_d = refs[p] - refs[i];  // 标准
    Eigen::Vector2d pt_d = pts[p] - pts[i];
    // 长度差代价 + 起点差代价(1 / 2)（0 度左右应该抛弃)
    double pixel_dis =  // dis 是指方差平面内到原点的距离
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    // 平方可能是为了配合 sin 和 cos
    // 弧度差代价（0 度左右占比应该大）
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;  // DETECTOR_ERROR_PIXEL_BY_SLOPE
    // 重投影像素误差越大，越相信斜率
    cost += std::sqrt(cost_i);
  }
  return cost;
}

double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  if (cost_fn_ == "l2") {
    for (int i = 0; i < 4; i++) error += std::pow(cv::norm(armor.points[i] - image_points[i]), 2);
  } else if (cost_fn_ == "l1") {
    for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  }
  // auto error = SJTU_cost(image_points, armor.points, inclined);

  return error;
}

// 世界坐标到像素坐标的转换
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim
