#ifndef AUTO_BUFF_V2__RUNE_MODEL_HPP
#define AUTO_BUFF_V2__RUNE_MODEL_HPP

#include <array>
#include <chrono>
#include <optional>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "buff_config.hpp"
#include "rune.hpp"
#include "rune_energy_fitter.hpp"

namespace auto_buff_v2
{
using Timestamp = std::chrono::steady_clock::time_point;

struct RuneEstimate
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Timestamp start_timestamp{};
  Timestamp timestamp{};
  double rotation_speed = 0;
  double fitted_rotation_speed = 0;
  bool has_fitted_motion = false;
  double motion_fit_cost = 0;
  double rotation_angle = 0;
  double face_yaw = 0;
  std::array<bool, 5> inactive{};
  double sine_v = 0;
  double sine_a = 0;
  double sine_omega = 0;
  double sine_phase = 0;
  double sine_t = 0;
  bool sine_valid = false;
  std::size_t update_count = 0;
};

struct RuneReprojectedFeature
{
  int id = 0;
  cv::Point2f point;
};

class RuneModel
{
public:
  using Config = BuffConfig::Model;

  RuneModel(BuffConfig::Camera camera, Config config, bool big_rune);
  void update_transform(const Eigen::Quaterniond & q_gimbal2world);
  bool update(const RuneElements & elements, Timestamp timestamp);
  void reset();
  std::optional<RuneEstimate> state() const;
  std::vector<RuneReprojectedFeature> reprojected_features() const;
  std::optional<cv::Point2f> reprojected_center() const;

private:
  bool initialize(const RuneElements & elements, Timestamp timestamp,
                  const Eigen::Matrix3d & R_camera2world,
                  const Eigen::Vector3d & t_camera2world);
  void update_motion_fit(RuneEstimate & state, double elapsed_seconds);
  bool diverged() const;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_gimbal2world_ = Eigen::Quaterniond::Identity();
  bool big_rune_ = false;
  Config config_;
  std::optional<RuneEstimate> state_;
  RuneEnergyFitter fitter_;
  Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> ekf_state_ = Eigen::Matrix<double, 6, 1>::Zero();
  std::array<Timestamp, 5> inactive_timeout_{};
  Timestamp last_inactive_corrected_{};
  Timestamp force_sine_until_{};
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_MODEL_HPP
