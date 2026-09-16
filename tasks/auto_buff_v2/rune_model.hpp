#ifndef AUTO_BUFF_V2__RUNE_MODEL_HPP
#define AUTO_BUFF_V2__RUNE_MODEL_HPP

#include <array>
#include <chrono>
#include <optional>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "rune.hpp"
#include "rune_energy_fitter.hpp"

namespace auto_buff_v2
{
using Timestamp = std::chrono::steady_clock::time_point;

struct RuneState
{
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Timestamp start_timestamp{};
  Timestamp timestamp{};
  double rotation_speed = 0;
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

  void transition(double seconds);
  std::optional<Eigen::Vector3d> aimpoint() const;
};

class RuneModel
{
public:
  RuneModel(const std::string & config_path, bool big_rune);
  void update_transform(const Eigen::Quaterniond & q_gimbal2world);
  bool update(const RuneElements & elements, Timestamp timestamp);
  void reset();
  std::optional<RuneState> state() const;

private:
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_gimbal2world_ = Eigen::Quaterniond::Identity();
  bool big_rune_ = false;
  std::optional<RuneState> state_;
  RuneEnergyFitter fitter_;
  Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
  std::array<Timestamp, 5> inactive_timeout_{};
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_MODEL_HPP
