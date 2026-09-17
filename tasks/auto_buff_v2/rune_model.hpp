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
  double prediction_speed = 0;
  bool use_prediction_speed = false;
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
  std::optional<Eigen::Vector3d> aimpoint_at(Timestamp prediction_time) const;
};

class RuneModel
{
public:
  struct Config
  {
    double timeout_seconds = 1.5;
    double noise_x = 1e-5;
    double noise_y = 1e-5;
    double noise_z = 1e-5;
    double noise_rotation_speed = 1;
    double noise_rotation_angle = 1e-3;
    double noise_face_yaw = 1e-5;
    double noise_observation = 20;
    double gate_threshold = 13.816;
    double init_seed_mean_error = 10;
    double init_seed_max_error = 20;
    double init_center_gate = 30;
    double init_pitch_bound = 20;
    double diverge_face_angle = 45;
  };

  RuneModel(const std::string & config_path, bool big_rune);
  void update_transform(const Eigen::Quaterniond & q_gimbal2world);
  bool update(const RuneElements & elements, Timestamp timestamp);
  void reset();
  std::optional<RuneState> state() const;

private:
  bool initialize(const RuneElements & elements, Timestamp timestamp,
                  const Eigen::Matrix3d & R_camera2world,
                  const Eigen::Vector3d & t_camera2world);
  void update_motion_fit(RuneState & state, double elapsed_seconds);
  bool diverged() const;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_gimbal2world_ = Eigen::Quaterniond::Identity();
  bool big_rune_ = false;
  Config config_;
  std::optional<RuneState> state_;
  RuneEnergyFitter fitter_;
  Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> ekf_state_ = Eigen::Matrix<double, 6, 1>::Zero();
  std::array<Timestamp, 5> inactive_timeout_{};
  Timestamp last_inactive_corrected_{};
  Timestamp force_sine_until_{};
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_MODEL_HPP
