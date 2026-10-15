#ifndef AUTO_BUFF_V2__NJUST_RUNE_MODEL_HPP
#define AUTO_BUFF_V2__NJUST_RUNE_MODEL_HPP

#include "../buff_config.hpp"
#include "rune_model.hpp"
#include "njust_ekf_state.hpp"
#include "njust_rune_config.hpp"

namespace auto_buff_v2
{
class NjustRuneModel final : public RuneModel
{
public:
  using Config = NjustRuneConfig;

  NjustRuneModel(BuffConfig::Camera camera, Config config, bool big_rune);
  void set_q_gimbal2world(const Eigen::Quaterniond & q_gimbal2world) override;
  bool update(const RuneElements & elements, Timestamp timestamp) override;
  void reset() override;
  std::optional<RuneEstimate> state() const override;
  const std::string & diagnostic() const override { return diagnostic_; }
  std::vector<RuneReprojectedFeature> reprojected_features() const override;
  std::optional<cv::Point2f> reprojected_center() const override;

private:
  bool initialize(const RuneElements & elements, Timestamp timestamp,
                  const Eigen::Matrix3d & R_camera2world,
                  const Eigen::Vector3d & t_camera2world);
  void update_motion_fit(RuneEstimate & state, double elapsed_seconds);
  bool diverged() const;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_gimbal2world_ = Eigen::Quaterniond::Identity();
  bool big_rune_ = false;
  Config config_;
  std::string diagnostic_;
  std::optional<RuneEstimate> state_;
  NjustEnergyFitter fitter_;
  Eigen::Matrix<double, 6, 6> covariance_ = Eigen::Matrix<double, 6, 6>::Identity();
  NjustEkfState ekf_state_;
  std::array<Timestamp, 5> inactive_timeout_{};
  Timestamp last_inactive_corrected_{};
  Timestamp force_sine_until_{};
};
}  // namespace auto_buff_v2

#endif
