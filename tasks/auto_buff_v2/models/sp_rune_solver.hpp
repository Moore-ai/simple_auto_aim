#ifndef AUTO_BUFF_V2__SP_RUNE_SOLVER_HPP
#define AUTO_BUFF_V2__SP_RUNE_SOLVER_HPP

#include <optional>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "../buff_config.hpp"
#include "../rune.hpp"
#include "sp_rune_config.hpp"

namespace auto_buff_v2
{
// 供移植的球坐标 EKF 使用的 PnP 观测。
struct SPRuneObservation
{
  Eigen::Vector3d ypd_in_world;
  Eigen::Vector3d ypr_in_world;
  Eigen::Vector3d blade_ypd_in_world;
  Eigen::Vector3d detected_r_in_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d target_in_world = Eigen::Vector3d::Zero();
  Eigen::Matrix3d rotation_in_world = Eigen::Matrix3d::Identity();
  int slot_id = -1;
  double score = 0;

};

class SPRuneSolver
{
public:
  explicit SPRuneSolver(BuffConfig::Camera camera, SPRuneConfig config = {});
  void set_q_gimbal2world(const Eigen::Quaterniond & q);
  std::optional<SPRuneObservation> solve(const RuneElements & elements) const;
  std::vector<SPRuneObservation> solve_all(const RuneElements & elements) const;
  std::optional<cv::Point2f> project(const Eigen::Vector3d & world) const;

private:
  std::optional<SPRuneObservation> solve_one(
    const RuneBullseye & bull, const RuneIcon & icon) const;
  BuffConfig::Camera camera_;
  SPRuneConfig config_;
  bool attitude_valid_ = true;
  Eigen::Quaterniond q_gimbal2world_ = Eigen::Quaterniond::Identity();
};
}  // namespace auto_buff_v2

#endif
