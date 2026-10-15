#ifndef AUTO_BUFF_V2__RUNE_MODEL_HPP
#define AUTO_BUFF_V2__RUNE_MODEL_HPP

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "../rune.hpp"

namespace auto_buff_v2
{
enum class RunePredictionMethod { weighted_least_squares, ekf };

using Timestamp = std::chrono::steady_clock::time_point;

struct RuneEstimate
{
  RunePredictionMethod prediction_method = RunePredictionMethod::weighted_least_squares;
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Timestamp start_timestamp{};
  Timestamp timestamp{};
  double rotation_speed = 0;
  double fitted_rotation_speed = 0;
  bool has_fitted_motion = false;
  double motion_fit_cost = 0;
  double rotation_angle = 0;
  double face_yaw = 0;
  double target_radius = kRuneGlobalRadius;
  int slot_id = -1;  // 图像平面槽位；rotation_angle 表示该槽位的物理自转角。
  int blade_id = -1;  // 持久物理编号，与 slot_id 无关。
  double center_yaw_speed = 0;  // 中心方位角速度，单位 rad/s；位姿锚定时为零。
  std::array<bool, 5> inactive{};
  double sine_v = 0;
  double sine_a = 0;
  double sine_omega = 0;
  double sine_phase = 0;
  double sine_t = 0;
  bool sine_valid = false;
  bool requires_warmup = true;  // Njust 运动拟合需要历史数据；SP 可以立即瞄准。
  std::size_t update_count = 0;
  bool predicted = false;  // 当前帧没有未激活扇叶参与校正。
};

struct RuneReprojectedFeature
{
  int id = 0;
  cv::Point2f point;
};

class RuneModel
{
public:
  virtual ~RuneModel() = default;
  virtual void set_q_gimbal2world(const Eigen::Quaterniond & q_gimbal2world) = 0;
  virtual bool update(const RuneElements & elements, Timestamp timestamp) = 0;
  virtual void reset() = 0;
  virtual std::optional<RuneEstimate> state() const = 0;
  virtual std::vector<RuneEstimate> candidates() const
  {
    const auto estimate = state();
    return estimate ? std::vector<RuneEstimate>{*estimate} : std::vector<RuneEstimate>{};
  }
  virtual const std::string & diagnostic() const = 0;
  virtual std::vector<RuneReprojectedFeature> reprojected_features() const = 0;
  virtual std::optional<cv::Point2f> reprojected_center() const = 0;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_MODEL_HPP
