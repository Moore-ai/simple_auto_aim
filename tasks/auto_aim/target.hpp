#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <vector>

#include "armor.hpp"
#include "observation_geometry.hpp"
#include "target_estimator.hpp"

namespace auto_aim
{

class Solver;

struct ReprojectionMeasurement
{
  int armor_id = 0;
  bool right = false;
  Lightbar lightbar;
};

struct ReprojectionArmorMeasurement
{
  int armor_id = 0;
  Armor armor;
};

// 过程噪声参数（所有滤波器共用）
struct ProcessNoiseConfig
{
  double accel_var = 100;
  double angular_accel_var = 400;
  double outpost_accel_var = 10;
  double outpost_angular_accel_var = 0.1;
};

// EKF 观测噪声（ypd 空间）
struct EKFObservationConfig
{
  double yaw_var = 4e-3;
  double pitch_var = 4e-3;
  double armor_yaw_base = 9e-2;
};

// InEKF 观测噪声（xyz 空间）
struct InEKFObservationConfig
{
  double xy_var = 0.0036;
  double z_var = 0.0064;
  double yaw_var = 0.0144;
  double dist_scale_denom = 25.0;  // 距离自适应分母 d²/dist_scale_denom，d=sqrt(denom) 时 scale=2
};

// 滤波器配置聚合
struct VelClampConfig
{
  bool enable = false;
  double max_linear_speed = 5.0;  // m/s
  double max_yaw_rate = 10.0;     // rad/s
};

// UKF 观测噪声（xyz 空间，与 InEKF 共用结构）
struct UKFObservationConfig
{
  double xy_var = 0.0036;
  double z_var = 0.0064;
  double yaw_var = 0.0144;
  double dist_scale_denom = 25.0;
  double sigma_alpha = 0.001;  // UKF sigma point spread
  double sigma_beta = 2.0;     // UKF prior knowledge (2.0 = Gaussian)
  double sigma_kappa = 0.0;    // UKF secondary scaling
};

struct FilterConfig
{
  ProcessNoiseConfig process_noise;
  EKFObservationConfig ekf;
  InEKFObservationConfig inekf;
  UKFObservationConfig ukf;
  VelClampConfig vel_clamp;
};

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(const Target & other);
  Target & operator=(const Target & other);
  Target(Target &&) = default;
  Target & operator=(Target &&) = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, const FilterConfig & filter_config = {},
    FilterMethod filter_method = FilterMethod::EKF, bool reprojection_mode = false,
    ReprojectionObservationConfig reprojection_config = {});
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  void update(const Armor & armor);
  bool update_reprojection(
    const std::vector<ReprojectionMeasurement> & measurements, const Solver & solver,
    const ReprojectionObservationConfig & observation_config);
  bool update_reprojection(
    const std::vector<ReprojectionMeasurement> & measurements,
    const std::vector<ReprojectionArmorMeasurement> & armor_measurements, const Solver & solver,
    const ReprojectionObservationConfig & observation_config);
  bool update_lightbar_assist(
    const std::vector<ReprojectionMeasurement> & measurements, const Solver & solver,
    const ReprojectionObservationConfig & observation_config);

  TargetState state() const;
  Eigen::VectorXd state_vector() const;
  Eigen::VectorXd ekf_x() const;
  double last_nis() const;
  const TargetEstimatorDiagnostics & diagnostics() const;
  bool has_bad_nis_convergence(double failure_rate = 0.4) const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  void mark_armor_id(int id);
  bool all_armor_ids_seen() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  int armor_num_;
  int switch_count_;
  int update_count_;
  std::vector<bool> seen_armor_ids_;

  FilterConfig config_;

  bool is_switch_, is_converged_;

  TargetEstimator estimator_;
  FilterMethod filter_method_;
  bool reprojection_mode_ = false;
  ReprojectionObservationConfig reprojection_config_;
  std::chrono::steady_clock::time_point t_;

  void update_filter(const Armor & armor, int id);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const TargetState & state, int id) const;
  Eigen::Matrix<double, 3, TargetState::dimension> h_armor_xyz_jacobian(
    const TargetState & state, int id) const;
  void constrain_reprojection_state();
  bool update_reprojection_impl(
    const std::vector<ReprojectionMeasurement> & measurements,
    const std::vector<ReprojectionArmorMeasurement> & armor_measurements, const Solver & solver,
    const ReprojectionObservationConfig & observation_config, bool auxiliary_only);
  Eigen::MatrixXd h_jacobian(const TargetState & state, int id) const;
  Eigen::MatrixXd h_jacobian_xyza(const TargetState & state, int id) const;
  void constrain_velocity();
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
