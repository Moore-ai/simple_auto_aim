#include "rune_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include "tools/camera2gimbal_extrinsic.hpp"
#include "hungarian.hpp"

namespace auto_buff_v2
{
void RuneState::transition(double seconds)
{
  if (sine_valid) {
    const auto old_phase = sine_phase;
    sine_t += seconds;
    sine_phase += sine_omega * seconds;
    if (std::abs(sine_omega) > 1e-12) {
      rotation_angle += sine_v * seconds +
                        sine_a / sine_omega * (std::cos(old_phase) - std::cos(sine_phase));
    } else {
      rotation_angle += rotation_speed * seconds;
    }
    rotation_speed = sine_v + sine_a * std::sin(sine_phase);
  } else {
    rotation_angle += rotation_speed * seconds;
  }
}

std::optional<Eigen::Vector3d> RuneState::aimpoint() const
{
  const auto delay = std::chrono::seconds(sine_valid ? 6 : 3);
  if (std::chrono::steady_clock::now() - start_timestamp < delay) return std::nullopt;
  constexpr double kPi = 3.14159265358979323846;
  for (std::size_t i = 0; i < inactive.size(); ++i) {
    if (!inactive[i]) continue;
    const double angle = rotation_angle + i * 2 * kPi / 5;
    const Eigen::Vector3d local(0, -kRuneGlobalRadius * std::sin(angle),
                                kRuneGlobalRadius * std::cos(angle));
    return center + Eigen::AngleAxisd(face_yaw, Eigen::Vector3d::UnitZ()) * local;
  }
  return std::nullopt;
}

RuneModel::RuneModel(const std::string & config_path, bool big_rune) : big_rune_(big_rune)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto intrinsics = yaml["camera_matrix"].as<std::vector<double>>();
  const auto distortion = yaml["distort_coeffs"].as<std::vector<double>>();
  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3], intrinsics[4], intrinsics[5],
    intrinsics[6], intrinsics[7], intrinsics[8]);
  distort_coeffs_ = cv::Mat(1, 5, CV_64F);
  for (int i = 0; i < 5; ++i) distort_coeffs_.at<double>(0, i) = distortion[i];
  const auto extrinsic = tools::load_camera2gimbal_extrinsic(yaml);
  R_camera2gimbal_ = extrinsic.rotation;
  t_camera2gimbal_ = extrinsic.translation;
  const auto body = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(body.data());
}

void RuneModel::update_transform(const Eigen::Quaterniond & q_gimbal2world)
{
  q_gimbal2world_ = q_gimbal2world;
}

void RuneModel::reset()
{
  state_.reset();
  fitter_.reset();
  inactive_timeout_.fill(Timestamp{});
}

std::optional<RuneState> RuneModel::state() const { return state_; }

namespace
{
constexpr double kPi = 3.14159265358979323846;
using Vector = Eigen::Matrix<double, 6, 1>;
using Matrix = Eigen::Matrix<double, 6, 6>;

double normalize_angle(double angle)
{
  return std::remainder(angle, 2 * kPi);
}

Vector vector_from(const RuneState & state)
{
  Vector result;
  result << state.center.x(), state.center.y(), state.center.z(), state.rotation_speed,
    state.rotation_angle, state.face_yaw;
  return result;
}

void vector_into(RuneState & state, const Vector & value)
{
  state.center = value.head<3>();
  state.rotation_speed = value[3];
  state.rotation_angle = value[4];
  state.face_yaw = normalize_angle(value[5]);
}

std::optional<cv::Point2f> project_feature(
  const Vector & value, int feature, const Eigen::Matrix3d & R_camera2world,
  const Eigen::Vector3d & t_camera2world, const cv::Mat & camera_matrix,
  const cv::Mat & distortion)
{
  Eigen::Vector3d local;
  if (feature == 0) {
    local = {-kRuneIconProminentDistance, 0, 0};
  } else {
    const double angle = value[4] + (feature - 1) * 2 * kPi / 5;
    local = {0, -kRuneGlobalRadius * std::sin(angle), kRuneGlobalRadius * std::cos(angle)};
  }
  const Eigen::Vector3d world = value.head<3>() +
                                Eigen::AngleAxisd(value[5], Eigen::Vector3d::UnitZ()) * local;
  const Eigen::Vector3d camera = R_camera2world.transpose() * (world - t_camera2world);
  if (camera.z() <= 0.1) return std::nullopt;
  std::vector<cv::Point2f> pixels;
  cv::projectPoints(std::vector<cv::Point3f>{{static_cast<float>(camera.x()),
                                             static_cast<float>(camera.y()),
                                             static_cast<float>(camera.z())}},
                    cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix, distortion, pixels);
  return pixels.front();
}

Eigen::Matrix<double, 2, 6> observation_jacobian(
  const Vector & x, int feature, const Eigen::Matrix3d & R_camera2world,
  const Eigen::Vector3d & t_camera2world, const cv::Mat & camera_matrix,
  const cv::Mat & distortion)
{
  Eigen::Matrix<double, 2, 6> H = Eigen::Matrix<double, 2, 6>::Zero();
  const auto predicted = project_feature(x, feature, R_camera2world, t_camera2world,
                                          camera_matrix, distortion);
  if (!predicted) return H;
  for (int j = 0; j < 6; ++j) {
    Vector perturb = x;
    perturb[j] += 1e-4;
    const auto p = project_feature(perturb, feature, R_camera2world, t_camera2world,
                                   camera_matrix, distortion);
    if (!p) continue;
    H(0, j) = (p->x - predicted->x) / 1e-4;
    H(1, j) = (p->y - predicted->y) / 1e-4;
  }
  return H;
}

bool solve_seed(const RuneIcon & icon, const RuneBullseye & bull, const cv::Mat & camera_matrix,
                const cv::Mat & distortion, const Eigen::Matrix3d & R_camera2world,
                const Eigen::Vector3d & t_camera2world, RuneState & state)
{
  std::size_t bottom = 0, top = 0;
  double nearest = std::numeric_limits<double>::max(), farthest = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const double distance = cv::norm(bull.corners[i] - icon.center);
    if (distance < nearest) { nearest = distance; bottom = i; }
    if (distance > farthest) { farthest = distance; top = i; }
  }
  if (bottom == top) return false;
  std::array<std::size_t, 2> sides{};
  int count = 0;
  for (std::size_t i = 0; i < 4; ++i)
    if (i != bottom && i != top) sides[count++] = i;
  if (count != 2) return false;
  const auto & b = bull.corners[bottom];
  const auto & t = bull.corners[top];
  const cv::Point2f up = t - b;
  const cv::Point2f right(up.y, -up.x);
  auto l = bull.corners[sides[0]], r = bull.corners[sides[1]];
  if ((l - bull.center).dot(right) < (r - bull.center).dot(right)) std::swap(l, r);

  const std::vector<cv::Point3f> object = {
    {-0.1f, 0, 0}, {0, 0, 0.85f}, {0, 0.15f, 0.7f},
    {0, 0, 0.55f}, {0, -0.15f, 0.7f}};
  const std::vector<cv::Point2f> image = {icon.center, t, l, b, r};
  cv::Vec3d rvec, tvec;
  try {
    if (!cv::solvePnP(object, image, camera_matrix, distortion, rvec, tvec, false,
                      cv::SOLVEPNP_EPNP) || tvec[2] <= 0) return false;
    if (!cv::solvePnP(object, image, camera_matrix, distortion, rvec, tvec, true,
                      cv::SOLVEPNP_ITERATIVE) || tvec[2] <= 0) return false;
  } catch (const cv::Exception &) { return false; }
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  Eigen::Matrix3d R_page2camera;
  cv::cv2eigen(rotation, R_page2camera);
  const Eigen::Matrix3d R_page2world = R_camera2world * R_page2camera;
  const Eigen::Vector3d face = R_page2world.col(0);
  if (std::abs(face.z()) > std::sin(20 * kPi / 180)) return false;
  state.center = R_camera2world * Eigen::Vector3d(tvec[0], tvec[1], tvec[2]) + t_camera2world;
  state.face_yaw = std::atan2(face.y(), face.x());
  const Eigen::Matrix3d local_rotation =
    Eigen::AngleAxisd(-state.face_yaw, Eigen::Vector3d::UnitZ()) * R_page2world;
  state.rotation_angle = std::atan2(-local_rotation(1, 2), local_rotation(2, 2));
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object, rvec, tvec, camera_matrix, distortion, projected);
  double squared_error = 0, max_error = 0;
  for (std::size_t i = 0; i < image.size(); ++i) {
    const double error = cv::norm(projected[i] - image[i]);
    squared_error += error * error;
    max_error = std::max(max_error, error);
  }
  return std::sqrt(squared_error / image.size()) <= 10 && max_error <= 20 &&
         state.center.allFinite();
}
}  // namespace

bool RuneModel::update(const RuneElements & elements, Timestamp timestamp)
{
  const Eigen::Matrix3d R_camera2world = q_gimbal2world_.toRotationMatrix() *
                                          R_gimbal2imubody_ * R_camera2gimbal_;
  const Eigen::Vector3d t_camera2world = q_gimbal2world_ *
                                          R_gimbal2imubody_ * t_camera2gimbal_;
  if (!state_) {
    int inactive_count = 0;
    for (const auto & bull : elements.bullseyes) {
      if (!bull.active) {
        ++inactive_count;
      }
    }
    if (elements.icons.empty() || inactive_count == 0 || inactive_count > 2) return false;
    for (const auto & icon : elements.icons) {
      for (const auto & bull : elements.bullseyes) {
        if (bull.active) continue;
        RuneState seed;
        if (!solve_seed(icon, bull, camera_matrix_, distort_coeffs_, R_camera2world,
                        t_camera2world, seed)) continue;
        seed.start_timestamp = timestamp;
        seed.timestamp = timestamp;
        seed.inactive[0] = true;
        inactive_timeout_[0] = timestamp + std::chrono::milliseconds(100);
        state_ = seed;
        covariance_.diagonal() << 64, 64, 64, 100, 25, 10;
        fitter_.reset();
        return true;
      }
    }
    return false;
  }

  auto & state = *state_;
  const double dt = std::chrono::duration<double>(timestamp - state.timestamp).count();
  if (dt < 0 || dt > 0.5) { reset(); return false; }
  state.transition(dt);
  state.timestamp = timestamp;
  Vector x = vector_from(state);
  Matrix F = Matrix::Identity();
  F(4, 3) = dt;
  Matrix Q = Matrix::Zero();
  Q.diagonal() << 1e-5, 1e-5, 1e-5, 1, 1e-3, 1e-5;
  covariance_ = F * covariance_ * F.transpose() + Q * std::max(dt, 1e-3);
  for (std::size_t i = 0; i < 5; ++i)
    if (timestamp >= inactive_timeout_[i]) state.inactive[i] = false;

  struct Observation { cv::Point2f pixel; bool icon; bool inactive; };
  std::vector<Observation> observations;
  for (const auto & icon : elements.icons) observations.push_back({icon.center, true, false});
  for (const auto & bull : elements.bullseyes)
    observations.push_back({bull.center, false, !bull.active});
  constexpr double kGate = 13.816;
  const double huge = std::numeric_limits<double>::max() / 4;
  Eigen::MatrixXd cost(observations.size(), 6);
  cost.setConstant(huge);
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const auto & observation = observations[i];
    for (int feature = observation.icon ? 0 : 1; feature <= (observation.icon ? 0 : 5); ++feature) {
      const auto predicted = project_feature(x, feature, R_camera2world, t_camera2world,
                                              camera_matrix_, distort_coeffs_);
      if (!predicted) continue;
      const Eigen::Vector2d residual(observation.pixel.x - predicted->x,
                                     observation.pixel.y - predicted->y);
      const auto H = observation_jacobian(x, feature, R_camera2world, t_camera2world,
                                           camera_matrix_, distort_coeffs_);
      const Eigen::Matrix2d S =
        H * covariance_ * H.transpose() + Eigen::Matrix2d::Identity() * 20;
      cost(i, feature) = residual.transpose() * S.inverse() * residual;
    }
  }
  const auto assignments = hungarian_assign(cost, kGate);
  int corrected_blades = 0;
  for (std::size_t i = 0; i < observations.size(); ++i) {
    if (!assignments[i]) continue;
    const int best = *assignments[i];
    const auto & observation = observations[i];
    const auto predicted = *project_feature(x, best, R_camera2world, t_camera2world,
                                            camera_matrix_, distort_coeffs_);
    Eigen::Vector2d residual(observation.pixel.x - predicted.x,
                             observation.pixel.y - predicted.y);
    const auto H = observation_jacobian(x, best, R_camera2world, t_camera2world,
                                        camera_matrix_, distort_coeffs_);
    Eigen::Matrix2d S = H * covariance_ * H.transpose() + Eigen::Matrix2d::Identity() * 20;
    if (residual.transpose() * S.inverse() * residual > kGate) continue;
    const Eigen::Matrix<double, 6, 2> K = covariance_ * H.transpose() * S.inverse();
    x += K * residual;
    const Matrix I = Matrix::Identity() - K * H;
    covariance_ = I * covariance_ * I.transpose() + K *
                   (Eigen::Matrix2d::Identity() * 20) * K.transpose();
    if (best > 0) {
      ++corrected_blades;
      const int blade = best - 1;
      if (observation.inactive) {
        state.inactive[blade] = true;
        inactive_timeout_[blade] = timestamp + std::chrono::milliseconds(100);
      }
    }
  }
  vector_into(state, x);
  if (corrected_blades == 0) return false;
  ++state.update_count;
  const double t = std::chrono::duration<double>(timestamp - state.start_timestamp).count();
  if (t >= 1) {
    fitter_.push(t, state.rotation_angle);
    const auto linear = fitter_.fit_linear();
    const auto sine = big_rune_ ? fitter_.fit_sine() : std::nullopt;
    if (sine && (!linear || (sine->cost < linear->cost && sine->a >= 0.6))) {
      state.sine_v = sine->v;
      state.sine_a = sine->a;
      state.sine_omega = sine->omega;
      state.sine_phase = sine->omega * t + sine->phi;
      state.sine_t = t;
      state.sine_valid = true;
    } else if (linear) {
      state.rotation_speed = linear->speed;
      state.sine_valid = false;
    }
  }
  return true;
}
}  // namespace auto_buff_v2
