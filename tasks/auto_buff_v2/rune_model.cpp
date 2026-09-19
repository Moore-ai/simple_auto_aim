#include "rune_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
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

std::optional<Eigen::Vector3d> RuneState::aimpoint_at(Timestamp prediction_time) const
{
  const auto delay = std::chrono::seconds(sine_valid ? 6 : 3);
  if (prediction_time < timestamp || prediction_time - start_timestamp < delay)
    return std::nullopt;
  RuneState predicted = *this;
  predicted.transition(std::chrono::duration<double>(prediction_time - timestamp).count());
  constexpr double kPi = 3.14159265358979323846;
  for (std::size_t i = 0; i < predicted.inactive.size(); ++i) {
    if (!predicted.inactive[i]) continue;
    const double angle = predicted.rotation_angle + i * 2 * kPi / 5;
    const Eigen::Vector3d local(0, -kRuneGlobalRadius * std::sin(angle),
                                kRuneGlobalRadius * std::cos(angle));
    return predicted.center +
           Eigen::AngleAxisd(predicted.face_yaw, Eigen::Vector3d::UnitZ()) * local;
  }
  return std::nullopt;
}

RuneModel::RuneModel(const std::string & config_path, bool big_rune) : big_rune_(big_rune)
{
  const auto yaml = YAML::LoadFile(config_path);
  const auto buff = yaml["buff_v2"];
  if (buff) {
    config_.timeout_seconds = buff["timeout_seconds"].as<double>(config_.timeout_seconds);
    config_.noise_x = buff["noise_x"].as<double>(config_.noise_x);
    config_.noise_y = buff["noise_y"].as<double>(config_.noise_y);
    config_.noise_z = buff["noise_z"].as<double>(config_.noise_z);
    config_.noise_rotation_speed =
      buff["noise_rotation_speed"].as<double>(config_.noise_rotation_speed);
    config_.noise_rotation_angle =
      buff["noise_rotation_angle"].as<double>(config_.noise_rotation_angle);
    config_.noise_face_yaw = buff["noise_face_yaw"].as<double>(config_.noise_face_yaw);
    config_.noise_observation =
      buff["noise_observation"].as<double>(config_.noise_observation);
    config_.gate_threshold = buff["gate_threshold"].as<double>(config_.gate_threshold);
    config_.init_seed_mean_error =
      buff["init_seed_mean_error"].as<double>(config_.init_seed_mean_error);
    config_.init_seed_max_error =
      buff["init_seed_max_error"].as<double>(config_.init_seed_max_error);
    config_.init_center_gate =
      buff["init_center_gate"].as<double>(config_.init_center_gate);
    config_.init_pitch_bound =
      buff["init_pitch_bound"].as<double>(config_.init_pitch_bound);
    config_.diverge_face_angle =
      buff["diverge_face_angle"].as<double>(config_.diverge_face_angle);
  }
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
  last_inactive_corrected_ = Timestamp{};
  force_sine_until_ = Timestamp{};
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

double squared_pixel_error(const cv::Point2f & a, const cv::Point2f & b)
{
  const double dx = static_cast<double>(a.x) - b.x;
  const double dy = static_cast<double>(a.y) - b.y;
  return dx * dx + dy * dy;
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
                const Eigen::Vector3d & t_camera2world, const RuneModel::Config & config,
                RuneState & state, double & seed_sse, double & seed_max_error)
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
  const double up_length = cv::norm(up);
  if (up_length <= 1e-6) return false;
  const cv::Point2f right(up.y / up_length, -up.x / up_length);
  auto l = bull.corners[sides[0]], r = bull.corners[sides[1]];
  const double first_projection = (l - bull.center).dot(right);
  const double second_projection = (r - bull.center).dot(right);
  if (std::abs(first_projection - second_projection) <= 1e-6) return false;
  if (first_projection < second_projection) std::swap(l, r);

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
  if (face.head<2>().squaredNorm() <= 1e-6) return false;
  if (std::asin(std::abs(face.z())) > config.init_pitch_bound * kPi / 180) return false;
  state.center = R_camera2world * Eigen::Vector3d(tvec[0], tvec[1], tvec[2]) + t_camera2world;
  state.face_yaw = std::atan2(face.y(), face.x());
  const Eigen::Vector3d horizontal_face(std::cos(state.face_yaw),
                                         std::sin(state.face_yaw), 0);
  if ((R_camera2world.transpose() * horizontal_face).z() <= 0) return false;
  const Eigen::Matrix3d local_rotation =
    Eigen::AngleAxisd(-state.face_yaw, Eigen::Vector3d::UnitZ()) * R_page2world;
  state.rotation_angle = std::atan2(-local_rotation(1, 2), local_rotation(2, 2));
  std::vector<cv::Point2f> projected;
  cv::projectPoints(object, rvec, tvec, camera_matrix, distortion, projected);
  seed_sse = 0;
  seed_max_error = 0;
  for (std::size_t i = 0; i < image.size(); ++i) {
    const double error2 = squared_pixel_error(projected[i], image[i]);
    const double error = std::sqrt(error2);
    seed_sse += error2;
    seed_max_error = std::max(seed_max_error, error);
  }
  return std::sqrt(seed_sse / image.size()) <= config.init_seed_mean_error &&
         seed_max_error <= config.init_seed_max_error &&
         state.center.allFinite();
}

void correct_initial_observation(Vector & x, Matrix & covariance, int feature,
                                  const cv::Point2f & observed,
                                  const Eigen::Matrix3d & R_camera2world,
                                  const Eigen::Vector3d & t_camera2world,
                                  const cv::Mat & camera_matrix, const cv::Mat & distortion,
                                  double observation_noise)
{
  const auto predicted = project_feature(x, feature, R_camera2world, t_camera2world,
                                          camera_matrix, distortion);
  if (!predicted) return;
  const Eigen::Vector2d residual(observed.x - predicted->x, observed.y - predicted->y);
  const auto H = observation_jacobian(x, feature, R_camera2world, t_camera2world,
                                       camera_matrix, distortion);
  const Eigen::Matrix2d R = Eigen::Matrix2d::Identity() * observation_noise;
  const Eigen::Matrix2d S = H * covariance * H.transpose() + R;
  const Eigen::Matrix<double, 6, 2> K = covariance * H.transpose() * S.inverse();
  Vector corrected = x + K * residual;
  corrected[5] = normalize_angle(corrected[5]);
  if (!corrected.allFinite()) return;
  const Matrix I = Matrix::Identity() - K * H;
  Matrix posterior = I * covariance * I.transpose() + K * R * K.transpose();
  posterior = 0.5 * (posterior + posterior.transpose());
  if (!posterior.allFinite()) return;
  x = corrected;
  covariance = posterior;
}
}  // namespace

std::vector<RuneReprojectedFeature> RuneModel::reprojected_features() const
{
  if (!state_) return {};
  const Eigen::Matrix3d R_camera2world = q_gimbal2world_.toRotationMatrix() *
                                          R_gimbal2imubody_ * R_camera2gimbal_;
  const Eigen::Vector3d t_camera2world = q_gimbal2world_ *
                                          R_gimbal2imubody_ * t_camera2gimbal_;
  const auto state = vector_from(*state_);
  std::vector<RuneReprojectedFeature> result;
  result.reserve(6);
  for (int id = 0; id <= 5; ++id) {
    const auto point = project_feature(
      state, id, R_camera2world, t_camera2world, camera_matrix_, distort_coeffs_);
    if (point) result.push_back({id, *point});
  }
  return result;
}

std::optional<cv::Point2f> RuneModel::reprojected_center() const
{
  if (!state_) return std::nullopt;
  const Eigen::Matrix3d R_camera2world = q_gimbal2world_.toRotationMatrix() *
                                          R_gimbal2imubody_ * R_camera2gimbal_;
  const Eigen::Vector3d t_camera2world = q_gimbal2world_ *
                                          R_gimbal2imubody_ * t_camera2gimbal_;
  const Eigen::Vector3d camera =
    R_camera2world.transpose() * (state_->center - t_camera2world);
  if (camera.z() <= 0.1) return std::nullopt;
  std::vector<cv::Point2f> pixels;
  cv::projectPoints(
    std::vector<cv::Point3f>{{static_cast<float>(camera.x()), static_cast<float>(camera.y()),
                              static_cast<float>(camera.z())}},
    cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera_matrix_, distort_coeffs_, pixels);
  return pixels.front();
}

bool RuneModel::update(const RuneElements & elements, Timestamp timestamp)
{
  const Eigen::Matrix3d R_camera2world = q_gimbal2world_.toRotationMatrix() *
                                          R_gimbal2imubody_ * R_camera2gimbal_;
  const Eigen::Vector3d t_camera2world = q_gimbal2world_ *
                                          R_gimbal2imubody_ * t_camera2gimbal_;
  if (!state_) return initialize(elements, timestamp, R_camera2world, t_camera2world);

  auto & state = *state_;
  if (std::chrono::duration<double>(timestamp - last_inactive_corrected_).count() >
      config_.timeout_seconds) {
    reset();
    return initialize(elements, timestamp, R_camera2world, t_camera2world);
  }
  const double dt = std::chrono::duration<double>(timestamp - state.timestamp).count();
  if (dt < 0 || dt > 0.5) { reset(); return false; }
  state.transition(dt);
  state.timestamp = timestamp;
  Vector x = ekf_state_;
  x[4] += x[3] * dt;
  Matrix F = Matrix::Identity();
  F(4, 3) = dt;
  Matrix Q = Matrix::Zero();
  Q.diagonal() << config_.noise_x, config_.noise_y, config_.noise_z,
    config_.noise_rotation_speed, config_.noise_rotation_angle, config_.noise_face_yaw;
  covariance_ = F * covariance_ * F.transpose() + Q;
  for (std::size_t i = 0; i < 5; ++i)
    if (timestamp >= inactive_timeout_[i]) state.inactive[i] = false;

  struct Observation { cv::Point2f pixel; bool icon; bool inactive; };
  std::vector<Observation> observations;
  for (const auto & icon : elements.icons) observations.push_back({icon.center, true, false});
  for (const auto & bull : elements.bullseyes)
    observations.push_back({bull.center, false, !bull.active});
  const double kGate = config_.gate_threshold;
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
        H * covariance_ * H.transpose() +
        Eigen::Matrix2d::Identity() * config_.noise_observation;
      cost(i, feature) = residual.transpose() * S.inverse() * residual;
    }
  }
  const auto assignments = hungarian_assign(cost, kGate);
  int corrected_blades = 0;
  int corrected_inactive = 0;
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
    Eigen::Matrix2d S = H * covariance_ * H.transpose() +
                        Eigen::Matrix2d::Identity() * config_.noise_observation;
    if (residual.transpose() * S.inverse() * residual > kGate) continue;
    const Eigen::Matrix<double, 6, 2> K = covariance_ * H.transpose() * S.inverse();
    x += K * residual;
    const Matrix I = Matrix::Identity() - K * H;
    covariance_ = I * covariance_ * I.transpose() + K *
                   (Eigen::Matrix2d::Identity() * config_.noise_observation) * K.transpose();
    if (best > 0) {
      ++corrected_blades;
      const int blade = best - 1;
      if (observation.inactive) {
        ++corrected_inactive;
        state.inactive[blade] = true;
        inactive_timeout_[blade] = timestamp + std::chrono::milliseconds(100);
      }
    }
  }
  ekf_state_ = x;
  vector_into(state, x);
  if (state.sine_valid)
    state.rotation_speed = state.sine_v + state.sine_a * std::sin(state.sine_phase);
  else if (state.use_prediction_speed)
    state.rotation_speed = state.prediction_speed;
  if (diverged()) { reset(); return false; }
  if (corrected_inactive > 0) last_inactive_corrected_ = timestamp;
  if (corrected_inactive > 1) force_sine_until_ = timestamp + std::chrono::seconds(3);
  if (corrected_blades == 0) return false;
  ++state.update_count;
  const double elapsed_seconds =
    std::chrono::duration<double>(timestamp - state.start_timestamp).count();
  if (elapsed_seconds >= 1) update_motion_fit(state, elapsed_seconds);
  return true;
}

bool RuneModel::initialize(const RuneElements & elements, Timestamp timestamp,
                           const Eigen::Matrix3d & R_camera2world,
                           const Eigen::Vector3d & t_camera2world)
{
  const auto inactive_count = std::count_if(elements.bullseyes.begin(),
                                            elements.bullseyes.end(),
                                            [](const auto & bull) { return !bull.active; });
  if (elements.icons.empty() || inactive_count == 0 || inactive_count > 2) return false;
  struct Candidate
  {
    RuneState state;
    cv::Point2f icon_pixel;
    cv::Point2f seed_pixel;
    int inactive_inliers = 0;
    double seed_center_error = 0;
    double icon_error = 0;
    double seed_max_error = 0;
    double seed_sse = 0;
    double inactive_center_sse = 0;
  };
  const auto rank = [](const Candidate & item) {
    return std::make_tuple(-item.inactive_inliers, item.seed_center_error,
                           item.icon_error, item.seed_max_error, item.seed_sse,
                           item.inactive_center_sse);
  };
  std::optional<Candidate> best;
  for (const auto & icon : elements.icons) {
    for (const auto & bull : elements.bullseyes) {
      if (bull.active) continue;
      RuneState seed;
      double seed_sse = 0, seed_max_error = 0;
      if (!solve_seed(icon, bull, camera_matrix_, distort_coeffs_, R_camera2world,
                      t_camera2world, config_, seed, seed_sse, seed_max_error)) continue;
      const Vector x = vector_from(seed);
      const auto projected_icon = project_feature(x, 0, R_camera2world, t_camera2world,
                                                   camera_matrix_, distort_coeffs_);
      const auto projected_seed = project_feature(x, 1, R_camera2world, t_camera2world,
                                                   camera_matrix_, distort_coeffs_);
      if (!projected_icon || !projected_seed) continue;
      Candidate candidate;
      candidate.state = seed;
      candidate.icon_pixel = icon.center;
      candidate.seed_pixel = bull.center;
      candidate.icon_error = squared_pixel_error(*projected_icon, icon.center);
      candidate.seed_center_error = squared_pixel_error(*projected_seed, bull.center);
      candidate.seed_sse = seed_sse;
      candidate.seed_max_error = seed_max_error;
      const double center_gate2 = config_.init_center_gate * config_.init_center_gate;
      if (candidate.seed_center_error > center_gate2) continue;
      if (inactive_count > 1)
        candidate.inactive_center_sse = std::numeric_limits<double>::max();
      for (const auto & other : elements.bullseyes) {
        if (&other == &bull || other.active) continue;
        double nearest2 = std::numeric_limits<double>::max();
        for (int feature = 2; feature <= 5; ++feature) {
          const auto predicted = project_feature(x, feature, R_camera2world,
                                                  t_camera2world, camera_matrix_,
                                                  distort_coeffs_);
          if (predicted) {
            nearest2 = std::min(nearest2,
                                squared_pixel_error(*predicted, other.center));
          }
        }
        if (nearest2 <= center_gate2) {
          ++candidate.inactive_inliers;
          if (candidate.inactive_inliers == 1) candidate.inactive_center_sse = 0;
          candidate.inactive_center_sse += nearest2;
        }
      }
      const bool better = !best || rank(candidate) < rank(*best);
      if (better) best = candidate;
    }
  }
  if (!best) return false;
  RuneState seed = best->state;
  seed.start_timestamp = timestamp;
  seed.timestamp = timestamp;
  inactive_timeout_.fill(Timestamp{});
  state_ = seed;
  ekf_state_ = vector_from(seed);
  last_inactive_corrected_ = timestamp;
  covariance_.diagonal() << 64, 64, 64, 100, 25, 10;
  correct_initial_observation(ekf_state_, covariance_, 0, best->icon_pixel,
                               R_camera2world, t_camera2world, camera_matrix_,
                               distort_coeffs_, config_.noise_observation);
  correct_initial_observation(ekf_state_, covariance_, 1, best->seed_pixel,
                               R_camera2world, t_camera2world, camera_matrix_,
                               distort_coeffs_, config_.noise_observation);
  vector_into(*state_, ekf_state_);
  fitter_.reset();
  return true;
}

void RuneModel::update_motion_fit(RuneState & state, double elapsed_seconds)
{
  fitter_.push(elapsed_seconds, state.rotation_angle);
  const auto linear = fitter_.fit_linear();
  const auto sine = big_rune_ ? fitter_.fit_sine() : std::nullopt;
  if (sine && (!linear || state.timestamp < force_sine_until_ ||
               (sine->cost < linear->cost && sine->a >= 0.6))) {
    state.sine_v = sine->v;
    state.sine_a = sine->a;
    state.sine_omega = sine->omega;
    state.sine_phase = sine->omega * elapsed_seconds + sine->phi;
    state.sine_t = elapsed_seconds;
    state.sine_valid = true;
    state.use_prediction_speed = false;
    state.prediction_cost = sine->cost;
    state.rotation_speed = sine->v + sine->a * std::sin(state.sine_phase);
  } else if (linear) {
    state.prediction_speed = linear->speed;
    state.rotation_speed = linear->speed;
    state.use_prediction_speed = true;
    state.sine_valid = false;
    state.prediction_cost = linear->cost;
  }
}

bool RuneModel::diverged() const
{
  if (!ekf_state_.allFinite() || !covariance_.allFinite()) return true;
  if (covariance_(0, 0) > 150 || covariance_(1, 1) > 150 ||
      std::abs(ekf_state_[0]) > 15 || std::abs(ekf_state_[1]) > 15 ||
      std::abs(ekf_state_[2]) > 5 || std::abs(ekf_state_[3]) > 10 * kPi)
    return true;
  const double radius = std::hypot(ekf_state_[0], ekf_state_[1]);
  if (radius > 0.5) {
    const double to_center = std::atan2(ekf_state_[1], ekf_state_[0]);
    if (std::abs(normalize_angle(ekf_state_[5] - to_center)) >
        config_.diverge_face_angle * kPi / 180) return true;
  }
  return false;
}
}  // namespace auto_buff_v2
