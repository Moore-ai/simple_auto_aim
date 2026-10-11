#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "fixtures/sp_legacy_target.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_tracker.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_model.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"

// Same solved observations go to the frozen reference and the port. Compare
// every state component, including the sine parameters, from the first frame.
template <typename Reference>
void compare(bool big, int direction, const std::string & scenario)
{
  using namespace auto_buff_v2;
  Reference reference;
  SPTracker port(big, SPRuneConfig{});
  const auto start = Timestamp{} + std::chrono::seconds(10);
  int divergent_frames = 0;
  for (int i = 0; i < 250; ++i) {
    const double t = i * 0.02;
    double angle = direction * (big ?
      1.1775 * t + 0.9125 / 1.942 * (1 - std::cos(1.942 * t)) : M_PI / 3 * t);
    if (scenario == "stationary") angle = 0.2;
    if (scenario == "divergence") angle = direction * 4 * t;
    if (i >= 100) angle += 2 * M_PI / 5;
    angle = tools::limit_rad(angle);
    const Eigen::Vector3d center(3, -0.05 + 0.02 * t, -0.03);
    const Eigen::Vector3d blade = center +
      tools::rotation_matrix(Eigen::Vector3d(0.1, 0, angle)) * Eigen::Vector3d(0, 0, 0.7);
    sp_legacy_reference::PowerRune observation;
    observation.ypd_in_world = tools::xyz2ypd(center);
    observation.ypr_in_world = {0.1, 0, angle};
    observation.blade_ypd_in_world = tools::xyz2ypd(blade);
    auto timestamp = start + std::chrono::milliseconds(i * 20);
    const bool missing = scenario == "losses" &&
      ((i >= 20 && i <= 140 && i % 20 == 0) || (i >= 160 && i <= 166));
    std::optional<sp_legacy_reference::PowerRune> reference_observation;
    std::optional<SPRuneObservation> port_observation;
    if (!missing) {
      reference_observation = observation;
      port_observation = SPRuneObservation{observation.ypd_in_world, observation.ypr_in_world,
                                          observation.blade_ypd_in_world};
    }
    reference.get_target(reference_observation, timestamp);
    const bool valid = port.update(port_observation, timestamp);
    assert(valid == !reference.is_unsolve());
    const auto state = reference.ekf_x();
    if (!big ? reference.is_unsolve() :
        state[7] > 1.045 * 1.5 || state[7] < 0.78 / 1.5 ||
        state[8] > 2.0 * 1.5 || state[8] < 1.884 / 1.5) ++divergent_frames;
    const Eigen::VectorXd difference = reference.ekf_x() - port.state();
    const double error = difference.lpNorm<Eigen::Infinity>();
    if (!(error < 1e-8)) {
      std::cerr << "frame=" << i << " max_state_error=" << error << '\n';
      std::cerr << "reference=" << reference.ekf_x().transpose() << '\n';
      std::cerr << "port=" << port.state().transpose() << '\n';
      assert(false);
    }
  }
  if (scenario == "divergence") assert(divergent_frames > 0);
}

// Exercise the actual model adapter and shared planner predictor as well as
// the tracker. The reference receives the same solved camera observation.
template <typename Reference>
void compare_model(bool big, int direction)
{
  using namespace auto_buff_v2;
  BuffConfig::Camera camera;
  camera.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
  camera.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  camera.R_camera2gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  SPRuneSolver solver(camera);
  SPRuneModel model(camera, SPRuneConfig{}, big);
  Reference reference;
  const auto start = Timestamp{} + std::chrono::seconds(10);
  for (int i = 0; i < 250; ++i) {
    const double t = i * 0.02;
    const double angle = direction * (big ?
      1.1775 * t + 0.9125 / 1.942 * (1 - std::cos(1.942 * t)) : M_PI / 3 * t);
    const Eigen::Matrix3d rotation = camera.R_camera2gimbal.transpose() *
      Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
    cv::Mat rmat, rvec;
    cv::eigen2cv(rotation, rmat);
    cv::Rodrigues(rmat, rvec);
    const Eigen::Vector3d center = camera.R_camera2gimbal.transpose() *
      Eigen::Vector3d(3, -0.05 + 0.02 * t, -0.03);
    std::vector<cv::Point2f> pixels;
    cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.827f}, {0, 0.127f, 0.7f},
      {0, 0, 0.573f}, {0, -0.127f, 0.7f}, {0, 0, 0.7f}, {0, 0, 0}},
      rvec, cv::Vec3d(center.x(), center.y(), center.z()),
      camera.camera_matrix, camera.distort_coeffs, pixels);
    RuneElements elements;
    if (!(i >= 20 && i <= 140 && i % 20 == 0)) {
      elements.icons.push_back({pixels[5], 1});
      elements.bullseyes.push_back(
        {pixels[4], {pixels[2], pixels[0], pixels[3], pixels[1]}, false, 1});
    }
    std::optional<sp_legacy_reference::PowerRune> observation;
    if (const auto solved = solver.solve(elements)) {
      observation = sp_legacy_reference::PowerRune{
        solved->ypd_in_world, solved->ypr_in_world, solved->blade_ypd_in_world};
    }
    auto timestamp = start + std::chrono::milliseconds(i * 20);
    reference.get_target(observation, timestamp);
    const bool valid = model.update(elements, timestamp);
    assert(valid == !reference.is_unsolve());
    if (!valid) {
      assert(!model.state());
      continue;
    }
    const auto estimate = *model.state();
    assert(estimate.start_timestamp == start);
    const auto state = reference.ekf_x();
    assert(std::abs(estimate.rotation_angle - state[5]) < 1e-8);
    assert(std::abs(estimate.face_yaw - state[4]) < 1e-8);
    for (const int horizon_ms : {0, 100, 350, 1000}) {
      auto future = reference;
      future.predict(horizon_ms / 1000.0);
      const auto point = RunePredictor{}.aimpoint_at(
        estimate, timestamp + std::chrono::milliseconds(horizon_ms));
      assert(point);
      const auto expected = future.point_buff2world(Eigen::Vector3d(0, 0, 0.7));
      assert((*point - expected).norm() < 1e-8);
    }
  }
}

int main(int argc, char ** argv)
{
  assert(argc == 4);
  const bool big = std::string(argv[1]) == "big";
  const int direction = std::stoi(argv[2]);
  if (std::string(argv[3]) == "model") {
    if (big) compare_model<sp_legacy_reference::BigTarget>(true, direction);
    else compare_model<sp_legacy_reference::SmallTarget>(false, direction);
    return 0;
  }
  if (big) compare<sp_legacy_reference::BigTarget>(true, direction, argv[3]);
  else compare<sp_legacy_reference::SmallTarget>(false, direction, argv[3]);
}
