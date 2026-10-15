#include <cassert>
#include <cmath>
#include <fstream>

#include <opencv2/calib3d.hpp>

#include "tasks/auto_buff_v2/aimers/rune_aimer_factory.hpp"
#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tasks/auto_buff_v2/models/njust_rune_model.hpp"
#include "tasks/auto_buff_v2/models/rune_predictor.hpp"

int main()
{
  const char * path = "/tmp/buff_v2_pnp_test.yaml";
  std::ofstream(path)
    << "camera_matrix: [1000, 0, 320, 0, 1000, 240, 0, 0, 1]\n"
    << "distort_coeffs: [0, 0, 0, 0, 0]\n"
    << "camera2gimbal_mode: matrix\n"
    << "R_camera2gimbal: [0, 0, 1, -1, 0, 0, 0, -1, 0]\n"
    << "t_camera2gimbal: [0, 0, 0]\n";
  const auto config = auto_buff_v2::BuffConfig::load(path);
  const auto model_config = auto_buff_v2::NjustRuneModel::Config::load(
    config.model.parameters["njust"]);

  cv::Mat camera = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
  cv::Mat rotation = (cv::Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  const std::vector<cv::Point3f> object = {
    {-0.1f, 0, 0}, {0, 0, 0.85f}, {0, 0.15f, 0.7f},
    {0, 0, 0.55f}, {0, -0.15f, 0.7f}};
  std::vector<cv::Point2f> pixel;
  cv::projectPoints(object, rvec, cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), pixel);
  std::vector<cv::Point2f> bull_center;
  cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.7f}}, rvec, cv::Vec3d(0, 0, 3),
                    camera, cv::Mat::zeros(1, 5, CV_64F), bull_center);
  auto_buff_v2::RuneElements elements;
  elements.icons.push_back({pixel[0], 1});
  elements.bullseyes.push_back(
    {bull_center[0], {pixel[2], pixel[3], pixel[4], pixel[1]}, false, 0.5});
  auto_buff_v2::NjustRuneModel model(config.camera, model_config, false);
  model.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  const auto now = std::chrono::steady_clock::now();
  assert(model.update(elements, now));
  const auto state = model.state();
  assert(state && !state->predicted);
  assert(!model.update({}, now + std::chrono::milliseconds(10)));
  assert(model.state() && model.state()->predicted);
  assert(model.update(elements, now + std::chrono::milliseconds(20)));
  assert(!model.state()->predicted);
  assert(state);
  assert((state->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);
  assert(std::abs(state->rotation_angle) < 0.03);
  auto_buff_v2::NjustRuneModel rotated(config.camera, model_config, false);
  const Eigen::Quaterniond q_gimbal2world(
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()));
  rotated.set_q_gimbal2world(q_gimbal2world);
  assert(rotated.update(elements, now));
  assert((rotated.state()->center - Eigen::Vector3d(3 * std::cos(0.2),
                                                   3 * std::sin(0.2), 0)).norm() < 0.03);
  assert(!state->inactive[0]);
  const auto reprojected = model.reprojected_features();
  assert(reprojected.size() == 6);
  for (std::size_t id = 0; id < reprojected.size(); ++id)
    assert(reprojected[id].id == static_cast<int>(id));
  assert(cv::norm(reprojected.front().point - pixel.front()) < 2.0);
  assert(model.reprojected_center());

  auto_buff_v2::RuneElements active_only = elements;
  active_only.bullseyes.front().active = true;
  auto_buff_v2::NjustRuneModel no_inactive(config.camera, model_config, false);
  assert(!no_inactive.update(active_only, now));
  assert(!no_inactive.state());
  assert(no_inactive.diagnostic().find("inactive=0") != std::string::npos);
  assert(no_inactive.diagnostic().find("need 1 or 2 inactive") != std::string::npos);

  auto_buff_v2::NjustRuneModel delayed(config.camera, model_config, false);
  delayed.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(delayed.update(elements, now));
  assert(delayed.update(elements, now + std::chrono::milliseconds(600)));
  assert(delayed.state());

  auto_buff_v2::RuneElements candidates = elements;
  candidates.icons.insert(candidates.icons.begin(),
                          {pixel[0] + cv::Point2f(12, 0), 0.5});
  auto_buff_v2::NjustRuneModel selected(config.camera, model_config, false);
  selected.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(selected.update(candidates, now));
  assert((selected.state()->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);

  constexpr double kBladeAngle = 2 * 3.14159265358979323846 / 5;
  const auto rotate_blade = [=](double y, double z) {
    return cv::Point3f(0, static_cast<float>(std::cos(kBladeAngle) * y -
                                               std::sin(kBladeAngle) * z),
                       static_cast<float>(std::sin(kBladeAngle) * y +
                                          std::cos(kBladeAngle) * z));
  };
  const std::vector<cv::Point3f> second_object = {
    rotate_blade(0, 0.7), rotate_blade(0, 0.85),
    rotate_blade(0.15, 0.7), rotate_blade(0, 0.55),
    rotate_blade(-0.15, 0.7)};
  std::vector<cv::Point2f> second_pixel;
  cv::projectPoints(second_object, rvec, cv::Vec3d(0, 0, 3), camera,
                    cv::Mat::zeros(1, 5, CV_64F), second_pixel);
  auto_buff_v2::RuneElements layout = candidates;
  layout.bullseyes.insert(layout.bullseyes.begin(),
                          {second_pixel[0], {second_pixel[2], second_pixel[3],
                                             second_pixel[4], second_pixel[1]}, false, 0.5});
  auto_buff_v2::NjustRuneModel layout_model(config.camera, model_config, false);
  layout_model.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(layout_model.update(layout, now));
  assert((layout_model.state()->center - Eigen::Vector3d(3, 0, 0)).norm() < 0.03);
  assert(std::abs(layout_model.state()->rotation_angle) < 0.03);

  auto_buff_v2::RuneElements wrong_center = elements;
  wrong_center.bullseyes.front().center += cv::Point2f(35, 0);
  auto_buff_v2::NjustRuneModel gated(config.camera, model_config, false);
  gated.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(!gated.update(wrong_center, now));
  assert(gated.diagnostic().find("center error") != std::string::npos);

  auto_buff_v2::RuneElements degenerate = elements;
  degenerate.bullseyes.front().corners[0] =
    degenerate.bullseyes.front().corners[2];
  auto_buff_v2::NjustRuneModel rejected(config.camera, model_config, false);
  rejected.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(!rejected.update(degenerate, now));
  assert(rejected.diagnostic().find("last seed rejected") != std::string::npos);

  auto_buff_v2::RuneElements noisy = elements;
  noisy.bullseyes.front().corners[3] += cv::Point2f(3, 0);
  const std::vector<cv::Point2f> noisy_points = {
    noisy.icons.front().center, noisy.bullseyes.front().corners[3],
    noisy.bullseyes.front().corners[0], noisy.bullseyes.front().corners[1],
    noisy.bullseyes.front().corners[2]};
  cv::Vec3d noisy_rvec, noisy_tvec;
  assert(cv::solvePnP(object, noisy_points, camera, cv::Mat::zeros(1, 5, CV_64F),
                      noisy_rvec, noisy_tvec, false, cv::SOLVEPNP_EPNP));
  assert(cv::solvePnP(object, noisy_points, camera, cv::Mat::zeros(1, 5, CV_64F),
                      noisy_rvec, noisy_tvec, true, cv::SOLVEPNP_ITERATIVE));
  const Eigen::Vector3d raw_center(noisy_tvec[2], -noisy_tvec[0], -noisy_tvec[1]);
  auto_buff_v2::NjustRuneModel corrected_seed(config.camera, model_config, false);
  corrected_seed.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(corrected_seed.update(noisy, now));
  assert((corrected_seed.state()->center - raw_center).norm() > 0.002);

  const double seed_angle = 0.3;
  cv::Mat tilted = (cv::Mat_<double>(3, 3) <<
    1, 0, 0, 0, std::cos(seed_angle), -std::sin(seed_angle),
    0, std::sin(seed_angle), std::cos(seed_angle));
  cv::Mat tilted_rvec;
  cv::Rodrigues(rotation * tilted, tilted_rvec);
  std::vector<cv::Point2f> tilted_pixel;
  cv::projectPoints(object, tilted_rvec, cv::Vec3d(0, 0, 3), camera,
                    cv::Mat::zeros(1, 5, CV_64F), tilted_pixel);
  std::vector<cv::Point2f> tilted_center;
  cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.7f}}, tilted_rvec,
                    cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), tilted_center);
  auto_buff_v2::RuneElements tilted_elements;
  tilted_elements.icons.push_back({tilted_pixel[0], 1});
  tilted_elements.bullseyes.push_back(
    {tilted_center[0], {tilted_pixel[2], tilted_pixel[3], tilted_pixel[4], tilted_pixel[1]},
     false, 0.5});
  auto_buff_v2::NjustRuneModel tilted_model(config.camera, model_config, false);
  tilted_model.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(tilted_model.update(tilted_elements, now));
  assert(std::abs(tilted_model.state()->rotation_angle - seed_angle) < 0.03);

  // A stationary rune can change its hittable blade without changing the polygon.
  // Fresh observations must supersede the old blade's 100 ms detection grace period.
  for (const bool observe_old_active : {false, true}) {
    auto_buff_v2::NjustRuneModel switching(config.camera, model_config, false);
    switching.set_q_gimbal2world(q_gimbal2world);
    assert(switching.update(elements, now));
    for (int i = 1; i <= 80; ++i)
      assert(switching.update(elements, now + std::chrono::milliseconds(50 * i)));
    const auto before_switch = switching.reprojected_features();
    auto_buff_v2::RuneElements switched;
    switched.icons = elements.icons;
    if (observe_old_active) {
      switched.bullseyes = elements.bullseyes;
      switched.bullseyes.front().active = true;
    }
    switched.bullseyes.push_back(
      {second_pixel[0], {second_pixel[2], second_pixel[3], second_pixel[4], second_pixel[1]},
       false, 0.5});
    const auto stamp = now + std::chrono::milliseconds(4010);
    assert(switching.update(switched, stamp));
    const auto features = switching.reprojected_features();
    assert(features.size() == 6);
    for (std::size_t i = 0; i < features.size(); ++i)
      assert(cv::norm(features[i].point - before_switch[i].point) < 0.1);
    assert(cv::norm(features[2].point - second_pixel[0]) < 0.1);
    for (const auto & type : {"njust", "climber"}) {
      auto planner_config = config.planner;
      planner_config.aimer = type;
      planner_config.ballistic.type = "vacuum";
      const auto aimer = auto_buff_v2::make_rune_aimer(planner_config);
      const auto aimed = aimer->aim(1, switching.state(), 20, {}, stamp);
      assert(aimed && aimed->plan.control);
      const Eigen::Vector3d camera_point = config.camera.R_camera2gimbal.transpose() *
        (q_gimbal2world.conjugate() * aimed->solution.point - config.camera.t_camera2gimbal);
      std::vector<cv::Point2f> aimed_pixels;
      cv::projectPoints(
        std::vector<cv::Point3f>{{static_cast<float>(camera_point.x()),
                                 static_cast<float>(camera_point.y()),
                                 static_cast<float>(camera_point.z())}},
        cv::Vec3d{}, cv::Vec3d{}, camera, config.camera.distort_coeffs, aimed_pixels);
      assert(cv::norm(aimed_pixels.front() - second_pixel[0]) < 0.1);
      // Independent world yaw: gimbal yaw plus the second blade's bearing.
      const double expected_yaw = 0.2 + std::atan2(-0.7 * std::sin(kBladeAngle), 3.0);
      assert(std::abs(aimed->plan.yaw - expected_yaw) < 1e-4);
    }
    // A missed frame retains the selected blade, but an explicit active observation revokes it.
    const auto missing_stamp = stamp + std::chrono::milliseconds(10);
    assert(!switching.update({}, missing_stamp));
    assert(auto_buff_v2::RunePredictor{}.aimpoint_at(*switching.state(), missing_stamp));
    for (auto & bull : switched.bullseyes) bull.active = true;
    const auto active_stamp = stamp + std::chrono::milliseconds(20);
    assert(switching.update(switched, active_stamp));
    assert(!auto_buff_v2::RunePredictor{}.aimpoint_at(*switching.state(), active_stamp));
  }

  auto_buff_v2::NjustRuneModel big(config.camera, model_config, true);
  auto yaml = YAML::LoadFile(path);
  yaml["buff_v2"]["model"]["njust"]["motion_fit"]["small"]["min_fit_seconds"] = 0.5;
  yaml["buff_v2"]["model"]["njust"]["motion_fit"]["big"]["min_fit_seconds"] = 2.0;
  std::ofstream(path) << yaml;
  const auto fit_config = auto_buff_v2::BuffConfig::load(path);
  const auto fit_model_config = auto_buff_v2::NjustRuneModel::Config::load(
    fit_config.model.parameters["njust"]);
  yaml["buff_v2"]["model"]["njust"]["prediction_method"] = "ekf";
  std::ofstream(path) << yaml;
  const auto ekf_config = auto_buff_v2::BuffConfig::load(path);
  const auto ekf_model_config = auto_buff_v2::NjustRuneModel::Config::load(
    ekf_config.model.parameters["njust"]);
  auto_buff_v2::NjustRuneModel ekf_small(ekf_config.camera, ekf_model_config, false);
  auto_buff_v2::NjustRuneModel big_with_ekf_config(ekf_config.camera, ekf_model_config, true);
  assert(ekf_small.update(elements, now));
  assert(big_with_ekf_config.update(elements, now));
  assert(big_with_ekf_config.state()->prediction_method ==
         auto_buff_v2::RunePredictionMethod::weighted_least_squares);
  auto_buff_v2::NjustRuneModel fast_small(fit_config.camera, fit_model_config, false);
  auto_buff_v2::NjustRuneModel slow_big(fit_config.camera, fit_model_config, true);
  assert(fast_small.update(elements, now));
  assert(slow_big.update(elements, now));
  big.set_q_gimbal2world(Eigen::Quaterniond::Identity());
  assert(big.update(elements, now));
  for (int i = 1; i <= 120; ++i) {
    const double t = 0.05 * i;
    const double angle = t - 0.4 * (std::cos(2 * t + 0.4) - std::cos(0.4));
    std::vector<cv::Point2f> blade;
    cv::projectPoints(std::vector<cv::Point3f>{{0, static_cast<float>(-0.7 * std::sin(angle)),
                                                static_cast<float>(0.7 * std::cos(angle))}},
                      rvec, cv::Vec3d(0, 0, 3), camera, cv::Mat::zeros(1, 5, CV_64F), blade);
    elements.bullseyes.front().center = blade.front();
    const auto stamp = now + std::chrono::milliseconds(50 * i);
    model.update(elements, stamp);
    big.update(elements, stamp);
    fast_small.update(elements, stamp);
    slow_big.update(elements, stamp);
    assert(ekf_small.update(elements, stamp));
    assert(big_with_ekf_config.update(elements, stamp));
    assert(big_with_ekf_config.state()->prediction_method ==
           auto_buff_v2::RunePredictionMethod::weighted_least_squares);
    assert(std::abs(big_with_ekf_config.state()->rotation_angle -
                    slow_big.state()->rotation_angle) < 1e-9);
    assert(std::abs(big_with_ekf_config.state()->rotation_speed -
                    slow_big.state()->rotation_speed) < 1e-9);
    assert(big_with_ekf_config.state()->sine_valid == slow_big.state()->sine_valid);
    const auto current = *ekf_small.state();
    assert(current.prediction_method == auto_buff_v2::RunePredictionMethod::ekf);
    assert(!current.has_fitted_motion && !current.sine_valid);
    assert(std::isfinite(current.rotation_speed));
    const auto future = auto_buff_v2::RunePredictor{}.predict(
      current, stamp + std::chrono::milliseconds(500));
    assert(std::abs(future.rotation_angle - current.rotation_angle -
                    0.5 * current.rotation_speed) < 1e-9);
    assert((future.center - current.center).norm() < 1e-12);
    assert(future.face_yaw == current.face_yaw);
    // Motion samples start after the model's existing one-second warmup.
    if (i == 35) {
      assert(fast_small.state() && fast_small.state()->has_fitted_motion);
      assert(slow_big.state() && !slow_big.state()->has_fitted_motion &&
             !slow_big.state()->sine_valid);
    }
  }
  assert(model.state() && !model.state()->sine_valid);
  assert(big.state() && big.state()->sine_valid);
  assert(big_with_ekf_config.state() && big_with_ekf_config.state()->sine_valid);
  assert(model.state()->prediction_method ==
         auto_buff_v2::RunePredictionMethod::weighted_least_squares);
  assert(std::abs(ekf_small.state()->rotation_speed) > 0.1);
  const auto current = *ekf_small.state();
  assert(!ekf_small.update({}, now + std::chrono::milliseconds(6050)));
  assert(ekf_small.state() && ekf_small.state()->predicted);
  assert(std::abs(ekf_small.state()->rotation_angle - current.rotation_angle -
                  0.05 * current.rotation_speed) < 1e-9);
  ekf_small.reset();
  assert(!ekf_small.state());
  assert(ekf_small.update(tilted_elements, now + std::chrono::milliseconds(6100)));
  assert(ekf_small.state()->prediction_method == auto_buff_v2::RunePredictionMethod::ekf);
  assert(!ekf_small.state()->has_fitted_motion && !ekf_small.state()->sine_valid);

  auto_buff_v2::RuneElements empty;
  for (int i = 121; i <= 153; ++i)
    big.update(empty, now + std::chrono::milliseconds(50 * i));
  assert(!big.state());
  assert(big.diagnostic().find("missing R") != std::string::npos);
  assert(big.update(tilted_elements, now + std::chrono::milliseconds(7700)));
  assert(std::abs(big.state()->rotation_angle - seed_angle) < 0.03);
}
