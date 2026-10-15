#include <cassert>
#include <cmath>
#include <stdexcept>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "tasks/auto_buff_v2/models/rune_model_factory.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"

using namespace auto_buff_v2;

RuneElements observation(const BuffConfig::Camera & camera, double angle, double center_yaw = 0)
{
  const Eigen::Matrix3d rotation = camera.R_camera2gimbal.transpose() *
    Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitX()).toRotationMatrix();
  cv::Mat rmat, rvec;
  cv::eigen2cv(rotation, rmat);
  cv::Rodrigues(rmat, rvec);
  std::vector<cv::Point2f> pixels;
  // 独立于解算器构造 Climber 当前使用的非对称平面靶面尺寸。
  const Eigen::Vector3d center = camera.R_camera2gimbal.transpose() *
    (Eigen::AngleAxisd(center_yaw, Eigen::Vector3d::UnitZ()) * Eigen::Vector3d(3, -0.05, -0.03));
  cv::projectPoints(std::vector<cv::Point3f>{{0, 0, 0.815f}, {0, 0.150f, 0.7f},
    {0, 0, 0.585f}, {0, -0.115f, 0.7f}, {0, 0, 0.7f}, {0, 0, 0}},
    rvec, cv::Vec3d(center.x(), center.y(), center.z()),
    camera.camera_matrix, camera.distort_coeffs, pixels);
  RuneElements result;
  result.icons.push_back({pixels[5], 1});
  result.bullseyes.push_back({pixels[4], {pixels[2], pixels[0], pixels[3], pixels[1]}, false, 1});
  return result;
}

int main()
{
  BuffConfig config;
  config.camera.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
  config.camera.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  config.camera.R_camera2gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  auto original = make_rune_model(config.camera, config.model, false);
  assert(dynamic_cast<NjustRuneModel *>(original.get()));
  config.model.type = "sp";
  const auto start = Timestamp{} + std::chrono::seconds(10);
  for (const bool big : {false, true}) {
    for (const int direction : {-1, 1}) {
      auto model = make_rune_model(config.camera, config.model, big);
      assert(dynamic_cast<SPRuneModel *>(model.get()));
      for (int i = 0; i < 350; ++i) {
        const double t = i * 0.02;
        const double angle = direction * (big ?
          1.1775 * t + 0.9125 / 1.942 * (1 - std::cos(1.942 * t)) : M_PI / 3 * t);
        const double blade_offset = i >= 175 ? 2 * M_PI / 5 : 0;
        assert(model->update(observation(config.camera, angle + blade_offset),
          start + std::chrono::milliseconds(i * 20)));
      }
      const auto state = model->state();
      assert(state && !state->predicted && state->inactive[0]);
      assert((state->center - Eigen::Vector3d(3, -0.05, -0.03)).norm() < 0.08);
      assert(direction * state->rotation_speed > 0);
      assert(state->sine_valid == big);
      if (!big) assert(std::abs(state->rotation_speed - direction * M_PI / 3) < 0.15);
      const auto future = RunePredictor{}.predict(
        *state, state->timestamp + std::chrono::milliseconds(100));
      assert(direction *
        std::remainder(future.rotation_angle - state->rotation_angle, 2 * M_PI) > 0);
      const double t = 6.98;
      const double expected = direction * (big ?
        1.1775 * t + 0.9125 / 1.942 * (1 - std::cos(1.942 * t)) : M_PI / 3 * t) + 2 * M_PI / 5;
      assert(std::abs(std::remainder(state->rotation_angle - expected, 2 * M_PI)) < 0.08);
      assert(model->reprojected_features().size() == 6);
      assert(model->reprojected_center());
      assert(model->update({}, state->timestamp + std::chrono::milliseconds(20)));
      assert(model->state()->predicted);
      assert(!model->update({}, state->timestamp + std::chrono::seconds(2)));
      assert(!model->state());
      assert(model->update(observation(config.camera, 0.2), start + std::chrono::seconds(10)));
      model->reset();
      assert(!model->state());
    }
  }
  auto active = observation(config.camera, 0.2);
  active.bullseyes[0].active = true;
  auto model = make_rune_model(config.camera, config.model, false);
  assert(!model->update(active, start) && !model->state());
  assert(model->update(observation(config.camera, 0.2), start));
  assert(std::abs(model->state()->rotation_speed - 1.047) < 1e-9);
  assert(RunePredictor{}.aimpoint_at(*model->state(), start));
  for (int i = 1; i <= 20; ++i)
    assert(model->update({}, start + std::chrono::milliseconds(i * 20)));
  assert(!model->update({}, start + std::chrono::milliseconds(420)));
  assert(!model->state());
  assert(model->update(observation(config.camera, 0.2),
    start + std::chrono::milliseconds(440)));
  assert(RunePredictor{}.aimpoint_at(*model->state(), start + std::chrono::milliseconds(440)));
  assert(!model->update(observation(config.camera, 0.2), start - std::chrono::seconds(1)));
  assert(!model->state());
  // 预测未来位置时，Climber 位姿锚点保持在最近一次测量的中心。
  auto moving = make_rune_model(config.camera, config.model, false);
  for (int i = 0; i <= 200; ++i) {
    const double t = i * 0.02;
    assert(moving->update(observation(config.camera, M_PI / 3 * t, 0.1 * t),
      start + std::chrono::milliseconds(i * 20)));
  }
  const auto moving_state = moving->state();
  assert(moving_state);
  const auto moving_future = RunePredictor{}.predict(*moving_state,
    moving_state->timestamp + std::chrono::milliseconds(200));
  const Eigen::Vector3d expected_center = Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
    Eigen::Vector3d(3, -0.05, -0.03);
  assert((moving_future.center - expected_center).norm() < 0.02);
  auto rotated = make_rune_model(config.camera, config.model, false);
  rotated->set_q_gimbal2world(Eigen::Quaterniond(
    Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ())));
  assert(rotated->update(observation(config.camera, 0.2), start));
  const Eigen::Vector3d rotated_center = Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()) *
    Eigen::Vector3d(3, -0.05, -0.03);
  assert((rotated->state()->center - rotated_center).norm() < 0.03);
  // 工厂必须使用 SP 参数，不读取 Njust 参数或旧的根节点配置。
  config.model.parameters = YAML::Load(R"(
sp:
  max_lost_frames: 2
  initial_small_speed: 0.6
  observation_yaw: 0.02
njust:
  timeout_seconds: 5.0
)");
  auto configured = make_rune_model(config.camera, config.model, false);
  assert(configured->update(observation(config.camera, 0.2), start));
  assert(std::abs(configured->state()->rotation_speed - 0.6) < 1e-12);
  for (int i = 1; i <= 2; ++i)
    assert(configured->update({}, start + std::chrono::milliseconds(i * 20)));
  assert(!configured->update({}, start + std::chrono::milliseconds(60)));
  assert(!configured->state());
  assert(configured->update(observation(config.camera, 0.2),
    start + std::chrono::milliseconds(80)));
  assert(configured->update(observation(config.camera, 0.2),
    start + std::chrono::milliseconds(100)));

  config.model.type = "unknown";
  bool rejected = false;
  try { make_rune_model(config.camera, config.model, false); }
  catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);
}
