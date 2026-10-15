#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include "fixtures/climber_6f8ae25/buff_solver.hpp"
#include "fixtures/sp_scene.hpp"
#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "tasks/auto_buff_v2/models/sp_rune_model.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"
#include "tools/math_tools.hpp"
using namespace auto_buff_v2;

using sp_test::camera;
using sp_test::scene;
int main()
{
  auto c = camera(); const auto start = Timestamp{} + std::chrono::seconds(10);
  const Eigen::Quaterniond q(Eigen::AngleAxisd(.17, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-.06, Eigen::Vector3d::UnitY()));
  // 通过独立的实际尺寸几何场景，检查尺寸、角点顺序和重复坐标变换错误。
  for (double distance:{2., 4., 7.}) for (double yaw:{-.3, .15}) for (double height:{-.5, .8}) {
        const Eigen::Vector3d center(distance, .1, height);
        auto e = scene(c, q, center, yaw, .4);
        SPRuneModel model(c, {}, true); model.set_q_gimbal2world(q);
        assert(model.update(e, start));
        assert((model.state()->center - center).norm() < .003);
        const auto point = RunePredictor{}.aimpoint_at(*model.state(), start);
        const Eigen::Vector3d expected = center +
          tools::rotation_matrix({yaw, 0, .4}) * Eigen::Vector3d(0, 0, .7);
        if (!point || (*point - expected).norm() >= .003) {
          const auto x = *model.state();
          std::cerr << "distance=" << distance << " yaw=" << yaw << " height=" << height
                    << " angle=" << x.rotation_angle << " face_yaw=" << x.face_yaw
                    << " slot=" << x.slot_id << " center=" << x.center.transpose() <<
            " expected=" << expected.transpose()
                    << " actual=" <<
              (point ? point->transpose().eval() : Eigen::Vector3d::Zero().transpose().eval()) <<
            "\n";
        }
        assert(point && (*point - expected).norm() < .003);
        assert(cv::norm(*model.reprojected_center() - e.icons[0].center) < .1);
        const auto features = model.reprojected_features();
        bool found = false;
        for (auto f:
          features) if (f.id > 0 && cv::norm(f.point - e.bullseyes[0].center) < .2) found = true;
        assert(found);
      }

  // 使用相同输入运行固定版本的上游解算器。本项目增加了
  // 去畸变处理，因此在零畸变条件下比较射线交点，
  // 并另行使用几何真值验证含畸变场景。
  const std::string path = "/tmp/sp_pnp_reference_" + std::to_string(getpid()) + ".yaml";
  YAML::Node yaml;
  yaml["R_gimbal2imubody"] = std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1};
  yaml["R_camera2gimbal"] = std::vector<double>{0, 0, 1, -1, 0, 0, 0, -1, 0};
  yaml["t_camera2gimbal"] = std::vector<double>{.05, -.02, .03};
  yaml["camera_matrix"] = std::vector<double>{1000, 0, 640, 0, 1000, 480, 0, 0, 1};
  yaml["distort_coeffs"] = std::vector<double>{0, 0, 0, 0, 0};
  std::ofstream(path) << yaml;
  auto pinhole = c; pinhole.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  climber_reference::Solver reference(path);
  reference.set_R_gimbal2world(q);
  for (const auto & mode : {"4", "4i", "5"}) {
    SPRuneConfig cfg; cfg.pnp_mode = mode;
    SPRuneSolver solver(pinhole, cfg); solver.set_q_gimbal2world(q);
    reference.set_pnp_mode(mode);
    for (double distance :
      {2., 4., 7.}) for (double yaw : {-.3, .15}) for (double height : {-.5, .8}) {
          const Eigen::Vector3d center(distance, .1, height);
          auto e = scene(pinhole, q, center, yaw, .4);
          auto port = solver.solve(e); assert(port);
          climber_reference::PowerRune p;
          p.r_center = e.icons[0].center;
          const auto & corners = e.bullseyes[0].corners;
          p.fanblades.push_back({e.bullseyes[0].center, {corners[1], corners[3], corners[0],
                                                         corners[2]}});
          std::optional<climber_reference::PowerRune> obs = p;
          reference.solve(obs); assert(obs);
          assert((port->rotation_in_world - obs->R_buff2world).norm() < 1e-8);
          assert((tools::ypd2xyz(port->ypd_in_world) - obs->xyz_in_world).norm() < 1e-8);
          assert((port->detected_r_in_world - obs->trad_xyz_in_world).norm() < 2e-6);
          assert((port->target_in_world - obs->target_xyz_in_world).norm() < 2e-6);
          assert((tools::ypd2xyz(port->blade_ypd_in_world) - obs->blade_xyz_in_world).norm() <
            1e-7);
          assert(cv::norm(*solver.project(port->target_in_world) - e.bullseyes[0].center) < .001);
          // 实际尺寸下瞄准点相同，意味着方位角和俯仰角几何关系一致。
          assert((tools::xyz2ypd(port->target_in_world) - obs->target_ypd_in_world).norm() < 2e-6);
        }
  }
  std::filesystem::remove(path);

  // 可配置尺寸必须用于 PnP、未来瞄准点预测和 R 标重投影。
  {
    SPRuneConfig cfg; cfg.target_radius = .6; cfg.r_center_z = .02;
    auto e = scene(c, Eigen::Quaterniond::Identity(), {4, 0, 0}, .1, .4, false, .6, .02);
    SPRuneModel model(c, cfg, true);
    assert(model.update(e, start));
    const auto point = RunePredictor{}.aimpoint_at(*model.state(), start);
    const Eigen::Vector3d expected = Eigen::Vector3d(4, 0, 0) +
      tools::rotation_matrix({.1, 0, .4}) * Eigen::Vector3d(0, 0, .6);
    assert(point && (*point - expected).norm() < .003);
    assert(cv::norm(*model.reprojected_center() - e.icons[0].center) < .1);
  }
  {
    auto e = scene(c, Eigen::Quaterniond::Identity(), {4, 0, 0}, .1, .2);
    const auto second =
      scene(c, Eigen::Quaterniond::Identity(), {4.02, 0, 0}, .1, .2 + 2 * M_PI / 5);
    e.bullseyes.push_back(second.bullseyes[0]);
    SPRuneModel model(c, {}, true);
    assert(model.update(e, start));
    assert((model.state()->center - Eigen::Vector3d(4.01, 0, 0)).norm() < .003);
    assert(model.candidates().size() == 2);
  }

  {
    auto e = scene(c, Eigen::Quaterniond::Identity(), {4, 0, 0}, .1, .4);
    SPRuneSolver solver(c);
    solver.set_q_gimbal2world(Eigen::Quaterniond(0, 0, 0, 0));
    assert(!solver.solve(e));
    solver.set_q_gimbal2world(Eigen::Quaterniond::Identity());
    const auto exact = solver.solve(e); assert(exact);
    e.icons[0].center.x += 5;
    const auto shifted = solver.solve(e); assert(shifted);
    assert((shifted->ypd_in_world - exact->ypd_in_world).norm() < 1e-9);
    assert((shifted->detected_r_in_world - exact->detected_r_in_world).norm() > .01);
    e.bullseyes[0].corners.fill({100, 100});
    assert(!solver.solve(e));
  }
  for (const bool big : {false, true}) {
    SPTracker tracker(big, {});
    SPRuneObservation valid{{0, 0, 4}, {0, 0, .2}, {0, 0, 4.7}};
    assert(tracker.update(valid, start));
    auto outlier = valid; outlier.ypd_in_world[2] = 10;
    assert(!tracker.update(outlier, start + std::chrono::milliseconds(20)));
    assert(tracker.update(valid, start + std::chrono::milliseconds(40)));
    outlier = valid; outlier.ypr_in_world[0] = std::numeric_limits<double>::quiet_NaN();
    assert(tracker.update(outlier, start + std::chrono::milliseconds(60)));
    assert(tracker.predicted());
    tracker.reset();
    assert(tracker.direction() == 0);
  }

  // 速度发散判断必须独立于振幅和频率边界。
  {
    SPRuneConfig cfg;
    cfg.divergence_factor = 1000000;
    SPTracker tracker(true, cfg);
    const Eigen::Vector3d center(4, 0, 0);
    const auto observation = [&](double roll) {
      const Eigen::Vector3d blade = center +
        tools::rotation_matrix({0, 0, roll}) * Eigen::Vector3d(0, 0, .7);
      return SPRuneObservation{tools::xyz2ypd(center), {0, 0, roll}, tools::xyz2ypd(blade)};
    };
    assert(tracker.update(observation(0), start));
    const bool accepted = tracker.update(observation(.4), start + std::chrono::milliseconds(20));
    assert(std::abs(tracker.state()[6]) > 4);
    assert(!accepted);
  }
  // 大小符均须支持丢帧续推、超时失效和恢复，并标记纯预测帧以禁止开火。
  for (bool big:{false, true}) {
    SPRuneModel model(c, {}, big);
    auto e = scene(c, Eigen::Quaterniond::Identity(), {4, 0, 0}, 0, .2);
    assert(model.update(e, start));
    assert(model.update({}, start + std::chrono::milliseconds(20)));
    assert(model.state()->predicted);
    for (int i = 2; i <= 20;
      ++i) assert(model.update({}, start + std::chrono::milliseconds(i * 20)));
    assert(!model.update({}, start + std::chrono::milliseconds(420)));
    assert(!model.state());
    assert(model.update(e, start + std::chrono::milliseconds(440)));
    assert(!model.state()->predicted);
    e.bullseyes[0].corners[0].x = std::numeric_limits<float>::quiet_NaN();
    assert(model.update(e, start + std::chrono::milliseconds(460)));
    assert(model.state()->predicted);
  }
  // 图像槽位变化时，物理编号应保持稳定，覆盖多扇叶同时可见、
  // 参考扇叶切换、跨越正负 π 边界及部分遮挡等情况。
  for (int direction : {-1, 1}) {
    SPRuneModel model(c, {}, true);
    std::array<int, 2> identities{{-1, -1}};
    for (int frame = 0; frame < 180; ++frame) {
      const double t = frame * .02;
      const double roll = direction * (1.1775 * t + .9125 / 1.942 * (1 - std::cos(1.942 * t)));
      auto e = scene(c, Eigen::Quaterniond::Identity(), {4, 0, .1}, .12, roll, true);
      // 遮挡第一片物理扇叶，强制切换参考槽位。
      if (frame >= 60 && frame < 80) e.bullseyes.erase(e.bullseyes.begin());
      assert(model.update(e, start + std::chrono::milliseconds(frame * 20)));
      if (model.candidates().size() != e.bullseyes.size()) {
        const auto observations = SPRuneSolver(c).solve_all(e);
        std::cerr << "frame=" << frame << " direction=" << direction << " observed=" <<
          observations.size() << " candidates=" << model.candidates().size() << "\n";
        for (const auto & obs:observations) std::cerr << "slot=" << obs.slot_id << " ypr=" <<
            obs.ypr_in_world.transpose() << "\n";
      }
      assert(model.candidates().size() == e.bullseyes.size());
      const auto current = *model.state();
      for (const auto & candidate : model.candidates()) {
        const auto p = RunePredictor{}.aimpoint_at(candidate, candidate.timestamp);
        assert(p);
        const Eigen::Vector3d base(4, 0, .1);
        int physical = -1;
        for (int b = 0; b < 2; ++b) {
          const Eigen::Vector3d expected = base + tools::rotation_matrix(
            {.12, 0, roll + b * 2 * M_PI / 5}) * Eigen::Vector3d(0, 0, .7);
          if ((*p - expected).norm() < .004) physical = b;
        }
        if (physical <
          0) std::cerr << "physical mismatch frame=" << frame << " direction=" << direction
                       << " slot=" << candidate.slot_id << " roll=" <<
            candidate.rotation_angle << " actual=" << p->transpose() << "\n";
        assert(physical >= 0);
        if (identities[physical] < 0) identities[physical] = candidate.blade_id;
        assert(candidate.blade_id == identities[physical]);
        assert(identities[0] < 0 || identities[1] < 0 || identities[0] != identities[1]);
        const auto future = RunePredictor{}.predict(candidate,
          candidate.timestamp + std::chrono::milliseconds(100));
        if (frame >
          10) assert(direction * tools::limit_rad(
          future.rotation_angle - candidate.rotation_angle) >= 0);
      }
      // 预测在副本上进行，不得改变下一帧的滤波状态。
      RunePredictor{}.predict(current, current.timestamp + std::chrono::seconds(1));
      assert(model.state()->rotation_angle == current.rotation_angle);
    }
    assert(model.state()->has_fitted_motion);
    assert(direction * model.state()->fitted_rotation_speed > 0);
  }
  for (const char * invalid : {"pnp_mode: broken", "max_lost_frames: -1",
                               "target_radius: -0.2", "observation_yaw: .nan",
                               "initial_frequency: 0"}) {
    bool rejected = false;
    try { SPRuneConfig::load(YAML::Load(invalid)); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
  }

}
