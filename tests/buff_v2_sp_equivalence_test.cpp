#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "fixtures/climber_6f8ae25/buff_target.hpp"
#include "fixtures/sp_scene.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_model.hpp"
#include "tasks/auto_buff_v2/models/rune_predictor.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_tracker.hpp"
#include "tools/math_tools.hpp"

using namespace auto_buff_v2;

template <class Reference>
void compare(bool big, int direction, const std::string & scenario)
{
  Reference reference;
  SPTracker port(big, SPRuneConfig{});
  const auto start = Timestamp{} + std::chrono::seconds(10);
  for (int i = 0; i < 250; ++i) {
    const double t = i * 0.02;
    double angle = direction * (big ?
      1.1775 * t + 0.9125 / 1.942 * (1 - std::cos(1.942 * t)) : 1.047 * t);
    if (scenario == "stationary") angle = 0.2;
    if (scenario == "divergence") angle = direction * 4 * t;
    if (i >= 100) angle += 2 * M_PI / 5;
    angle = tools::limit_rad(angle);
    const Eigen::Vector3d center(3, -0.05 + 0.02 * t, -0.03);
    const Eigen::Vector3d blade = center +
      tools::rotation_matrix(Eigen::Vector3d(0.1, 0, angle)) * Eigen::Vector3d(0, 0, 0.7);
    climber_reference::PowerRune observation;
    observation.xyz_in_world = center;
    observation.trad_xyz_in_world = center;
    observation.ypd_in_world = tools::xyz2ypd(center);
    observation.ypr_in_world = {0.1, 0, angle};
    observation.blade_ypd_in_world = tools::xyz2ypd(blade);
    observation.R_buff2world = tools::rotation_matrix(observation.ypr_in_world);
    auto timestamp = start + std::chrono::milliseconds(i * 20);
    // 上游小符模型不支持丢帧续推；这里只比较有效观测序列，
    // 本项目增加的丢帧续推能力单独测试。
    const bool missing = big && scenario == "losses" &&
      ((i >= 20 && i <= 140 && i % 20 == 0) || (i >= 160 && i <= 166));
    std::optional<climber_reference::PowerRune> ref_obs;
    std::optional<SPRuneObservation> obs;
    if (!missing) {
      ref_obs = observation;
      obs = SPRuneObservation{observation.ypd_in_world, observation.ypr_in_world,
                              observation.blade_ypd_in_world};
    }
    if constexpr (std::is_same_v<Reference, climber_reference::BigTarget>) {
      if (obs) {
        // 切换扇叶时物理自转角增加 72 度，
        // 图像平面槽位编号则沿相反方向变化。
        const int slot = i >= 100 ? 4 : 0;
        reference.set_blade_index(slot);
        port.set_slot(slot);
      }
    }
    reference.get_target(ref_obs, timestamp);
    const bool valid = port.update(obs, timestamp);
    const auto rx = reference.ekf_x();
    const bool diverged = big && (rx[7] > 1.045 * 1.5 || rx[7] < .78 / 1.5 ||
      rx[8] > 2.0 * 1.5 || rx[8] < 1.884 / 1.5);
    // 上游重新初始化前仍输出发散帧，本项目将该帧
    // 立即标为无效；数值状态仍须与上游一致。
    assert(valid == (!reference.is_unsolve() && !diverged));
    if (reference.is_unsolve()) continue;
    const double cov_error = (reference.covariance() - port.covariance()).cwiseAbs().maxCoeff();
    if (cov_error >= 1e-8) std::cerr << "covariance frame=" << i << " error=" << cov_error << "\n"
                                     << "ref=" << reference.ekf_x().transpose() << "\nport=" <<
        port.state().transpose() << "\n";
    assert(cov_error < 1e-8);
    const double error = (reference.ekf_x() - port.state()).template lpNorm<Eigen::Infinity>();
    if (!(error < 1e-8)) {
      std::cerr << "frame=" << i << " max_state_error=" << error << '\n'
                << "reference=" << reference.ekf_x().transpose() << '\n'
                << "port=" << port.state().transpose() << '\n';
      assert(false);
    }
  }
}

template <class Reference>
void compare_model(bool big, int direction)
{
  auto c = sp_test::camera();
  SPRuneModel model(c, {}, big);
  SPRuneSolver solver(c);
  Reference reference;
  const auto start = Timestamp{} + std::chrono::seconds(10);
  for (int i = 0; i < 180; ++i) {
    const double t = i * .02;
    const double roll = direction * (big ?
      1.1775 * t + .9125 / 1.942 * (1 - std::cos(1.942 * t)) : 1.047 * t);
    auto elements = sp_test::scene(c, Eigen::Quaterniond::Identity(),
        {4, -.05 + .02 * t, -.03}, .1, roll);
    auto timestamp = start + std::chrono::milliseconds(i * 20);
    const auto obs = solver.solve(elements); assert(obs);
    climber_reference::PowerRune p;
    p.ypd_in_world = obs->ypd_in_world;
    p.ypr_in_world = obs->ypr_in_world;
    p.blade_ypd_in_world = obs->blade_ypd_in_world;
    p.xyz_in_world = tools::ypd2xyz(obs->ypd_in_world);
    p.trad_xyz_in_world = obs->detected_r_in_world;
    p.R_buff2world = obs->rotation_in_world;
    if constexpr (std::is_same_v<Reference, climber_reference::BigTarget>)
      reference.set_blade_index(obs->slot_id);
    reference.get_target(p, timestamp);
    assert(model.update(elements, timestamp));
    const auto state = *model.state();
    assert(std::abs(state.rotation_angle - reference.get_roll()) < 1e-8);
    assert((state.center - reference.center_filtered()).norm() < 1e-8);
    for (const double dt : {0., .1, .35, 1.}) {
      auto future = reference;
      future.predict(dt);
      const auto point = RunePredictor{}.aimpoint_at(state,
        timestamp +
        std::chrono::duration_cast<Timestamp::duration>(std::chrono::duration<double>(dt)));
      assert(point);
      const auto expected = future.point_buff2world({0, 0, .7});
      if ((*point - expected).norm() >= 1e-7)
        std::cerr << "model frame=" << i << " dt=" << dt << " error=" <<
            (*point - expected).norm() << "\n";
      assert((*point - expected).norm() < 1e-7);
    }
    assert(model.state()->rotation_angle == state.rotation_angle);
  }
}

int main(int argc, char * * argv)
{
  assert(argc == 4);
  const bool big = std::string(argv[1]) == "big";
  const int direction = std::stoi(argv[2]);
  if (std::string(argv[3]) == "model") {
    if (big) compare_model<climber_reference::BigTarget>(big, direction);
    else compare_model<climber_reference::SmallTarget>(big, direction);
    return 0;
  }
  if (big) compare<climber_reference::BigTarget>(true, direction, argv[3]);
  else compare<climber_reference::SmallTarget>(false, direction, argv[3]);
}
