#include <cassert>
#include <cmath>

#include "fixtures/climber_6f8ae25/buff_big_group.hpp"
#include "fixtures/sp_scene.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_model.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"

using namespace auto_buff_v2;

int main()
{
  auto camera = sp_test::camera();
  camera.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  const auto start = Timestamp{} + std::chrono::seconds(10);
  for (const int direction : {-1, 1}) {
    climber_reference::BigTargetGroup reference;
    SPRuneModel model(camera, {}, true);
    SPRuneSolver solver(camera);
    for (int frame = 0; frame < 220; ++frame) {
      const double t = frame * .02;
      const double roll = direction *
        (1.1775 * t + .9125 / 1.942 * (1 - std::cos(1.942 * t)));
      auto elements = sp_test::scene(camera, Eigen::Quaterniond::Identity(),
          {4, 0, .1}, 0, roll, true);
      if (frame >= 60 && frame < 66) elements.bullseyes.erase(elements.bullseyes.begin());
      if (frame >= 100 && frame < 105) elements = {};
      std::vector<climber_reference::PowerRune> observations;
      for (const auto & obs : solver.solve_all(elements)) {
        climber_reference::PowerRune p;
        p.xyz_in_world = tools::ypd2xyz(obs.ypd_in_world);
        p.trad_xyz_in_world = obs.detected_r_in_world;
        p.ypd_in_world = obs.ypd_in_world;
        p.ypr_in_world = obs.ypr_in_world;
        p.blade_ypd_in_world = obs.blade_ypd_in_world;
        p.R_buff2world = obs.rotation_in_world;
        p.slot_id = obs.slot_id;
        observations.push_back(p);
      }
      auto timestamp = start + std::chrono::milliseconds(frame * 20);
      reference.update(observations, timestamp);
      assert(model.update(elements, timestamp));
      const auto candidates = model.candidates();
      int reference_count = 0;
      for (int slot = 0; slot < 5; ++slot) {
        const auto expected = reference.get_target_copy(slot);
        if (!expected) continue;
        ++reference_count;
        bool found = false;
        for (const auto & candidate : candidates) {
          if (candidate.slot_id != slot) continue;
          found = true;
          assert(std::abs(candidate.rotation_angle - expected->get_roll()) < 1e-8);
          assert(candidate.predicted == expected->is_predicted());
          // 整帧丢失续推时，上游清空物理编号，本项目保留编号。
          if (!candidate.predicted) assert(candidate.blade_id == expected->blade_id());
          auto future = *expected;
          future.predict(.2);
          const auto point = RunePredictor{}.aimpoint_at(
            candidate, timestamp + std::chrono::milliseconds(200));
          assert(point && (*point - future.point_buff2world({0, 0, .7})).norm() < 1e-7);
        }
        assert(found);
      }
      assert(reference_count == static_cast<int>(candidates.size()));
    }
  }
}
