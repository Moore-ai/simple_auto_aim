#pragma once

#include <cassert>

#include "tasks/auto_buff_v2/aimers/rune_aimer_factory.hpp"
#include "tasks/auto_buff_v2/models/sp_rune_model.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"

// 使用检测器的实际输出，保留其关键点和状态标记。
inline void check_sp_detector_contract(
  const auto_buff_v2::RuneElements & elements, auto_buff_v2::BuffConfig::Camera camera)
{
  using namespace auto_buff_v2;
  camera.R_camera2gimbal << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  const auto timestamp = Timestamp{} + std::chrono::seconds(10);
  for (const bool big : {false, true}) {
    SPRuneModel model(camera, {}, big);
    assert(model.update(elements, timestamp));
    assert(model.state() && !model.state()->predicted);
    assert(!model.candidates().empty());
    assert(model.reprojected_features().size() == 6);
    for (const auto & type : {"njust", "climber"}) {
      BuffConfig::Planner config;
      config.aimer = type;
      config.ballistic.type = "vacuum";
      const auto aimer = make_rune_aimer(config);
      assert(aimer->aim_at(*model.state(), timestamp + std::chrono::milliseconds(100), 24));
    }
  }
}
