#ifndef AUTO_BUFF_V2__SP_RUNE_MODEL_HPP
#define AUTO_BUFF_V2__SP_RUNE_MODEL_HPP

#include <memory>

#include "rune_model.hpp"
#include "sp_rune_config.hpp"
#include "sp_motion_fitter.hpp"
#include "sp_rune_solver.hpp"
#include "sp_rune_tracker.hpp"

namespace auto_buff_v2
{
// 在 v2 接口中适配 Climber 大小符状态估计与五槽位观测。
class SPRuneModel final : public RuneModel
{
public:
  using Config = SPRuneConfig;
  SPRuneModel(BuffConfig::Camera camera, Config config, bool big_rune);
  void set_q_gimbal2world(const Eigen::Quaterniond & q) override;
  bool update(const RuneElements & elements, Timestamp timestamp) override;
  void reset() override;
  std::optional<RuneEstimate> state() const override { return state_; }
  std::vector<RuneEstimate> candidates() const override;
  const std::string & diagnostic() const override { return diagnostic_; }
  std::vector<RuneReprojectedFeature> reprojected_features() const override;
  std::optional<cv::Point2f> reprojected_center() const override;

private:
  void update_blade_tracks();
  struct BladeTrack
  {
    bool active = false;
    double roll = 0;
    int missed = 0;
  };
  std::array<BladeTrack, 5> blade_tracks_{};
  std::array<bool, 5> visible_slots_{};
  std::array<int, 5> blade_ids_{{-1, -1, -1, -1, -1}};
  int reference_slot_ = -1;
  Eigen::Vector3d pose_center_ = Eigen::Vector3d::Zero();
  SPRuneSolver solver_;
  Config config_;
  SPMotionFitter motion_fitter_;
  bool big_rune_;
  std::unique_ptr<SPTracker> tracker_;
  std::optional<RuneEstimate> state_;
  std::string diagnostic_;
};
}  // namespace auto_buff_v2

#endif
