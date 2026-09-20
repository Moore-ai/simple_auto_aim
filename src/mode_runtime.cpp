#include "mode_runtime.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/frame_runtime.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_buff_v2/buff_planner.hpp"
#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tasks/auto_buff_v2/frame_runtime.hpp"
#include "tools/detect_factory.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/processed_frame.hpp"
#include "tools/recorder.hpp"
#include "tools/thread_safe_queue.hpp"

namespace standard
{
namespace
{
using namespace std::chrono_literals;

void send_plan(io::Gimbal & gimbal, const auto_aim::Plan & plan, io::InfantryFireCommand fire)
{
  gimbal.send(
    plan.control, fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
    plan.pitch_acc, plan.distance);
}

void stop_gimbal(io::Gimbal & gimbal)
{
  gimbal.send(false, io::InfantryFireCommand::none, 0, 0, 0, 0, 0, 0, -1);
}

class ModeSession
{
public:
  virtual ~ModeSession() = default;

  virtual bool next(tools::ProcessedFrame & processed) = 0;
};

class AutoAimSession final : public ModeSession
{
public:
  AutoAimSession(const std::string & config_path, io::Camera & camera, io::Gimbal & gimbal,
                 auto_aim::Solver & solver, tools::FoxgloveVisualizer & foxglove)
  : gimbal_(gimbal), tracker_(config_path, solver), planner_(config_path),
    detector_(tools::create_detector(config_path)),
    runtime_(camera, gimbal, solver, tracker_, *detector_), foxglove_(foxglove)
  {
    target_queue_.push({0, Target{}});
    plan_thread_ = std::thread([this] { plan(); });
  }

  ~AutoAimSession() override
  {
    quit_ = true;
    plan_thread_.join();
    stop_gimbal(gimbal_);
  }

  bool next(tools::ProcessedFrame & processed) override
  {
    if (!runtime_.next(processed)) return false;
    Target target;
    if (!processed.targets.empty()) target = processed.targets.front();
    target_queue_.push({processed.snapshot.target_generation, target});
    return true;
  }

private:
  using Target = std::optional<auto_aim::Target>;
  using Request = std::pair<std::uint64_t, Target>;

  void plan()
  {
    while (!quit_) {
      if (target_queue_.empty()) {
        std::this_thread::sleep_for(200ms);
        continue;
      }

      const auto [target_generation, target] = target_queue_.front();
      const auto plan = planner_.plan(target, gimbal_.state().bullet_speed);
      foxglove_.update_plan(target_generation, plan);
      const auto fire =
        plan.fire ? io::InfantryFireCommand::continuous : io::InfantryFireCommand::none;
      send_plan(gimbal_, plan, fire);
      std::this_thread::sleep_for(1ms);
    }
  }

  io::Gimbal & gimbal_;
  auto_aim::Tracker tracker_;
  auto_aim::Planner planner_;
  std::unique_ptr<tools::DetectionBackend> detector_;
  auto_aim::FrameRuntime runtime_;
  tools::FoxgloveVisualizer & foxglove_;
  tools::ThreadSafeQueue<Request, true> target_queue_{1};
  std::atomic<bool> quit_{false};
  std::thread plan_thread_;
};

class BuffSession final : public ModeSession
{
public:
  BuffSession(
    const std::string & config_path, bool big_buff, io::Camera & camera, io::Gimbal & gimbal,
    tools::FoxgloveVisualizer & foxglove)
  : config_(auto_buff_v2::BuffConfig::load(config_path)), gimbal_(gimbal), planner_(config_path),
    model_(config_.camera, config_.model, big_buff), buff_planner_(config_.planner),
    runtime_(camera, gimbal, model_, config_.detector), foxglove_(foxglove)
  {
    target_queue_.push({0, Target{}});
    plan_thread_ = std::thread([this] { plan(); });
  }

  ~BuffSession() override
  {
    quit_ = true;
    plan_thread_.join();
    stop_gimbal(gimbal_);
  }

  bool next(tools::ProcessedFrame & processed) override
  {
    if (!runtime_.next(processed)) return false;
    target_queue_.push({processed.snapshot.target_generation, processed.buff_target});
    return true;
  }

private:
  using Target = std::optional<auto_buff_v2::RuneEstimate>;
  using Request = std::pair<std::uint64_t, Target>;

  void plan()
  {
    while (!quit_) {
      if (target_queue_.empty()) {
        std::this_thread::sleep_for(200ms);
        continue;
      }

      const auto [target_generation, target] = target_queue_.front();
      const auto state = gimbal_.state();
      const auto now = std::chrono::steady_clock::now();
      auto_aim::Plan plan;
      if (const auto request =
            buff_planner_.prepare(target_generation, target, state.bullet_speed, now)) {
        plan = planner_.plan(request->trajectory, request->yaw0, request->distance);
        plan.debug_xyza = {
          request->aimpoint.x(), request->aimpoint.y(), request->aimpoint.z(), request->yaw0};
        plan.fly_time = request->fly_time;
        plan.fire = buff_planner_.fire_advice(*request, plan, state, now);
      }
      foxglove_.update_plan(target_generation, plan);
      const auto fire = plan.fire ? io::InfantryFireCommand::single : io::InfantryFireCommand::none;
      send_plan(gimbal_, plan, fire);
      std::this_thread::sleep_for(1ms);
    }
  }

  auto_buff_v2::BuffConfig config_;
  io::Gimbal & gimbal_;
  auto_aim::Planner planner_;
  auto_buff_v2::RuneModel model_;
  auto_buff_v2::BuffPlanner buff_planner_;
  auto_buff_v2::FrameRuntime runtime_;
  tools::FoxgloveVisualizer & foxglove_;
  tools::ThreadSafeQueue<Request, true> target_queue_{1};
  std::atomic<bool> quit_{false};
  std::thread plan_thread_;
};

std::unique_ptr<ModeSession> make_session(Mode mode, const std::string & config_path,
                                          io::Camera & camera, io::Gimbal & gimbal,
                                          auto_aim::Solver & solver,
                                          tools::FoxgloveVisualizer & foxglove)
{
  switch (mode) {
    case Mode::auto_aim:
      return std::make_unique<AutoAimSession>(config_path, camera, gimbal, solver, foxglove);
    case Mode::small_buff:
      return std::make_unique<BuffSession>(config_path, false, camera, gimbal, foxglove);
    case Mode::big_buff:
      return std::make_unique<BuffSession>(config_path, true, camera, gimbal, foxglove);
  }
  return std::make_unique<AutoAimSession>(config_path, camera, gimbal, solver, foxglove);
}

}  // namespace

ModeRuntime::ModeRuntime(std::string config_path, ModeReader mode_reader)
: config_path_(std::move(config_path)), mode_reader_(std::move(mode_reader))
{
}

int ModeRuntime::run()
{
  tools::Exiter exiter;
  tools::Recorder recorder;
  io::Gimbal gimbal(config_path_);
  io::Camera camera(config_path_);
  auto_aim::Solver solver(config_path_);
  tools::FoxgloveVisualizer foxglove(solver, config_path_);

  auto mode = mode_reader_(gimbal);
  auto session = make_session(mode, config_path_, camera, gimbal, solver, foxglove);
  while (!exiter.exit()) {
    const auto next_mode = mode_reader_(gimbal);
    if (next_mode != mode) {
      session.reset();
      mode = next_mode;
      session = make_session(mode, config_path_, camera, gimbal, solver, foxglove);
    }

    tools::ProcessedFrame processed;
    if (!session->next(processed)) break;
    recorder.record(processed.snapshot);
    foxglove.publish(std::move(processed.snapshot));
  }

  session.reset();
  return 0;
}

}  // namespace standard
