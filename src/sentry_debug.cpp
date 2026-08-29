#include <fmt/core.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/cboard.hpp"
#include "io/http_sender.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/omniperception/decider.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  io::HttpSender http_sender(config_path);
  io::CBoard cboard(config_path);
  io::Camera camera(config_path);
  io::Camera back_camera("configs/camera.yaml");
  io::USBCamera usbcam1("video0", config_path);
  io::USBCamera usbcam2("video2", config_path);

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  omniperception::Decider decider(config_path);

  cv::Mat img;

  std::chrono::steady_clock::time_point timestamp;
  io::Command last_command;

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = cboard.imu_at(timestamp - 1ms);
    // recorder.record(img, q, timestamp);

    /// 自瞄核心逻辑
    solver.set_R_gimbal2world(q);

    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto detections = yolo.detect_result(img);
    auto & armors = detections.armors;

    decider.armor_filter(armors);

    auto targets = tracker.track(detections, timestamp);

    io::Command command{false, false, 0, 0};

    /// 全向感知逻辑
    if (tracker.state() == "lost")
      command = decider.decide(yolo, gimbal_pos, usbcam1, usbcam2, back_camera);
    else
      command = aimer.aim(targets, timestamp, cboard.bullet_speed, cboard.shoot_mode);

    /// 发射逻辑
    command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

    cboard.send(command);

    /// HTTP通信
    Eigen::Vector4d target_info = decider.get_target_info(armors, targets);

    http_sender.send(target_info);

    /// debug
    tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }  //always left
      solver.solve(armor);
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    }

    if (!targets.empty()) {
      auto target = targets.front();

      // 当前帧target更新后
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});
      else
        tools::draw_points(img, image_points, {255, 0, 0});

      // 观测器内部数据
      const auto state = target.state();
      data["x"] = state.center_x();
      data["vx"] = state.velocity_x();
      data["y"] = state.center_y();
      data["vy"] = state.velocity_y();
      data["z"] = state.center_z();
      data["vz"] = state.velocity_z();
      data["a"] = state.yaw() * 57.3;
      data["w"] = state.yaw_rate();
      data["r"] = state.radius();
      data["l"] = state.radius_diff();
      data["h"] = state.height_diff();
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.diagnostics().residual_yaw;
      data["residual_pitch"] = target.diagnostics().residual_pitch;
      data["residual_distance"] = target.diagnostics().residual_distance;
      data["residual_angle"] = target.diagnostics().residual_angle;
      data["nis"] = target.diagnostics().nis;
      data["nees"] = target.diagnostics().nees;
      data["nis_fail"] = target.diagnostics().nis_fail;
      data["nees_fail"] = target.diagnostics().nees_fail;
      data["recent_nis_failures"] = target.diagnostics().recent_nis_failures;
    }

    // 云台响应情况
    data["gimbal_yaw"] = gimbal_pos[0] * 57.3;
    data["gimbal_pitch"] = -gimbal_pos[1] * 57.3;
    data["shootmode"] = cboard.shoot_mode;
    if (command.control) {
      data["cmd_yaw"] = command.yaw * 57.3;
      data["cmd_pitch"] = command.pitch * 57.3;
      data["cmd_shoot"] = command.shoot;
    }

    data["bullet_speed"] = cboard.bullet_speed;

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }
  return 0;
}
