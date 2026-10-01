#include <iostream>
#include <optional>

#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_buff_v2/buff_config.hpp"
#include "tasks/auto_buff_v2/frame_runtime.hpp"
#include "tasks/auto_buff_v2/rune_model.hpp"
#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | 与主链路相同的 yaml 配置文件路径}"
  "{mode           |1| 1=小符，2=大符}"
  "{target-color   | | 必填：red、blue 或 none（由下位机决定）}";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const int mode = cli.get<int>("mode");
  const auto color = cli.get<std::string>("target-color");
  if (!cli.check() || (mode != 1 && mode != 2)) {
    cli.printErrors();
    std::cerr << "mode 必须为 1（小符）或 2（大符）\n";
    return 2;
  }
  std::optional<io::InfantryEnemyColor> target_color_override;
  if (color == "red") {
    target_color_override = io::InfantryEnemyColor::red;
  } else if (color == "blue") {
    target_color_override = io::InfantryEnemyColor::blue;
  } else if (color != "none") {
    std::cerr << "target-color 为必填参数，必须为 red、blue 或 none\n";
    return 2;
  }

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  auto_aim::Solver solver(config_path);
  tools::FoxgloveVisualizer foxglove(solver, config_path);
  const auto config = auto_buff_v2::BuffConfig::load(config_path);
  auto_buff_v2::RuneModel model(config.camera, config.model, mode == 2);
  auto_buff_v2::FrameRuntime runtime(
    camera, gimbal, model, config.detector, target_color_override, true);
  // 复用主链路的帧处理和观测机制；检测调试入口不启动规划与发射线程。
  while (!exiter.exit()) {
    tools::ProcessedFrame processed;
    if (!runtime.next(processed)) break;
    foxglove.publish(std::move(processed.snapshot));
  }
  return 0;
}
