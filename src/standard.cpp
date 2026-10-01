#include <iostream>
#include <opencv2/opencv.hpp>
#include <utility>

#include "io/gimbal/gimbal.hpp"
#include "mode_runtime.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }"
  "{mode           |0| 0=自瞄普通目标和前哨站，1=小符，2=大符 }"
  "{target-color   | | 必填：red 或 blue 指定目标颜色，none 表示由下位机决定 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const int mode_value = cli.get<int>("mode");
  if (!cli.check() || !standard::is_valid_mode(mode_value)) {
    cli.printErrors();
    std::cerr << "mode 必须为 0（自瞄）、1（小符）或 2（大符）\n";
    return 2;
  }
  const auto mode = standard::mode_from_value(mode_value);
  std::optional<io::InfantryEnemyColor> target_color_override;
  const auto color = cli.get<std::string>("target-color");
  if (color == "red") {
    target_color_override = io::InfantryEnemyColor::red;
  } else if (color == "blue") {
    target_color_override = io::InfantryEnemyColor::blue;
  } else if (color != "none") {
    std::cerr << "target-color 为必填参数，必须为 red、blue 或 none\n";
    return 2;
  }
  const standard::ModeRuntime::ModeReader mode_reader =
    [mode](const io::Gimbal &) { return mode; };
  return standard::ModeRuntime(std::move(config_path), mode_reader, target_color_override).run();
}
