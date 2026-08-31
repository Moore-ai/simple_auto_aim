#include <fmt/core.h>

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "calibration/pattern.hpp"
#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

bool should_save_capture(int event) { return event == cv::EVENT_LBUTTONDOWN; }

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(const std::string & config_path, const std::string & output_folder)
{
  auto yaml = YAML::LoadFile(config_path);
  cv::Size pattern_size(yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>());
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int count = 0;
  bool save_requested = false;
  cv::namedWindow("Press s to save, q to quit");
  cv::setMouseCallback(
    "Press s to save, q to quit",
    [](int event, int, int, int, void * userdata) {
      if (should_save_capture(event)) *static_cast<bool *>(userdata) = true;
    },
    &save_requested);
  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = gimbal.q(timestamp);

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

    std::vector<cv::Point2f> corners_2d;
    auto success = calibration::find_chessboard_pattern(img, pattern_size, corners_2d);
    cv::drawChessboardCorners(img_with_ypr, pattern_size, corners_2d, success);  // 显示识别结果
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按“s”或左键点击窗口保存图片和对应四元数，按“q”退出程序
    save_requested = false;
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's' && !save_requested)
      continue;

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  // 离开该作用域时，camera和gimbal会自动关闭
}

#ifndef CAPTURE_INPUT_TEST
int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  auto yaml = YAML::LoadFile(config_path);
  tools::logger()->info(
    "标定板尺寸为{}列{}行", yaml["pattern_cols"].as<int>(), yaml["pattern_rows"].as<int>());
  // 主循环，保存图片和对应四元数
  capture_loop(config_path, output_folder);

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
#endif
