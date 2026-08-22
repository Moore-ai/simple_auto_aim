#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"

int main()
{
  const auto video_path =
    (std::filesystem::temp_directory_path() / "sp_vision_video_camera_test.avi").string();
  const auto config_path =
    (std::filesystem::temp_directory_path() / "sp_vision_video_camera_test.yaml").string();
  cv::VideoWriter writer(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, {8, 8});
  if (!writer.isOpened()) {
    std::cerr << "failed to create test video\n";
    return 1;
  }
  writer.write(cv::Mat(cv::Size(8, 8), CV_8UC3, cv::Scalar(0, 0, 255)));
  writer.write(cv::Mat(cv::Size(8, 8), CV_8UC3, cv::Scalar(0, 255, 0)));
  writer.release();

  std::ofstream config(config_path);
  config << "virtual_camera:\n"
         << "  enable: true\n"
         << "  video_path: " << video_path << '\n';
  config.close();

  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point first_timestamp;
  std::chrono::steady_clock::time_point second_timestamp;

  if (!camera.read(img, first_timestamp) || img.empty()) return 2;
  if (!camera.read(img, second_timestamp) || img.empty()) return 3;
  if (second_timestamp < first_timestamp) return 4;
  if (camera.read(img, second_timestamp) || !img.empty()) return 5;

  std::filesystem::remove(video_path);
  std::filesystem::remove(config_path);
  return 0;
}
