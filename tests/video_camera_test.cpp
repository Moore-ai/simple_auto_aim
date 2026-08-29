#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"

int main()
{
  const auto video_path =
    (std::filesystem::temp_directory_path() / "simple_auto_aim_video_camera_test.avi").string();
  const auto config_path =
    (std::filesystem::temp_directory_path() / "simple_auto_aim_video_camera_test.yaml").string();
  cv::VideoWriter writer(video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, {8, 6});
  if (!writer.isOpened()) {
    std::cerr << "failed to create test video\n";
    return 1;
  }
  cv::Mat frame(cv::Size(8, 6), CV_8UC3, cv::Scalar::all(0));
  frame(cv::Rect(0, 0, 2, 2)) = cv::Scalar(0, 0, 255);
  frame(cv::Rect(6, 0, 2, 2)) = cv::Scalar(0, 255, 0);
  frame(cv::Rect(0, 4, 2, 2)) = cv::Scalar(255, 0, 0);
  frame(cv::Rect(6, 4, 2, 2)) = cv::Scalar(0, 255, 255);
  for (int i = 0; i < 6; ++i) writer.write(frame);
  writer.release();

  const auto is_dominant = [](const cv::Vec3b & pixel, int channel) {
    return pixel[channel] > pixel[(channel + 1) % 3] + 80 &&
           pixel[channel] > pixel[(channel + 2) % 3] + 80;
  };
  struct RotationCase
  {
    int degrees;
    cv::Size size;
    cv::Point red;
    cv::Point green;
    cv::Point blue;
    cv::Point yellow;
  };
  const std::vector<RotationCase> cases{
    {90, {6, 8}, {5, 0}, {5, 6}, {1, 0}, {1, 6}},
    {-90, {6, 8}, {0, 7}, {0, 1}, {4, 7}, {4, 1}},
    {180, {8, 6}, {7, 5}, {1, 5}, {7, 1}, {1, 1}},
    {-180, {8, 6}, {7, 5}, {1, 5}, {7, 1}, {1, 1}},
  };
  for (const auto & rotation : cases) {
    std::ofstream config(config_path);
    config << "image_rotation: " << rotation.degrees << '\n'
           << "virtual_camera:\n"
           << "  enable: true\n"
           << "  video_path: " << video_path << '\n';
    config.close();

    io::Camera camera(config_path);
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
    if (!camera.read(img, timestamp) || img.empty() || img.size() != rotation.size) return 2;
    if (!is_dominant(img.at<cv::Vec3b>(rotation.red), 2)) return 3;
    if (!is_dominant(img.at<cv::Vec3b>(rotation.green), 1)) return 4;
    if (!is_dominant(img.at<cv::Vec3b>(rotation.blue), 0)) return 5;
    const auto yellow_pixel = img.at<cv::Vec3b>(rotation.yellow);
    if (yellow_pixel[1] <= yellow_pixel[0] + 80 || yellow_pixel[2] <= yellow_pixel[0] + 80) return 6;
  }

  std::filesystem::remove(video_path);
  std::filesystem::remove(config_path);
  return 0;
}
