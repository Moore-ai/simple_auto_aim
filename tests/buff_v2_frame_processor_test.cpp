#include <cassert>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "tasks/auto_buff_v2/buff_frame_processor.hpp"

int main()
{
  auto_buff_v2::BuffConfig::Camera camera;
  camera.camera_matrix = (cv::Mat_<double>(3, 3) << 1000, 0, 200, 0, 1000, 200, 0, 0, 1);
  camera.distort_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  auto_buff_v2::BuffConfig::Detector detector;
  detector.fx = 1000;
  detector.fy = 1000;
  detector.min_distance = 2;
  detector.max_distance = 5;
  auto_buff_v2::BuffConfig::Model model;
  auto_buff_v2::RuneModel rune_model(camera, model, false);
  auto_buff_v2::BuffFrameProcessor processor(rune_model, detector);

  cv::Mat image = cv::Mat::zeros(400, 400, CV_8UC3);
  cv::circle(image, {200, 200}, 42, {255, 0, 0}, cv::FILLED);
  const auto timestamp = std::chrono::steady_clock::now();
  tools::FrameFacts facts{timestamp, image, Eigen::Quaterniond::Identity(), {}};
  facts.received.state.mode = static_cast<std::uint8_t>(io::InfantryEnemyColor::blue);

  const auto processed = processor.process(facts);

  assert(processed.buff_target == std::nullopt);
  assert(processed.snapshot.timestamp == timestamp);
  assert(processed.snapshot.image.data != image.data);
  assert(processed.snapshot.buff_debug.detections.size() == 1);
  assert(processed.snapshot.buff_debug.detections.front().label.rfind("B: ", 0) == 0);
  assert(processed.targets.empty());
}
