#include <cassert>
#include <chrono>

#include "tools/frame_snapshot.hpp"

int main()
{
  const auto timestamp = std::chrono::steady_clock::now();
  cv::Mat image(2, 3, CV_8UC1, cv::Scalar(7));
  auto_aim::TrackerDebugData tracker;
  auto frame = tools::FrameSnapshot::capture(
    timestamp, image, Eigen::Quaterniond::Identity(), {}, {}, {}, auto_aim::Color::blue, tracker);

  assert(frame.timestamp == timestamp);
  assert(frame.image.size() == image.size());
  assert(frame.image.data != image.data);
  assert(frame.image.at<uint8_t>(0, 0) == 7);
  assert(frame.target_color == auto_aim::Color::blue);
  return 0;
}
