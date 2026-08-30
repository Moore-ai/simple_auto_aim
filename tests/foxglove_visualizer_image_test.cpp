#include <cassert>

#include <opencv2/core.hpp>

#include "tools/foxglove_visualizer.hpp"

int main()
{
  cv::Mat input(1, 3, CV_8UC3);
  input.at<cv::Vec3b>(0, 0) = {1, 2, 3};
  input.at<cv::Vec3b>(0, 1) = {4, 5, 6};
  input.at<cv::Vec3b>(0, 2) = {7, 8, 9};

  const auto output = tools::detail::prepare_image_for_publish(input);
  assert(cv::norm(input, output, cv::NORM_INF) == 0.0);
  return 0;
}
