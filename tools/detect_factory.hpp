#ifndef SRC__DETECT_FACTORY_HPP
#define SRC__DETECT_FACTORY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"

std::function<auto_aim::DetectionResult(const cv::Mat &, int)> create_detector_result(
  const std::string & config_path);

#endif  // SRC__DETECT_FACTORY_HPP
