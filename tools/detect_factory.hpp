#ifndef SRC__DETECT_FACTORY_HPP
#define SRC__DETECT_FACTORY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/armor.hpp"

std::function<auto_aim::DetectionResult(const cv::Mat &, int)> create_detector_result(
  const std::string & config_path);

bool detector_debug_window_enabled(const YAML::Node & yaml);

#endif  // SRC__DETECT_FACTORY_HPP
