#ifndef TOOLS__DETECT_FACTORY_HPP
#define TOOLS__DETECT_FACTORY_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_aim/armor.hpp"

namespace tools
{
class DetectionBackend
{
public:
  virtual ~DetectionBackend() = default;
  virtual auto_aim::DetectionResult detect(const cv::Mat & image, int frame_count) = 0;
};

std::unique_ptr<DetectionBackend> create_detector_result(const std::string & config_path);

bool detector_debug_window_enabled(const YAML::Node & yaml);

}  // namespace tools

#endif  // TOOLS__DETECT_FACTORY_HPP
