#include "detect_factory.hpp"

#include <memory>

#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/yaml.hpp"

std::function<auto_aim::DetectionResult(const cv::Mat &, int)> create_detector_result(
  const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto detect_method = yaml["detect_method"] ? yaml["detect_method"].as<std::string>() : "yolo";

  if (detect_method == "traditional") {
    auto detector = std::make_shared<auto_aim::Detector>(config_path, false);
    return [detector](const cv::Mat & img, int fc) { return detector->detect_result(img, fc); };
  }

  auto yolo = std::make_shared<auto_aim::YOLO>(config_path, true);
  return [yolo](const cv::Mat & img, int fc) { return yolo->detect_result(img, fc); };
}
