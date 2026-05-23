#include "detect_factory.hpp"

#include <memory>

#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

std::function<std::list<auto_aim::Armor>(const cv::Mat &, int)> create_detector(
  const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto detect_method = yaml["detect_method"] ? yaml["detect_method"].as<std::string>() : "yolo";

  if (detect_method == "traditional") {
    tools::logger()->info("[DetectFactory] Using traditional detector");
    auto detector = std::make_shared<auto_aim::Detector>(config_path, false);
    return [detector](const cv::Mat & img, int fc) { return detector->detect(img, fc); };
  }

  tools::logger()->info("[DetectFactory] Using YOLO detector ({})", detect_method);
  auto yolo = std::make_shared<auto_aim::YOLO>(config_path, true);
  return [yolo](const cv::Mat & img, int fc) { return yolo->detect(img, fc); };
}
