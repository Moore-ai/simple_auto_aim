#include "detect_factory.hpp"

#include <memory>

#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/yaml.hpp"

namespace tools
{
bool detector_debug_window_enabled(const YAML::Node & yaml)
{
  return yaml["debug_window"] ? yaml["debug_window"].as<bool>() : true;
}

namespace
{
class TraditionalDetectorAdapter final : public DetectionBackend
{
public:
  TraditionalDetectorAdapter(const std::string & config_path, bool debug)
  : detector_(std::make_unique<auto_aim::Detector>(config_path, debug))
  {}

  auto_aim::DetectionResult detect(const cv::Mat & image, int frame_count) override
  {
    return detector_->detect_result(image, frame_count);
  }

private:
  std::unique_ptr<auto_aim::Detector> detector_;
};

class YoloDetectorAdapter final : public DetectionBackend
{
public:
  YoloDetectorAdapter(const std::string & config_path, bool debug)
  : detector_(std::make_unique<auto_aim::YOLO>(config_path, debug))
  {}

  auto_aim::DetectionResult detect(const cv::Mat & image, int frame_count) override
  {
    return detector_->detect_result(image, frame_count);
  }

private:
  std::unique_ptr<auto_aim::YOLO> detector_;
};
}  // namespace

std::unique_ptr<DetectionBackend> create_detector_result(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto detect_method = yaml["detect_method"] ? yaml["detect_method"].as<std::string>() : "yolo";
  const auto debug_window = detector_debug_window_enabled(yaml);

  if (detect_method == "traditional") {
    return std::make_unique<TraditionalDetectorAdapter>(config_path, debug_window);
  }

  return std::make_unique<YoloDetectorAdapter>(config_path, debug_window);
}

}  // namespace tools
