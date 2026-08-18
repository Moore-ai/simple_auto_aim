#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

#include "yolos/yolo11.hpp"
#include "yolos/yolov5.hpp"
#include "yolos/yolov8.hpp"

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path, bool debug)
  : lightbar_detector_(std::make_unique<LightbarDetector>(config_path))
{
  auto yaml = YAML::LoadFile(config_path);
  detection_options_ = DetectionOptions::from_yaml(yaml);
  auto yolo_name = yaml["yolo_name"].as<std::string>();

  if (yolo_name == "yolov8") {
    yolo_ = std::make_unique<YOLOV8>(config_path, debug);
  }

  else if (yolo_name == "yolo11") {
    yolo_ = std::make_unique<YOLO11>(config_path, debug);
  }

  else if (yolo_name == "yolov5") {
    yolo_ = std::make_unique<YOLOV5>(config_path, debug);
  }

  else {
    throw std::runtime_error("Unknown yolo name: " + yolo_name + "!");
  }
}

DetectionResult YOLO::detect_result(const cv::Mat & img, int frame_count)
{
  auto armors = yolo_->detect(img, frame_count);
  if (!detection_options_.collect_lightbars) return {std::move(armors), {}};
  DetectionResult result{std::move(armors), {}};
  result.lightbars_collected = true;
  result.lightbars = detect_lightbars(img);
  return result;
}

std::list<Lightbar> YOLO::detect_lightbars(const cv::Mat & img) const
{
  return lightbar_detector_->detect(img);
}

std::list<Armor> YOLO::detect(const cv::Mat & img, int frame_count)
{
  return yolo_->detect(img, frame_count);
}

std::list<Armor> YOLO::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return yolo_->postprocess(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim
