#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include "armor.hpp"
#include "lightbar_detector.hpp"

namespace auto_aim
{
class YOLOBase
{
public:
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

class YOLO
{
public:
  YOLO(const std::string & config_path, bool debug = true);

  DetectionResult detect_result(const cv::Mat & img, int frame_count = -1);
  std::list<Lightbar> detect_lightbars(const cv::Mat & img) const;
  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_;
  std::unique_ptr<LightbarDetector> lightbar_detector_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP
