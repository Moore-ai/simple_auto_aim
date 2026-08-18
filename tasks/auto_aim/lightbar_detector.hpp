#ifndef AUTO_AIM__LIGHTBAR_DETECTOR_HPP
#define AUTO_AIM__LIGHTBAR_DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "armor.hpp"

namespace auto_aim
{

class LightbarDetector
{
public:
  explicit LightbarDetector(const std::string & config_path);

  std::list<Lightbar> detect(const cv::Mat & bgr_img) const;

private:
  double threshold_;
  double max_angle_error_;
  double min_lightbar_ratio_;
  double max_lightbar_ratio_;
  double min_lightbar_length_;

  bool check_geometry(const Lightbar & lightbar) const;
  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__LIGHTBAR_DETECTOR_HPP
