#ifndef AUTO_BUFF_V2__RUNE_DETECTOR_HPP
#define AUTO_BUFF_V2__RUNE_DETECTOR_HPP

#include <vector>

#include "buff_config.hpp"
#include "rune.hpp"

namespace auto_buff_v2
{
struct RuneContourDebug
{
  cv::Point2f center;
  double radius = 0;
  double area = 0;
  bool radius_pass = false;
  bool icon_area_pass = false;
};

struct RuneDetectorDebug
{
  double min_radius = 0;
  double max_radius = 0;
  double min_icon_area = 0;
  double max_icon_area = 0;
  std::vector<RuneContourDebug> candidates;
};

class RuneDetector
{
public:
  struct Config : BuffConfig::Detector
  {
    bool enemy_red = false;
    double max_perspective = 60.0;
  } config;

  RuneElements detect(const cv::Mat & image, RuneDetectorDebug * debug = nullptr) const;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_DETECTOR_HPP
