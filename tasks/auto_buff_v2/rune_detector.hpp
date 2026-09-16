#ifndef AUTO_BUFF_V2__RUNE_DETECTOR_HPP
#define AUTO_BUFF_V2__RUNE_DETECTOR_HPP

#include "rune.hpp"

namespace auto_buff_v2
{
class RuneDetector
{
public:
  struct Config
  {
    double fx = 0;
    double fy = 0;
    bool enemy_red = false;
    double min_distance = 1.7;
    double max_distance = 5.0;
    double max_perspective = 60.0;
    double active_threshold = 0.2;
    double match_threshold = 0.5;
  } config;

  RuneElements detect(const cv::Mat & image) const;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_DETECTOR_HPP
