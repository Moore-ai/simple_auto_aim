#ifndef AUTO_BUFF_V2__RUNE_DETECTOR_HPP
#define AUTO_BUFF_V2__RUNE_DETECTOR_HPP

#include "buff_config.hpp"
#include "rune.hpp"

namespace auto_buff_v2
{
class RuneDetector
{
public:
  struct Config : BuffConfig::Detector
  {
    bool enemy_red = false;
    double max_perspective = 60.0;
  } config;

  RuneElements detect(const cv::Mat & image) const;
};
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_DETECTOR_HPP
