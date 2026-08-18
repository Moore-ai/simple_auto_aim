#ifndef AUTO_AIM__DETECTION_RESULT_HPP
#define AUTO_AIM__DETECTION_RESULT_HPP

namespace YAML
{
class Node;
}

namespace auto_aim
{

// Controls optional per-frame data production. Armor detection is always run;
// independent lightbars are collected only when an observation path needs them.
struct DetectionOptions
{
  bool collect_lightbars = false;

  static DetectionOptions from_yaml(const YAML::Node & yaml);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__DETECTION_RESULT_HPP
