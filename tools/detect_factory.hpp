#ifndef SRC__DETECT_FACTORY_HPP
#define SRC__DETECT_FACTORY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <string>

#include "tasks/auto_aim/armor.hpp"

/// 根据配置文件的 detect_method 字段创建对应的检测器
/// "yolo" → YOLOv5/v8/v11（默认）
/// "traditional" → 纯传统灯条检测
std::function<std::list<auto_aim::Armor>(const cv::Mat &, int)> create_detector(
  const std::string & config_path);

#endif  // SRC__DETECT_FACTORY_HPP
