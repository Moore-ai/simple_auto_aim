#ifndef OMNIPERCEPTION__DECIDER_HPP
#define OMNIPERCEPTION__DECIDER_HPP

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <iostream>
#include <list>

#include "detection.hpp"
#include "io/camera.hpp"
#include "io/command.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/yolo.hpp"

namespace omniperception
{
class Decider
{
public:
  Decider(const std::string & config_path);

  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::USBCamera & usbcam1,
    io::USBCamera & usbcam2, io::Camera & back_cammera);

  io::Command decide(
    auto_aim::YOLO & yolo, const Eigen::Vector3d & gimbal_pos, io::Camera & back_cammera);

  io::Command decide(const std::vector<DetectionResult> & detection_queue);

  Eigen::Vector2d delta_angle(
    const std::list<auto_aim::Armor> & armors, const std::string & camera);

  bool armor_filter(std::list<auto_aim::Armor> & armors);

  void filter(std::vector<DetectionResult> & detection_queue);

  Eigen::Vector4d get_target_info(
    const std::list<auto_aim::Armor> & armors, const std::list<auto_aim::Target> & targets);

private:
  int img_width_;
  int img_height_;
  double fov_h_, new_fov_h_;
  double fov_v_, new_fov_v_;
  int count_;

  auto_aim::Color enemy_color_;
  auto_aim::YOLO detector_;
};

}  // namespace omniperception

#endif
