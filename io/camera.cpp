#include "camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  if (yaml["image_rotation"]) image_rotation_ = yaml["image_rotation"].as<int>();
  const auto virtual_camera_yaml = yaml["virtual_camera"];
  const auto virtual_camera_enabled =
    virtual_camera_yaml && virtual_camera_yaml["enable"] && virtual_camera_yaml["enable"].as<bool>();
  if (virtual_camera_enabled && virtual_camera_yaml["image_rotation"]) {
    image_rotation_ = virtual_camera_yaml["image_rotation"].as<int>();
  }
  if (
    image_rotation_ != 0 && image_rotation_ != 90 && image_rotation_ != -90 &&
    image_rotation_ != 180 && image_rotation_ != -180) {
    throw std::runtime_error("image_rotation must be 0, 90, -90, 180, or -180");
  }
  if (virtual_camera_enabled) {
    auto video_path = tools::read<std::string>(virtual_camera_yaml, "video_path");
    virtual_camera_ = std::make_unique<cv::VideoCapture>(video_path);
    if (!virtual_camera_->isOpened()) throw std::runtime_error("Failed to open video: " + video_path);
    return;
  }

  auto camera_name = tools::read<std::string>(yaml, "camera_name");
  auto exposure_ms = tools::read<double>(yaml, "exposure_ms");

  const auto auto_exposure_yaml = yaml["auto_exposure"];
  const auto auto_exposure_enabled =
    auto_exposure_yaml && auto_exposure_yaml["enable"] &&
    auto_exposure_yaml["enable"].as<bool>();
  if (auto_exposure_enabled) {
    auto_exposure_ = std::make_unique<AutoExposure>(AutoExposureConfig{
      true,
      tools::read<double>(auto_exposure_yaml, "target_brightness"),
      tools::read<double>(auto_exposure_yaml, "tolerance"),
      tools::read<double>(auto_exposure_yaml, "step_gain"),
      tools::read<double>(auto_exposure_yaml, "decay_step"),
      tools::read<double>(auto_exposure_yaml, "exposure_min"),
      tools::read<double>(auto_exposure_yaml, "exposure_max"),
      std::chrono::milliseconds(tools::read<int>(auto_exposure_yaml, "control_interval_ms")),
    });
  }

  if (camera_name == "mindvision") {
    auto gamma = tools::read<double>(yaml, "gamma");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  else if (camera_name == "hikrobot") {
    camera_ = std::make_unique<HikRobot>(load_hikrobot_config(yaml));
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

bool Camera::read(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  if (virtual_camera_) {
    if (!virtual_camera_->read(img)) {
      img.release();
      return false;
    }
    timestamp = std::chrono::steady_clock::now();
  } else if (!camera_->read(img, timestamp, timeout)) {
    img.release();
    return false;
  }

  switch (image_rotation_) {
    case 90:
      cv::rotate(img, img, cv::ROTATE_90_CLOCKWISE);
      break;
    case -90:
      cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    case 180:
    case -180:
      cv::rotate(img, img, cv::ROTATE_180);
      break;
  }

  if (auto_exposure_) {
    auto_exposure_->update(
      img, camera_->get_exposure_us(), std::chrono::steady_clock::now(),
      [this](double exposure_us) { camera_->set_exposure_us(exposure_us); });
  }
  return true;
}

}  // namespace io
