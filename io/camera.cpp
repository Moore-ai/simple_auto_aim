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
    auto gain = tools::read<double>(yaml, "gain");
    auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, vid_pid);
  }

  else {
    throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);

  if (auto_exposure_) {
    auto_exposure_->update(
      img, camera_->get_exposure_us(), std::chrono::steady_clock::now(),
      [this](double exposure_us) { camera_->set_exposure_us(exposure_us); });
  }
}

}  // namespace io
