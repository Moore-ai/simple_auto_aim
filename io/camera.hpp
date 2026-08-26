#ifndef IO__CAMERA_HPP
#define IO__CAMERA_HPP

#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "auto_exposure.hpp"

namespace io
{
class CameraBase
{
public:
  virtual ~CameraBase() = default;
  virtual bool read(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) = 0;
  virtual double get_exposure_us() const = 0;
  virtual void set_exposure_us(double exposure_us) = 0;
};

class Camera
{
public:
  Camera(const std::string & config_path);
  bool read(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout = std::chrono::milliseconds(100));

private:
  std::unique_ptr<CameraBase> camera_;
  std::unique_ptr<cv::VideoCapture> virtual_camera_;
  std::unique_ptr<AutoExposure> auto_exposure_;
};

}  // namespace io

#endif  // IO__CAMERA_HPP
