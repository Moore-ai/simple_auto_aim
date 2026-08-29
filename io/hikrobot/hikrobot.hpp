#ifndef IO__HIKROBOT_HPP
#define IO__HIKROBOT_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"

namespace io
{
struct HikRobotConfig
{
  double exposure_us{0.0};
  double gain{0.0};
  int device_index{0};
  int image_width{0};
  int image_height{0};
  double fps{150.0};
};

HikRobotConfig load_hikrobot_config(const YAML::Node & yaml);

class HikRobot : public CameraBase
{
public:
  explicit HikRobot(HikRobotConfig config);
  ~HikRobot() override;
  bool read(
    cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
    std::chrono::milliseconds timeout) override;
  double get_exposure_us() const override;
  void set_exposure_us(double exposure_us) override;

private:
  HikRobotConfig config_;
  std::atomic<double> exposure_us_;
  void * handle_{nullptr};
  bool grabbing_{false};
  std::vector<unsigned char> frame_buffer_;
  std::vector<unsigned char> bgr_buffer_;

  bool open();
  void close();
  bool configure();
  bool prepare_frame_buffer();
  bool set_float_value(const std::string & name, double value);
  bool set_int_value(const std::string & name, unsigned int value);
  bool set_enum_value(const std::string & name, unsigned int value);
  bool set_enum_value(const std::string & name, const std::string & value);
};

}  // namespace io

#endif  // IO__HIKROBOT_HPP
