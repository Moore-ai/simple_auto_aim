#ifndef IO__AUTO_EXPOSURE_HPP
#define IO__AUTO_EXPOSURE_HPP

#include <chrono>
#include <functional>
#include <optional>

#include <opencv2/opencv.hpp>

namespace io
{
struct AutoExposureConfig
{
  bool enable;
  double target_brightness;
  double tolerance;
  double step_gain;
  double decay_step;
  double exposure_min;
  double exposure_max;
  std::chrono::milliseconds control_interval;
};

class AutoExposure
{
public:
  explicit AutoExposure(AutoExposureConfig config);

  void update(
    const cv::Mat & image, double current_exposure_us,
    std::chrono::steady_clock::time_point now, const std::function<void(double)> & set_exposure);

private:
  AutoExposureConfig config_;
  std::optional<std::chrono::steady_clock::time_point> last_control_time_;
  std::optional<double> last_set_exposure_us_;
};
}  // namespace io

#endif  // IO__AUTO_EXPOSURE_HPP
