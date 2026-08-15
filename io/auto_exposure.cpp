#include "auto_exposure.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace io
{
AutoExposure::AutoExposure(AutoExposureConfig config) : config_(config)
{
  if (!config_.enable) return;

  if (!std::isfinite(config_.target_brightness) || !std::isfinite(config_.tolerance) ||
      !std::isfinite(config_.step_gain) || !std::isfinite(config_.decay_step) ||
      !std::isfinite(config_.exposure_min) || !std::isfinite(config_.exposure_max) ||
      config_.tolerance < 0.0 || config_.step_gain <= 0.0 || config_.decay_step < 0.0 ||
      config_.exposure_min < 0.0 || config_.exposure_min > config_.exposure_max ||
      config_.control_interval.count() <= 0) {
    throw std::invalid_argument("Invalid auto_exposure configuration");
  }
}

void AutoExposure::update(
  const cv::Mat & image, double current_exposure_us,
  std::chrono::steady_clock::time_point now, const std::function<void(double)> & set_exposure)
{
  if (!config_.enable || image.empty()) return;

  if (last_control_time_ && now - *last_control_time_ < config_.control_interval) return;
  last_control_time_ = now;

  if (!std::isfinite(current_exposure_us) || current_exposure_us <= 0.0) return;

  cv::Mat gray;
  if (image.channels() == 1)
    gray = image;
  else
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

  const auto current_brightness = cv::mean(gray)[0];
  const auto diff = current_brightness - config_.target_brightness;
  const auto adjustment = std::abs(diff) <= config_.tolerance
                            ? config_.decay_step
                            : diff * config_.step_gain;
  const auto candidate = std::clamp(
    current_exposure_us - adjustment, config_.exposure_min, config_.exposure_max);
  const auto previous = last_set_exposure_us_.value_or(current_exposure_us);

  if (set_exposure && std::abs(candidate - previous) > 10.0) {
    set_exposure(candidate);
    last_set_exposure_us_ = candidate;
  }
}
}  // namespace io
