#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "io/auto_exposure.hpp"

namespace
{
using Clock = std::chrono::steady_clock;

void require(bool condition, const char * message)
{
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

io::AutoExposureConfig config()
{
  return {
    true,
    25.0,
    3.0,
    15.0,
    1.0,
    200.0,
    3500.0,
    std::chrono::milliseconds(300),
  };
}

cv::Mat image(double brightness)
{
  return cv::Mat(2, 2, CV_8UC3, cv::Scalar(brightness, brightness, brightness));
}

void test_brightness_feedback()
{
  auto exposure = io::AutoExposure(config());
  double requested = 0.0;

  exposure.update(image(0.0), 1000.0, Clock::time_point{}, [&](double value) { requested = value; });
  require(requested == 1375.0, "dark image should increase exposure");

  requested = 0.0;
  exposure.update(
    image(125.0), 1000.0, Clock::time_point{} + std::chrono::milliseconds(300),
    [&](double value) { requested = value; });
  require(requested == 200.0, "bright image should decrease and clamp exposure");
}

void test_tolerance_decay()
{
  auto tolerance_config = config();
  tolerance_config.decay_step = 15.0;
  auto exposure = io::AutoExposure(tolerance_config);
  double requested = 0.0;

  exposure.update(image(25.0), 1000.0, Clock::time_point{}, [&](double value) { requested = value; });
  require(requested == 985.0, "brightness in tolerance should decay exposure");
}

void test_control_interval()
{
  auto exposure = io::AutoExposure(config());
  int requests = 0;
  auto set = [&](double) { ++requests; };

  exposure.update(image(0.0), 1000.0, Clock::time_point{}, set);
  exposure.update(image(0.0), 1000.0, Clock::time_point{} + std::chrono::milliseconds(299), set);
  require(requests == 1, "exposure should not update before control interval");

  exposure.update(image(0.0), 1375.0, Clock::time_point{} + std::chrono::milliseconds(300), set);
  require(requests == 2, "exposure should update at control interval");
}

void test_limits_threshold_and_zero()
{
  auto upper = io::AutoExposure(config());
  double requested = 0.0;
  upper.update(image(0.0), 3400.0, Clock::time_point{}, [&](double value) { requested = value; });
  require(requested == 3500.0, "exposure should clamp to upper limit");

  auto lower = io::AutoExposure(config());
  requested = 0.0;
  lower.update(image(125.0), 300.0, Clock::time_point{}, [&](double value) { requested = value; });
  require(requested == 200.0, "exposure should clamp to lower limit");

  auto threshold_config = config();
  threshold_config.decay_step = 10.0;
  auto threshold = io::AutoExposure(threshold_config);
  int requests = 0;
  threshold.update(image(25.0), 1000.0, Clock::time_point{}, [&](double) { ++requests; });
  require(requests == 0, "a ten microsecond change should not be requested");

  auto zero = io::AutoExposure(config());
  zero.update(image(0.0), 0.0, Clock::time_point{}, [&](double) { ++requests; });
  require(requests == 0, "zero exposure should not use proportional control");
}

void test_disabled()
{
  auto disabled_config = config();
  disabled_config.enable = false;
  auto exposure = io::AutoExposure(disabled_config);
  int requests = 0;
  exposure.update(image(0.0), 1000.0, Clock::time_point{}, [&](double) { ++requests; });
  require(requests == 0, "disabled exposure controller should not request changes");
}
}  // namespace

int main()
{
  test_brightness_feedback();
  test_tolerance_decay();
  test_control_interval();
  test_limits_threshold_and_zero();
  test_disabled();
  return 0;
}
