#ifndef TOOLS__ADAPTIVE_DELAY_CONTROLLER_HPP
#define TOOLS__ADAPTIVE_DELAY_CONTROLLER_HPP

#include <cmath>

namespace tools
{

class AdaptiveDelayController
{
public:
  struct Config
  {
    bool enable = false;
    double initial_delay = 0.015;
    double min_delay = 0.0;
    double max_delay = 0.10;
    double add_step = 0.005;
    double mul_factor = 1.2;
    int fire_wait_threshold = 10;
    double max_linear_speed = 3.0;
    double max_angular_speed = 10.0;
  };

  AdaptiveDelayController() = default;

  void init(const Config & config);

  void reset();

  void update(bool fire_advice, double v_linear, double v_angular);

  double getDelay() const { return adaptive_delay_; }

  int getNoFireCount() const { return no_fire_count_; }

private:
  Config config_;
  double adaptive_delay_{0.0};
  int no_fire_count_{0};
};

}  // namespace tools

#endif  // TOOLS__ADAPTIVE_DELAY_CONTROLLER_HPP
