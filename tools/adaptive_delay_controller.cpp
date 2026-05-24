#include "adaptive_delay_controller.hpp"

#include <algorithm>

namespace tools
{

void AdaptiveDelayController::init(const Config & config)
{
  config_ = config;
  config_.min_delay = std::min(config_.min_delay, config_.max_delay);
  adaptive_delay_ = std::clamp(config_.initial_delay, config_.min_delay, config_.max_delay);
  no_fire_count_ = 0;
}

void AdaptiveDelayController::reset()
{
  adaptive_delay_ = config_.initial_delay;
  no_fire_count_ = 0;
}

void AdaptiveDelayController::update(bool fire_advice, double v_linear, double v_angular)
{
  if (!config_.enable) return;

  if (fire_advice) {
    adaptive_delay_ -= config_.add_step;
    adaptive_delay_ = std::max(adaptive_delay_, config_.min_delay);
    no_fire_count_ = 0;
  } else {
    no_fire_count_++;
    if (no_fire_count_ >= config_.fire_wait_threshold) {
      double w_linear = std::min(v_linear / std::max(config_.max_linear_speed, 1e-6), 1.0);
      double w_angular = std::min(v_angular / std::max(config_.max_angular_speed, 1e-6), 1.0);
      double speed_factor = std::max(w_linear, w_angular);

      adaptive_delay_ *= 1.0 + (config_.mul_factor - 1.0) * speed_factor;
      adaptive_delay_ = std::min(adaptive_delay_, config_.max_delay);
    }
  }
}

}  // namespace tools
