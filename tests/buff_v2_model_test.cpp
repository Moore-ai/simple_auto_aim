#include <cassert>
#include <cmath>

#include "tasks/auto_buff_v2/rune_energy_fitter.hpp"
#include "tasks/auto_buff_v2/rune_model.hpp"

int main()
{
  auto_buff_v2::RuneState small;
  small.rotation_speed = 1.2;
  small.rotation_angle = 0.1;
  small.transition(0.5);
  assert(std::abs(small.rotation_angle - 0.7) < 1e-9);

  auto_buff_v2::RuneState big;
  big.sine_valid = true;
  big.sine_v = 1.0;
  big.sine_a = 0.5;
  big.sine_omega = 2.0;
  big.sine_phase = 0.0;
  big.transition(0.5);
  assert(std::abs(big.rotation_angle - (0.5 + 0.25 * (1 - std::cos(1.0)))) < 1e-9);
  assert(std::abs(big.rotation_speed - (1 + 0.5 * std::sin(1.0))) < 1e-9);

  auto_buff_v2::RuneState warming;
  warming.center = {3, 0, 0};
  warming.start_timestamp = std::chrono::steady_clock::now() -
                            std::chrono::milliseconds(2900);
  warming.timestamp = std::chrono::steady_clock::now() +
                      std::chrono::milliseconds(300);
  warming.inactive[0] = true;
  assert(!warming.aimpoint());

  auto_buff_v2::RuneEnergyFitter fitter;
  for (int i = 0; i <= 40; ++i) {
    const double t = 0.05 * i;
    fitter.push(t, 0.3 + 1.2 * t);
  }
  const auto linear = fitter.fit_linear();
  assert(linear);
  assert(std::abs(linear->speed - 1.2) < 1e-5);
  assert(linear->cost < 1e-8);

  fitter.reset();
  for (int i = 0; i <= 120; ++i) {
    const double t = 0.05 * i;
    fitter.push(t, 0.3 + 1.2 * t - 0.35 / 2.0 * std::cos(2.0 * t + 0.4));
  }
  const auto sine = fitter.fit_sine();
  assert(sine);
  assert(std::abs(sine->v - 1.2) < 0.03);
  assert(std::abs(sine->a - 0.35) < 0.03);
  assert(std::abs(sine->omega - 2.0) < 0.03);
}
