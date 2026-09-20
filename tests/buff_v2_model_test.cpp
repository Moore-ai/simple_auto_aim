#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>

#include "tasks/auto_buff_v2/rune_energy_fitter.hpp"
#include "tasks/auto_buff_v2/rune_model.hpp"
#include "tasks/auto_buff_v2/rune_predictor.hpp"

static_assert(std::is_same_v<
              decltype(std::declval<auto_buff_v2::RuneEnergyFitter>().fit_linear()),
              std::optional<auto_buff_v2::RuneEnergyFitter::LinearFitResult>>);
static_assert(std::is_same_v<
              decltype(std::declval<auto_buff_v2::RuneEnergyFitter>().fit_sine()),
              std::optional<auto_buff_v2::RuneEnergyFitter::SineFitResult>>);

int main()
{
  auto_buff_v2::RunePredictor predictor;
  const auto start = auto_buff_v2::Timestamp{};
  auto_buff_v2::RuneEstimate small;
  small.rotation_speed = 1.2;
  small.rotation_angle = 0.1;
  const auto small_future = predictor.predict(
    small, start + std::chrono::duration_cast<auto_buff_v2::Timestamp::duration>(
                     std::chrono::duration<double>(0.5)));
  assert(std::abs(small_future.rotation_angle - 0.7) < 1e-9);

  auto_buff_v2::RuneEstimate big;
  big.sine_valid = true;
  big.sine_v = 1.0;
  big.sine_a = 0.5;
  big.sine_omega = 2.0;
  big.sine_phase = 0.0;
  const auto big_future = predictor.predict(
    big, start + std::chrono::duration_cast<auto_buff_v2::Timestamp::duration>(
                   std::chrono::duration<double>(0.5)));
  assert(std::abs(big_future.rotation_angle - (0.5 + 0.25 * (1 - std::cos(1.0)))) < 1e-9);
  assert(std::abs(big_future.rotation_speed - (1 + 0.5 * std::sin(1.0))) < 1e-9);

  auto_buff_v2::RuneEstimate warming;
  warming.center = {3, 0, 0};
  warming.start_timestamp = start;
  warming.timestamp = start + std::chrono::seconds(2);
  warming.inactive[0] = true;
  assert(!predictor.aimpoint_at(warming, start + std::chrono::milliseconds(2999)));
  assert(predictor.aimpoint_at(warming, start + std::chrono::seconds(3)));

  auto_buff_v2::RuneEstimate estimate;
  estimate.center = {3, 0, 0};
  estimate.start_timestamp = start - std::chrono::seconds(3);
  estimate.timestamp = start;
  estimate.rotation_speed = 1;
  estimate.inactive[0] = true;
  const auto aimpoint = predictor.aimpoint_at(
    estimate, start + std::chrono::duration_cast<auto_buff_v2::Timestamp::duration>(
                        std::chrono::duration<double>(std::acos(-1) / 2)));
  assert(aimpoint);
  assert(((*aimpoint - Eigen::Vector3d{3, -0.7, 0}).norm()) < 1e-9);
  assert(std::abs(estimate.rotation_angle) < 1e-12);
  assert(std::abs(estimate.rotation_speed - 1) < 1e-12);

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
