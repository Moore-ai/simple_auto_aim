#ifndef AUTO_BUFF_V2__RUNE_TRAJECTORY_HPP
#define AUTO_BUFF_V2__RUNE_TRAJECTORY_HPP

#include <cmath>
#include <optional>

namespace auto_buff_v2
{
struct RuneTrajectory
{
  double pitch = 0;
  double fly_time = 0;
};

inline std::optional<RuneTrajectory> solve_rune_trajectory(double speed, double distance,
                                                            double height)
{
  if (speed <= 0 || distance <= 0) return std::nullopt;
  constexpr double kStep = 0.005;
  constexpr double kGravity = 9.81;
  constexpr double kDrag = 0.003;
  double pitch = std::atan2(height, distance);
  for (int iteration = 0; iteration < 10; ++iteration) {
    double x = 0, y = 0, t = 0;
    double vx = speed * std::cos(pitch), vy = speed * std::sin(pitch);
    double previous_x = 0, previous_y = 0, previous_t = 0;
    while (x < distance) {
      previous_x = x;
      previous_y = y;
      previous_t = t;
      const double velocity = std::hypot(vx, vy);
      vx -= kDrag * velocity * vx * kStep;
      vy -= (kGravity + kDrag * velocity * vy) * kStep;
      x += vx * kStep;
      y += vy * kStep;
      t += kStep;
      if (t > 4 || vx <= 0.1) break;
    }
    if (x < distance || x <= previous_x) return std::nullopt;
    const double ratio = (distance - previous_x) / (x - previous_x);
    const double actual_height = previous_y + (y - previous_y) * ratio;
    const double fly_time = previous_t + (t - previous_t) * ratio;
    const double error = height - actual_height;
    if (std::abs(error) < 0.001) return RuneTrajectory{pitch, fly_time};
    pitch += std::atan2(error, distance);
    if (std::abs(pitch) > 80.0 / 57.3) break;
  }
  return std::nullopt;
}
}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_TRAJECTORY_HPP
