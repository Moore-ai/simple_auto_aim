#include "ballistic_solver.hpp"

#include <cmath>
#include <stdexcept>

#include "trajectory.hpp"

namespace tools
{
namespace
{
class NjustBallisticSolver final : public BallisticSolver
{
public:
  explicit NjustBallisticSolver(double air_resistance) : air_resistance_(air_resistance) {}

  std::optional<BallisticSolution> solve(double speed, double distance,
                                          double height) const override
  {
    if (speed <= 0 || distance <= 0) return std::nullopt;
    constexpr double kStep = 0.005;
    constexpr double kGravity = 9.81;
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
        vx -= air_resistance_ * velocity * vx * kStep;
        vy -= (kGravity + air_resistance_ * velocity * vy) * kStep;
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
      if (std::abs(error) < 0.001) return BallisticSolution{pitch, fly_time};
      pitch += std::atan2(error, distance);
      if (std::abs(pitch) > 80.0 / 57.3) break;
    }
    return std::nullopt;
  }
private:
  double air_resistance_;
};

class VacuumBallisticSolver final : public BallisticSolver
{
public:
  std::optional<BallisticSolution> solve(double speed, double distance,
                                          double height) const override
  {
    if (speed <= 0 || distance <= 0) return std::nullopt;
    const Trajectory trajectory(speed, distance, height);
    if (trajectory.unsolvable) return std::nullopt;
    return BallisticSolution{trajectory.pitch, trajectory.fly_time};
  }
};
}  // namespace

std::unique_ptr<BallisticSolver> make_ballistic_solver(const std::string & model,
                                                        const BallisticSolverConfig & config)
{
  if (model == "njust") {
    if (!std::isfinite(config.njust_air_resistance) || config.njust_air_resistance <= 0) {
      throw std::invalid_argument("njust_air_resistance 必须为有限正数");
    }
    return std::make_unique<NjustBallisticSolver>(config.njust_air_resistance);
  }
  if (model == "vacuum") return std::make_unique<VacuumBallisticSolver>();
  throw std::invalid_argument("未知弹道模型: " + model);
}
}  // namespace tools
