#ifndef AUTO_BUFF_V2__RUNE_ENERGY_FITTER_HPP
#define AUTO_BUFF_V2__RUNE_ENERGY_FITTER_HPP

#include <deque>
#include <limits>
#include <optional>

namespace auto_buff_v2
{

class RuneEnergyFitter
{
public:
  struct FitResult
  {
    double C = 0;
    double v = 0;
    double a = 0;
    double omega = 0;
    double phi = 0;
    double cost = std::numeric_limits<double>::max();
  };

  struct LinearResult
  {
    double C = 0;
    double speed = 0;
    double cost = std::numeric_limits<double>::max();
  };

  void push(double t, double theta);
  void reset();

  std::optional<LinearResult> fit_linear() const;
  std::optional<FitResult> fit_sine() const;

  static constexpr double kWindowSeconds = 6.0;
  static constexpr double kMinFitSeconds = 1.5;
  static constexpr double kWeightHalfLifeSeconds = 3.0;

private:
  struct Point
  {
    double t;
    double theta;
  };
  std::deque<Point> buffer_;

  template <typename Pred>
  static double compute_weighted_cost(const std::deque<Point> & buffer, Pred && pred_fn);
};

}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__RUNE_ENERGY_FITTER_HPP
