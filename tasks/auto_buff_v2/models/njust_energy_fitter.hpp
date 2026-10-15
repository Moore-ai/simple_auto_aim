#ifndef AUTO_BUFF_V2__NJUST_ENERGY_FITTER_HPP
#define AUTO_BUFF_V2__NJUST_ENERGY_FITTER_HPP

#include <deque>
#include <limits>
#include <optional>

namespace auto_buff_v2
{

class NjustEnergyFitter
{
public:
  struct Config
  {
    double window_seconds = 6.0;
    double min_fit_seconds = 1.5;
    double weight_half_life_seconds = 3.0;
  };

  NjustEnergyFitter() = default;
  explicit NjustEnergyFitter(Config config);

  struct SineFitResult
  {
    double C = 0;
    double v = 0;
    double a = 0;
    double omega = 0;
    double phi = 0;
    double cost = std::numeric_limits<double>::max();
  };

  struct LinearFitResult
  {
    double C = 0;
    double speed = 0;
    double cost = std::numeric_limits<double>::max();
  };

  void push(double t, double theta);
  void reset();

  std::optional<LinearFitResult> fit_linear() const;
  std::optional<SineFitResult> fit_sine() const;

private:
  Config config_;
  struct Point
  {
    double t;
    double theta;
  };
  std::deque<Point> buffer_;

  template <typename Pred>
  double compute_weighted_cost(const std::deque<Point> & buffer, Pred && pred_fn) const;
  double sample_weight(double t, double reference_t) const;
};

}  // namespace auto_buff_v2

#endif  // AUTO_BUFF_V2__NJUST_ENERGY_FITTER_HPP
