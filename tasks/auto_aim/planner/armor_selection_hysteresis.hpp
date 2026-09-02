#ifndef AUTO_AIM__ARMOR_SELECTION_HYSTERESIS_HPP
#define AUTO_AIM__ARMOR_SELECTION_HYSTERESIS_HPP

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

namespace auto_aim
{

struct ArmorSelectionHysteresisConfig
{
  bool enable = true;
  double switch_margin = 0.02;
  int switch_confirm_frames = 3;
};

class ArmorSelectionHysteresis
{
public:
  explicit ArmorSelectionHysteresis(ArmorSelectionHysteresisConfig config) : config_(config) {}

  int select(const std::vector<double> & scores)
  {
    const auto best = best_index(scores);
    if (best < 0) return -1;
    if (!config_.enable) {
      reset(best);
      return best;
    }
    if (!locked_ || *locked_ >= static_cast<int>(scores.size()) ||
        !std::isfinite(scores[*locked_])) {
      reset(best);
      return best;
    }
    if (best == *locked_ ||
        scores[best] + config_.switch_margin >= scores[*locked_]) {
      pending_.reset();
      pending_count_ = 0;
      return *locked_;
    }
    if (pending_ != best) {
      pending_ = best;
      pending_count_ = 1;
    } else {
      ++pending_count_;
    }
    if (pending_count_ >= std::max(config_.switch_confirm_frames, 1)) reset(best);
    return *locked_;
  }

  void clear()
  {
    locked_.reset();
    pending_.reset();
    pending_count_ = 0;
  }

private:
  static int best_index(const std::vector<double> & scores)
  {
    int result = -1;
    for (int index = 0; index < static_cast<int>(scores.size()); ++index) {
      if (std::isfinite(scores[index]) &&
          (result < 0 || scores[index] < scores[result])) {
        result = index;
      }
    }
    return result;
  }

  void reset(int locked)
  {
    locked_ = locked;
    pending_.reset();
    pending_count_ = 0;
  }

  ArmorSelectionHysteresisConfig config_;
  std::optional<int> locked_;
  std::optional<int> pending_;
  int pending_count_ = 0;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_SELECTION_HYSTERESIS_HPP
