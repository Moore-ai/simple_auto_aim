#ifndef AUTO_AIM__OUTPOST_MODEL_HPP
#define AUTO_AIM__OUTPOST_MODEL_HPP

#include <Eigen/Dense>

#include <memory>
#include <variant>
#include <vector>

#include "armor.hpp"
#include "outpost_state.hpp"
#include "outpost_state_v2.hpp"
#include "target_estimator.hpp"

namespace auto_aim
{

struct OutpostUpdateResult
{
  bool updated = false;
  int armor_id = -1;
  std::vector<int> armor_ids;
};

using OutpostDebugState = std::variant<OutpostState, OutpostStateV2>;

struct OutpostSnapshot
{
  TargetState compatibility_state;
  OutpostDebugState debug_state;
  Eigen::VectorXd state_vector;
  std::vector<PredictedArmorPose> armor_poses;
  TargetEstimatorDiagnostics diagnostics;
  double last_nis = 0.0;
  bool direction_locked = false;
  bool all_finite = false;
};

class OutpostModel
{
public:
  virtual ~OutpostModel() = default;

  virtual std::unique_ptr<OutpostModel> clone() const = 0;
  virtual void begin_frame() = 0;
  virtual void predict(double dt) = 0;
  virtual OutpostUpdateResult update(const std::vector<Armor> & armors) = 0;

  virtual OutpostSnapshot snapshot() const = 0;
  virtual const TargetEstimatorDiagnostics & diagnostics() const = 0;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_MODEL_HPP
