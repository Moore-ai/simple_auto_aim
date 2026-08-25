#ifndef AUTO_AIM__OUTPOST_MODEL_HPP
#define AUTO_AIM__OUTPOST_MODEL_HPP

#include <Eigen/Dense>

#include <memory>
#include <optional>
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

class OutpostModel
{
public:
  virtual ~OutpostModel() = default;

  virtual std::unique_ptr<OutpostModel> clone() const = 0;
  virtual void begin_frame() = 0;
  virtual void predict(double dt) = 0;
  virtual OutpostUpdateResult update(const std::vector<Armor> & armors) = 0;

  virtual TargetState compatibility_state() const = 0;
  virtual std::optional<OutpostState> outpost_state() const = 0;
  virtual std::optional<OutpostStateV2> outpost_state_v2() const = 0;
  virtual Eigen::VectorXd state_vector() const = 0;
  virtual std::vector<PredictedArmorPose> armor_pose_list() const = 0;
  virtual double last_nis() const = 0;
  virtual const TargetEstimatorDiagnostics & diagnostics() const = 0;
  virtual bool has_bad_nis_convergence(double failure_rate) const = 0;
  virtual bool direction_locked() const = 0;
  virtual bool all_finite() const = 0;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OUTPOST_MODEL_HPP
