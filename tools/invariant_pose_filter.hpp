#ifndef TOOLS__INVARIANT_POSE_FILTER_HPP
#define TOOLS__INVARIANT_POSE_FILTER_HPP

#include <memory>

#include "tools/filter_base.hpp"

namespace tools
{

class InvariantPoseFilter : public FilterBase
{
public:
  using FilterBase::FilterBase;

  Eigen::VectorXd update(
    const Eigen::VectorXd & z, const Eigen::MatrixXd & H, const Eigen::MatrixXd & R,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> h,
    std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> z_subtract) override;

  std::unique_ptr<FilterBase> clone() const override;
};

}  // namespace tools

#endif  // TOOLS__INVARIANT_POSE_FILTER_HPP
