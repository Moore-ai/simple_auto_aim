#include <cassert>

#include <Eigen/Core>

#include "tasks/auto_buff_v2/hungarian.hpp"

int main()
{
  Eigen::MatrixXd cost(2, 2);
  cost << 1, 2, 0, 100;
  const auto result = auto_buff_v2::hungarian_assign(cost, 10.0);
  assert(result[0] && *result[0] == 1);
  assert(result[1] && *result[1] == 0);

  cost << 100, 100, 0, 1;
  const auto rejected = auto_buff_v2::hungarian_assign(cost, 10.0);
  assert(!rejected[0]);
  assert(rejected[1] && *rejected[1] == 0);
}
