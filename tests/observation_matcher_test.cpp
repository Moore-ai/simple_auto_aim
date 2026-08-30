#include <cassert>

#include "tasks/auto_aim/observation_matcher.hpp"
#include "tasks/auto_aim/solver.hpp"

int main()
{
  auto_aim::Solver solver("configs/demo.yaml");
  auto_aim::ObservationGeometry geometry(solver);
  auto_aim::ObservationMatcher matcher(geometry);
  auto_aim::DetectionResult detections;
  const auto points = solver.reproject_armor(
    {2.0, 0.0, 0.0}, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);
  auto armor = auto_aim::Armor(0, 0.9F, cv::Rect{}, points);
  armor.name = auto_aim::ArmorName::sentry;
  armor.type = auto_aim::ArmorType::small;
  detections.armors.push_back(armor);

  auto_aim::ObservationMatchStats stats;
  const auto matches = matcher.match_armors(
    detections, auto_aim::ArmorName::sentry, auto_aim::ArmorType::small,
    {{2.0, 0.0, 0.0, 0.0}}, {0}, {}, true, stats);
  assert(matches.size() == 1);
  assert(matches.front().prediction_id == 0);
  assert(matches.front().observation == &detections.armors.front());
  assert(stats.rejected_armor_count == 0);
  return 0;
}
