#include "buff_big_group.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


#include "tools/math_tools.hpp"

namespace climber_reference
{
namespace
{
constexpr double kFaceAngle = 2.0 * CV_PI / 5.0;
}

double BigTargetGroup::slot_angle_from_observation(const PowerRune & observation)
{
  if (!observation.fanblades.empty()) {
    const cv::Point2f v = observation.target().center - observation.r_center;
    if (cv::norm(v) > 1e-3) {
      double angle = std::atan2(-v.y, v.x);
      if (angle < 0.0) angle += 2.0 * CV_PI;
      return angle;
    }
  }
  return tools::limit_rad(observation.ypr_in_world[2]);
}

int BigTargetGroup::slot_id_from_observation(const PowerRune & observation)
{
  if (observation.slot_id >= 0 && observation.slot_id < kSlotCount) return observation.slot_id;
  const double angle = slot_angle_from_observation(observation);
  int slot_id = static_cast<int>(std::round(angle / kFaceAngle)) % kSlotCount;
  if (slot_id < 0) slot_id += kSlotCount;
  return slot_id;
}

double BigTargetGroup::roll_distance(double a, double b)
{
  return std::abs(tools::limit_rad(a - b));
}

void BigTargetGroup::age_blade_tracks()
{
  slot_blade_ids_.fill(-1);
  for (auto & track : blade_tracks_) {
    if (!track.active) continue;
    ++track.missed_frames;
    if (track.missed_frames > max_blade_missed_frames_) {
      track = BladeTrack{};
    }
  }
}

void BigTargetGroup::update_blade_tracks()
{
  struct Candidate
  {
    int slot_id = -1;
    double roll = 0.0;
  };

  age_blade_tracks();

  std::array<Candidate, kSlotCount> candidates;
  int candidate_count = 0;
  for (int slot_id = 0; slot_id < kSlotCount; ++slot_id) {
    if (!visible_slots_[slot_id]) continue;
    auto target = model_.copy_for_slot(slot_id);
    if (!target.has_value()) continue;
    auto x = target->ekf_x();
    if (x.size() <= 5) continue;
    candidates[candidate_count++] = Candidate{slot_id, x[5]};
  }

  std::array<bool, kSlotCount> used_candidates = {false, false, false, false, false};
  std::array<bool, kSlotCount> used_tracks = {false, false, false, false, false};
  constexpr double kMatchThreshold = kFaceAngle * 0.5;

  while (true) {
    int best_track = -1;
    int best_candidate = -1;
    double best_cost = kMatchThreshold;

    for (int track_id = 0; track_id < kSlotCount; ++track_id) {
      if (used_tracks[track_id] || !blade_tracks_[track_id].active) continue;
      for (int candidate_id = 0; candidate_id < candidate_count; ++candidate_id) {
        if (used_candidates[candidate_id]) continue;
        const double cost = roll_distance(candidates[candidate_id].roll, blade_tracks_[track_id].roll);
        if (cost < best_cost) {
          best_cost = cost;
          best_track = track_id;
          best_candidate = candidate_id;
        }
      }
    }

    if (best_track < 0 || best_candidate < 0) break;

    auto & track = blade_tracks_[best_track];
    const auto & candidate = candidates[best_candidate];
    track.active = true;
    track.slot_id = candidate.slot_id;
    track.roll = candidate.roll;
    track.missed_frames = 0;
    slot_blade_ids_[candidate.slot_id] = best_track;
    used_tracks[best_track] = true;
    used_candidates[best_candidate] = true;
  }

  for (int candidate_id = 0; candidate_id < candidate_count; ++candidate_id) {
    if (used_candidates[candidate_id]) continue;

    int blade_id = -1;
    for (int track_id = 0; track_id < kSlotCount; ++track_id) {
      if (!blade_tracks_[track_id].active) {
        blade_id = track_id;
        break;
      }
    }
    if (blade_id < 0) {
      int max_missed = -1;
      for (int track_id = 0; track_id < kSlotCount; ++track_id) {
        if (used_tracks[track_id]) continue;
        if (blade_tracks_[track_id].missed_frames > max_missed) {
          max_missed = blade_tracks_[track_id].missed_frames;
          blade_id = track_id;
        }
      }
    }
    if (blade_id < 0) continue;

    const auto & candidate = candidates[candidate_id];
    auto & track = blade_tracks_[blade_id];
    track.active = true;
    track.slot_id = candidate.slot_id;
    track.roll = candidate.roll;
    track.missed_frames = 0;
    slot_blade_ids_[candidate.slot_id] = blade_id;
    used_tracks[blade_id] = true;
  }
}

void BigTargetGroup::update(
  const std::vector<PowerRune> & observations, std::chrono::steady_clock::time_point & timestamp)
{
  if (observations.empty()) {
    model_.get_target(std::nullopt, timestamp);
    if (model_.is_unsolve()) visible_slots_.fill(false);
    age_blade_tracks();
    return;
  }

  std::array<std::optional<PowerRune>, kSlotCount> slots;
  std::array<bool, kSlotCount> new_visible = {false, false, false, false, false};

  for (const auto & observation : observations) {
    const int slot_id = slot_id_from_observation(observation);
    if (slot_id < 0 || slot_id >= kSlotCount) continue;

    PowerRune slot_observation = observation;
    slot_observation.slot_id = slot_id;
    slot_observation.slot_angle = slot_angle_from_observation(observation);
    slots[slot_id] = slot_observation;
    new_visible[slot_id] = true;
  }

  int track_slot = -1;
  const int previous_slot = model_.blade_index();
  if (previous_slot >= 0 && previous_slot < kSlotCount && slots[previous_slot].has_value()) {
    track_slot = previous_slot;
  } else {
    for (int i = 0; i < kSlotCount; ++i) {
      if (slots[i].has_value()) {
        track_slot = i;
        break;
      }
    }
  }

  if (track_slot < 0) {
    model_.get_target(std::nullopt, timestamp);
    if (model_.is_unsolve()) visible_slots_.fill(false);
    age_blade_tracks();
    return;
  }

  Eigen::Vector3d fused_xyz = Eigen::Vector3d::Zero();
  int fused_count = 0;
  for (const auto & slot : slots) {
    if (!slot.has_value()) continue;
    const auto & p = slot.value();
    if (p.xyz_in_world.allFinite() && p.xyz_in_world.norm() > 1e-6) {
      fused_xyz += p.xyz_in_world;
      ++fused_count;
    }
  }
  if (fused_count > 0) {
    fused_xyz /= static_cast<double>(fused_count);
    const Eigen::Vector3d fused_ypd = tools::xyz2ypd(fused_xyz);
    for (auto & slot : slots) {
      if (!slot.has_value()) continue;
      slot->xyz_in_world = fused_xyz;
      slot->ypd_in_world = fused_ypd;
    }
  }

  model_.set_blade_index(track_slot);
  model_.get_target(slots[track_slot], timestamp);
  if (model_.is_unsolve()) {
    visible_slots_.fill(false);
    model_.set_blade_index(-1);
    age_blade_tracks();
    return;
  }
  visible_slots_ = new_visible;
  update_blade_tracks();
}

std::optional<BigTarget> BigTargetGroup::get_target_copy(int id) const
{
  if (id < 0 || id >= kSlotCount) return std::nullopt;
  if (!visible_slots_[id] || model_.is_unsolve()) return std::nullopt;
  auto target = model_.copy_for_slot(id);
  if (!target.has_value()) return std::nullopt;
  target->set_blade_id(blade_id_for_slot(id));
  return target;
}

int BigTargetGroup::blade_id_for_slot(int id) const
{
  if (id < 0 || id >= kSlotCount) return -1;
  return slot_blade_ids_[id];
}


}  // namespace climber_reference
