#include "target_state.hpp"

#include <stdexcept>

namespace auto_aim
{
TargetState::TargetState() : values_(Eigen::VectorXd::Zero(dimension)) {}

TargetState::TargetState(const Eigen::VectorXd & values) : values_{values}
{
  if (values_.size() != dimension) {
    throw std::invalid_argument("TargetState must contain exactly 11 values");
  }
}

Eigen::VectorXd TargetState::vector() const { return values_; }

bool TargetState::all_finite() const { return values_.allFinite(); }

Eigen::Index TargetState::index(TargetStateComponent component)
{
  return static_cast<Eigen::Index>(component);
}

double TargetState::center_x() const { return values_[index(TargetStateComponent::center_x)]; }
double TargetState::velocity_x() const { return values_[index(TargetStateComponent::velocity_x)]; }
double TargetState::center_y() const { return values_[index(TargetStateComponent::center_y)]; }
double TargetState::velocity_y() const { return values_[index(TargetStateComponent::velocity_y)]; }
double TargetState::center_z() const { return values_[index(TargetStateComponent::center_z)]; }
double TargetState::velocity_z() const { return values_[index(TargetStateComponent::velocity_z)]; }
double TargetState::yaw() const { return values_[index(TargetStateComponent::yaw)]; }
double TargetState::yaw_rate() const { return values_[index(TargetStateComponent::yaw_rate)]; }
double TargetState::radius() const { return values_[index(TargetStateComponent::radius)]; }
double TargetState::radius_diff() const { return values_[index(TargetStateComponent::radius_diff)]; }
double TargetState::height_diff() const { return values_[index(TargetStateComponent::height_diff)]; }

void TargetState::set_center_x(double value) { values_[index(TargetStateComponent::center_x)] = value; }
void TargetState::set_velocity_x(double value) { values_[index(TargetStateComponent::velocity_x)] = value; }
void TargetState::set_center_y(double value) { values_[index(TargetStateComponent::center_y)] = value; }
void TargetState::set_velocity_y(double value) { values_[index(TargetStateComponent::velocity_y)] = value; }
void TargetState::set_center_z(double value) { values_[index(TargetStateComponent::center_z)] = value; }
void TargetState::set_velocity_z(double value) { values_[index(TargetStateComponent::velocity_z)] = value; }
void TargetState::set_yaw(double value) { values_[index(TargetStateComponent::yaw)] = value; }
void TargetState::set_yaw_rate(double value) { values_[index(TargetStateComponent::yaw_rate)] = value; }
void TargetState::set_radius(double value) { values_[index(TargetStateComponent::radius)] = value; }
void TargetState::set_radius_diff(double value) { values_[index(TargetStateComponent::radius_diff)] = value; }
void TargetState::set_height_diff(double value) { values_[index(TargetStateComponent::height_diff)] = value; }

double TargetState::radius(bool long_axis) const
{
  return radius() + (long_axis ? radius_diff() : 0.0);
}

double TargetState::armor_height(bool long_axis) const
{
  return center_z() + (long_axis ? height_diff() : 0.0);
}
}  // namespace auto_aim
