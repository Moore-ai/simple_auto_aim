#include <cassert>
#include <chrono>

#include "tools/frame_facts.hpp"

int main()
{
  const auto timestamp = std::chrono::steady_clock::now();
  cv::Mat image(2, 3, CV_8UC1, cv::Scalar(7));
  io::GimbalStatePacket received;
  received.state.yaw = 1.25F;
  received.packet[0] = 42;
  io::GimbalCommandPacket sent;
  sent.command.pitch = -0.5F;
  sent.packet[0] = 24;

  const tools::FrameFacts facts{
    timestamp, image, Eigen::Quaterniond::Identity(), received};
  const auto frame = facts.snapshot(sent, {}, {});

  assert(frame.timestamp == timestamp);
  assert(frame.image.size() == image.size());
  assert(frame.image.data != image.data);
  assert(frame.gimbal_state.yaw == 1.25F);
  assert(frame.gimbal_command.pitch == -0.5F);
  assert(frame.serial_receive_packet[0] == 42);
  assert(frame.serial_send_packet[0] == 24);

  auto color_facts = facts;
  color_facts.received.state.mode = 0;
  assert(color_facts.enemy_color() == io::InfantryEnemyColor::red);
  color_facts.received.state.mode = 1;
  assert(color_facts.enemy_color() == io::InfantryEnemyColor::blue);
  color_facts.received.state.mode = 2;
  assert(!color_facts.enemy_color());

  color_facts.target_color_override = io::InfantryEnemyColor::red;
  color_facts.received.state.mode = 1;
  assert(color_facts.enemy_color() == io::InfantryEnemyColor::red);
  color_facts.target_color_override = io::InfantryEnemyColor::blue;
  color_facts.received.state.mode = 0;
  assert(color_facts.enemy_color() == io::InfantryEnemyColor::blue);
  color_facts.received.state.mode = 2;
  assert(color_facts.enemy_color() == io::InfantryEnemyColor::blue);
  assert(color_facts.snapshot(sent, {}, {}).gimbal_state.mode == 2);
  return 0;
}
