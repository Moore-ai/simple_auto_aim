#include <array>
#include <cassert>
#include <cmath>
#include <cstring>

#include "io/gimbal/infantry_protocol.hpp"

template <typename T, size_t N>
T read_at(const std::array<uint8_t, N> & bytes, size_t offset)
{
  T value;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

int main()
{
  const uint8_t check[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  assert(io::infantry_crc8(check, sizeof(check)) == 0xA2);

  const auto command = io::make_infantry_command_packet(
    true, io::InfantryFireCommand::none, -0.2F, 0.3F, 5.0F,
    -0.1F, 0.4F, -0.5F, 0.6F);
  assert(command.size() == 64);
  assert(command[0] == 0xFF && command[1] == 0 && command[63] == 0x0D);
  assert(std::abs(read_at<float>(command, 2) - 11.459156F) < 1e-5F);
  assert(std::abs(read_at<float>(command, 6) - 17.188734F) < 1e-5F);
  assert(read_at<float>(command, 10) == 5.0F);
  assert(read_at<int32_t>(command, 14) == 0);
  assert(std::abs(read_at<float>(command, 18) - 5.729578F) < 1e-5F);
  assert(std::abs(read_at<float>(command, 22) - 22.918312F) < 1e-5F);
  for (size_t i = 26; i < 62; ++i) assert(command[i] == 0);
  assert(command[62] == io::infantry_crc8(command.data(), 62));
  const auto firing = io::make_infantry_command_packet(
    true, io::InfantryFireCommand::fire, 0, 0, 5, 0, 0, 0, 0);
  assert(firing[1] == 1);
  const auto disabled = io::make_infantry_command_packet(
    false, io::InfantryFireCommand::fire, 0, 0, -1, 0, 0, 0, 0);
  assert(disabled[1] == 0);
  assert(read_at<int32_t>(disabled, 14) == -1);

  // Firmware-shaped frame: roll=10, yaw=30, pitch=20, nonzero navigation payload.
  std::array<uint8_t, 64> feedback_bytes{};
  feedback_bytes[0] = 0xFF;
  feedback_bytes[1] = 0;
  const float angles[] = {10.0F, 30.0F, 20.0F};
  std::memcpy(feedback_bytes.data() + 2, angles, sizeof(angles));
  for (size_t i = 14; i < 62; ++i) feedback_bytes[i] = static_cast<uint8_t>(i);
  feedback_bytes[62] = io::infantry_crc8(feedback_bytes.data(), 62);
  feedback_bytes[63] = 0x0D;
  io::InfantryFeedback feedback{};
  assert(io::parse_infantry_feedback_packet(feedback_bytes.data(), feedback));
  assert(std::abs(feedback.roll - 0.17453293F) < 1e-6F);
  assert(std::abs(feedback.yaw - 0.52359878F) < 1e-6F);
  assert(std::abs(feedback.pitch + 0.34906585F) < 1e-6F);
  assert(io::infantry_enemy_color(feedback.mode) == io::InfantryEnemyColor::blue);
  assert(io::infantry_enemy_color(1) == io::InfantryEnemyColor::red);
  assert(!io::infantry_enemy_color(2));

  io::InfantryFeedbackStreamParser parser;
  const uint8_t noise[] = {0x01, 0xFF, 0x02};
  parser.push(noise, sizeof(noise));
  auto corrupt = feedback_bytes;
  corrupt[40] ^= 1;
  parser.push(corrupt.data(), corrupt.size());
  parser.push(feedback_bytes.data(), 17);
  assert(!parser.pop(feedback));
  parser.push(feedback_bytes.data() + 17, feedback_bytes.size() - 17);
  parser.push(feedback_bytes.data(), feedback_bytes.size());
  assert(parser.pop(feedback) && std::abs(feedback.yaw - 0.52359878F) < 1e-6F);
  assert(parser.pop(feedback) && std::abs(feedback.pitch + 0.34906585F) < 1e-6F);
  assert(!parser.pop(feedback));
  feedback_bytes[63] = 0;
  assert(!io::parse_infantry_feedback_packet(feedback_bytes.data(), feedback));
}
