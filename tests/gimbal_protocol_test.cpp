#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

#include <Eigen/Geometry>

#include "io/gimbal/infantry_protocol.hpp"

namespace
{
float float_at(const uint8_t * data, size_t offset)
{
  float value;
  std::memcpy(&value, data + offset, sizeof(value));
  return value;
}

uint8_t crc8(const uint8_t * data, size_t size)
{
  uint8_t crc = 0;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = crc & 0x80 ? (crc << 1) ^ 0x31 : crc << 1;
  }
  return crc;
}
}  // namespace

int main()
{
  static_assert(io::kInfantryDefaultBaudrate == 115200);
  static_assert(io::kInfantryDefaultBytesize == 8);
  static_assert(io::kInfantryReadTimeoutMs == 2);
  static_assert(io::kInfantryWriteTimeoutMs == 5);
  assert(io::is_supported_infantry_bytesize(5));
  assert(io::is_supported_infantry_bytesize(6));
  assert(io::is_supported_infantry_bytesize(7));
  assert(io::is_supported_infantry_bytesize(8));
  assert(!io::is_supported_infantry_bytesize(9));

  const auto command = io::make_infantry_command_packet(
    true, true, 1.25F, -2.5F, 8.5F, 3.75F, -4.0F, 5.0F, -6.0F);
  assert(command[0] == 0xFF);
  assert(command[1] == 1);
  // SP 内部 pitch 向上为负；下位机绝对角度约定向上为正。
  // 若此处改为传输相对增量或遗漏坐标系转换，下列绝对角字段会错误。
  assert(float_at(command.data(), 2) == -1.25F);
  assert(float_at(command.data(), 6) == -2.5F);
  assert(float_at(command.data(), 10) == 8.5F);
  assert(float_at(command.data(), 14) == -3.75F);
  assert(float_at(command.data(), 18) == -4.0F);
  assert(float_at(command.data(), 22) == -5.0F);
  assert(float_at(command.data(), 26) == -6.0F);
  assert(command[30] == crc8(command.data(), 30));
  assert(command[31] == 0x0D);

  const auto invalid_command = io::make_infantry_command_packet(
    true, false, NAN, NAN, NAN, NAN, NAN, NAN, NAN);
  assert(float_at(invalid_command.data(), 2) == 0.0F);
  assert(float_at(invalid_command.data(), 6) == 0.0F);
  assert(float_at(invalid_command.data(), 10) == 0.0F);
  assert(float_at(invalid_command.data(), 14) == 0.0F);
  assert(float_at(invalid_command.data(), 18) == 0.0F);
  assert(float_at(invalid_command.data(), 22) == 0.0F);
  assert(float_at(invalid_command.data(), 26) == 0.0F);

  std::array<uint8_t, io::kInfantryFeedbackPacketSize> feedback{};
  feedback[0] = 0xFF;
  feedback[1] = 2;
  const float roll = 0.1F;
  const float pitch = -0.2F;
  const float yaw = 0.3F;
  std::memcpy(feedback.data() + 2, &roll, sizeof(roll));
  std::memcpy(feedback.data() + 6, &pitch, sizeof(pitch));
  std::memcpy(feedback.data() + 10, &yaw, sizeof(yaw));
  feedback[22] = crc8(feedback.data(), 22);
  feedback[23] = 0x0D;

  io::InfantryFeedback decoded;
  assert(io::parse_infantry_feedback_packet(feedback.data(), decoded));
  assert(decoded.mode == 2);
  assert(std::abs(decoded.roll - roll) < 1e-6F);
  assert(std::abs(decoded.pitch - pitch) < 1e-6F);
  assert(std::abs(decoded.yaw - yaw) < 1e-6F);
  assert(io::infantry_feedback_pitch_to_sp(decoded.pitch) == -pitch);

  const auto simulated_feedback = io::make_infantry_feedback_packet(3, 0.4F, -0.5F, 0.6F);
  io::InfantryFeedback simulated_decoded;
  assert(io::parse_infantry_feedback_packet(simulated_feedback.data(), simulated_decoded));
  assert(simulated_decoded.mode == 3);
  assert(std::abs(simulated_decoded.roll - 0.4F) < 1e-6F);
  assert(std::abs(simulated_decoded.pitch + 0.5F) < 1e-6F);
  assert(std::abs(simulated_decoded.yaw - 0.6F) < 1e-6F);

  const auto feedback_q = io::infantry_feedback_quaternion(roll, pitch, yaw);
  const Eigen::Quaterniond expected_q(
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
  assert(feedback_q.angularDistance(expected_q) < 1e-12);

  io::InfantryFeedbackStreamParser parser;
  const std::array<uint8_t, 3> noise{0x01, 0x02, 0x03};
  parser.push(noise.data(), noise.size());
  auto bad_feedback = feedback;
  bad_feedback[22] ^= 0x01;
  parser.push(bad_feedback.data(), bad_feedback.size());
  parser.push(feedback.data(), 7);
  parser.push(feedback.data() + 7, feedback.size() - 7);
  io::InfantryFeedback streamed;
  assert(parser.pop(streamed));
  assert(streamed.mode == decoded.mode);
  assert(std::abs(streamed.pitch - decoded.pitch) < 1e-6F);

  feedback[22] ^= 0x01;
  assert(!io::parse_infantry_feedback_packet(feedback.data(), decoded));
}
