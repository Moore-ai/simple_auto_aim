#ifndef IO__GIMBAL__INFANTRY_PROTOCOL_HPP
#define IO__GIMBAL__INFANTRY_PROTOCOL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <deque>

#include <Eigen/Geometry>

namespace io
{
constexpr size_t kInfantryCommandPacketSize = 32;
constexpr size_t kInfantryFeedbackPacketSize = 24;
constexpr uint32_t kInfantryDefaultBaudrate = 115200;
constexpr int kInfantryDefaultBytesize = 8;
constexpr uint32_t kInfantryReadTimeoutMs = 2;
constexpr uint32_t kInfantryWriteTimeoutMs = 5;

inline bool is_supported_infantry_bytesize(int bytesize)
{
  return bytesize >= 5 && bytesize <= 8;
}

struct InfantryFeedback
{
  uint8_t mode;
  float roll;
  float pitch;
  float yaw;
};

inline Eigen::Quaterniond infantry_feedback_quaternion(float roll, float pitch, float yaw)
{
  return Eigen::Quaterniond(
           Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(-pitch, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
    .normalized();
}

inline float infantry_feedback_pitch_to_sp(float lower_pitch) { return -lower_pitch; }

inline uint8_t infantry_crc8(const uint8_t * data, size_t size)
{
  uint8_t crc = 0;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = crc & 0x80 ? (crc << 1) ^ 0x31 : crc << 1;
  }
  return crc;
}

inline void load_infantry_float(uint8_t * data, size_t offset, float value)
{
  if (!std::isfinite(value)) value = 0.0F;
  std::memcpy(data + offset, &value, sizeof(value));
}

inline std::array<uint8_t, kInfantryCommandPacketSize> make_infantry_command_packet(
  bool control, bool fire, float pitch_abs_sp, float yaw_abs_sp, float distance, float pitch_vel_sp,
  float yaw_vel_sp, float pitch_acc_sp, float yaw_acc_sp)
{
  std::array<uint8_t, kInfantryCommandPacketSize> packet{};
  packet[0] = 0xFF;
  packet[1] = control && fire ? 1 : 0;
  // 本项目内部 pitch 向上为负；下位机绝对角度约定向上为正。
  load_infantry_float(packet.data(), 2, -pitch_abs_sp);
  load_infantry_float(packet.data(), 6, yaw_abs_sp);
  load_infantry_float(packet.data(), 10, distance);
  load_infantry_float(packet.data(), 14, -pitch_vel_sp);
  load_infantry_float(packet.data(), 18, yaw_vel_sp);
  load_infantry_float(packet.data(), 22, -pitch_acc_sp);
  load_infantry_float(packet.data(), 26, yaw_acc_sp);
  packet[30] = infantry_crc8(packet.data(), 30);
  packet[31] = 0x0D;
  return packet;
}

inline bool parse_infantry_feedback_packet(const uint8_t * packet, InfantryFeedback & feedback)
{
  if (packet[0] != 0xFF || packet[23] != 0x0D || infantry_crc8(packet, 22) != packet[22]) {
    return false;
  }

  feedback.mode = packet[1];
  std::memcpy(&feedback.roll, packet + 2, sizeof(feedback.roll));
  std::memcpy(&feedback.pitch, packet + 6, sizeof(feedback.pitch));
  std::memcpy(&feedback.yaw, packet + 10, sizeof(feedback.yaw));
  return true;
}

inline std::array<uint8_t, kInfantryFeedbackPacketSize> make_infantry_feedback_packet(
  uint8_t mode, float roll, float pitch, float yaw)
{
  std::array<uint8_t, kInfantryFeedbackPacketSize> packet{};
  packet[0] = 0xFF;
  packet[1] = mode;
  load_infantry_float(packet.data(), 2, roll);
  load_infantry_float(packet.data(), 6, pitch);
  load_infantry_float(packet.data(), 10, yaw);
  packet[22] = infantry_crc8(packet.data(), 22);
  packet[23] = 0x0D;
  return packet;
}

class InfantryFeedbackStreamParser
{
public:
  void push(const uint8_t * data, size_t size)
  {
    for (size_t i = 0; i < size; ++i) buffer_.push_back(data[i]);
    if (buffer_.size() > kInfantryFeedbackPacketSize * 32) buffer_.clear();
  }

  bool pop(InfantryFeedback & feedback)
  {
    while (buffer_.size() >= kInfantryFeedbackPacketSize) {
      while (!buffer_.empty() && buffer_.front() != 0xFF) buffer_.pop_front();
      if (buffer_.size() < kInfantryFeedbackPacketSize) return false;

      std::array<uint8_t, kInfantryFeedbackPacketSize> candidate{};
      for (size_t i = 0; i < candidate.size(); ++i) candidate[i] = buffer_[i];
      if (parse_infantry_feedback_packet(candidate.data(), feedback)) {
        for (size_t i = 0; i < candidate.size(); ++i) buffer_.pop_front();
        return true;
      }
      buffer_.pop_front();
    }
    return false;
  }

  void clear() { buffer_.clear(); }

private:
  std::deque<uint8_t> buffer_;
};
}  // namespace io

#endif  // IO__GIMBAL__INFANTRY_PROTOCOL_HPP
