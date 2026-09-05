#ifndef IO__GIMBAL__INFANTRY_PROTOCOL_HPP
#define IO__GIMBAL__INFANTRY_PROTOCOL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <deque>
#include <optional>

#include <Eigen/Geometry>

#include "io/gimbal/infantry_packet.hpp"

namespace io
{
constexpr uint32_t kInfantryDefaultBaudrate = 115200;
constexpr int kInfantryDefaultBytesize = 8;
constexpr uint32_t kInfantryReadTimeoutMs = 2;
constexpr uint32_t kInfantryWriteTimeoutMs = 5;

// 下位机反馈模式中的目标颜色约定：0=红，1=蓝。
enum class InfantryEnemyColor : uint8_t
{
  red,
  blue,
};

inline std::optional<InfantryEnemyColor> infantry_enemy_color(uint8_t mode)
{
  switch (mode) {
    case 0:
      return InfantryEnemyColor::red;
    case 1:
      return InfantryEnemyColor::blue;
    default:
      return std::nullopt;
  }
}

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

inline float finite_or_zero(float value) { return std::isfinite(value) ? value : 0.0F; }

inline float infantry_yaw(float value) { return -value; }
inline float infantry_pitch(float value) { return -value; }

inline Eigen::Quaterniond infantry_feedback_quaternion(const InfantryFeedback & feedback)
{
  return Eigen::Quaterniond(
           Eigen::AngleAxisd(feedback.yaw, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(feedback.pitch, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(feedback.roll, Eigen::Vector3d::UnitX()))
    .normalized();
}

inline uint8_t infantry_crc8(const uint8_t * data, size_t size)
{
  uint8_t crc = 0;
  for (size_t i = 0; i < size; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) crc = crc & 0x80 ? (crc << 1) ^ 0x31 : crc << 1;
  }
  return crc;
}

inline std::array<uint8_t, kInfantryCommandPacketSize> make_infantry_command_packet(
  bool control, bool fire, float pitch_sp, float yaw_sp, float distance, float pitch_vel_sp,
  float yaw_vel_sp, float pitch_acc_sp, float yaw_acc_sp)
{
  InfantryCommandPacket command;
  command.fire = control && fire ? 1 : 0;
  command.pitch = infantry_pitch(finite_or_zero(pitch_sp));
  command.yaw = infantry_yaw(finite_or_zero(yaw_sp));
  command.distance = finite_or_zero(distance);
  command.pitch_vel = infantry_pitch(finite_or_zero(pitch_vel_sp));
  command.yaw_vel = infantry_yaw(finite_or_zero(yaw_vel_sp));
  command.pitch_acc = infantry_pitch(finite_or_zero(pitch_acc_sp));
  command.yaw_acc = infantry_yaw(finite_or_zero(yaw_acc_sp));
  command.crc8 = infantry_crc8(
    reinterpret_cast<const uint8_t *>(&command), offsetof(InfantryCommandPacket, crc8));

  std::array<uint8_t, kInfantryCommandPacketSize> packet{};
  std::memcpy(packet.data(), &command, sizeof(command));
  return packet;
}

inline bool parse_infantry_feedback_packet(const uint8_t * packet, InfantryFeedback & feedback)
{
  InfantryFeedbackPacket raw;
  std::memcpy(&raw, packet, sizeof(raw));
  if (raw.start != 0xFF || raw.end != 0x0D ||
      infantry_crc8(packet, offsetof(InfantryFeedbackPacket, crc8)) != raw.crc8) {
    return false;
  }

  feedback.mode = raw.mode;
  feedback.roll = raw.roll;
  feedback.pitch = infantry_pitch(raw.pitch);
  feedback.yaw = infantry_yaw(raw.yaw);
  return true;
}

inline std::array<uint8_t, kInfantryFeedbackPacketSize> make_infantry_feedback_packet(
  uint8_t mode, float roll, float pitch, float yaw)
{
  InfantryFeedbackPacket feedback;
  feedback.mode = mode;
  feedback.roll = std::isfinite(roll) ? roll : 0.0F;
  feedback.pitch = std::isfinite(pitch) ? pitch : 0.0F;
  feedback.yaw = std::isfinite(yaw) ? yaw : 0.0F;

  std::array<uint8_t, kInfantryFeedbackPacketSize> packet{};
  std::memcpy(packet.data(), &feedback, sizeof(feedback));
  packet[offsetof(InfantryFeedbackPacket, crc8)] =
    infantry_crc8(packet.data(), offsetof(InfantryFeedbackPacket, crc8));
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

  bool pop(
    InfantryFeedback & feedback,
    std::array<uint8_t, kInfantryFeedbackPacketSize> * raw_packet = nullptr)
  {
    while (buffer_.size() >= kInfantryFeedbackPacketSize) {
      while (!buffer_.empty() && buffer_.front() != 0xFF) buffer_.pop_front();
      if (buffer_.size() < kInfantryFeedbackPacketSize) return false;

      std::array<uint8_t, kInfantryFeedbackPacketSize> candidate{};
      for (size_t i = 0; i < candidate.size(); ++i) candidate[i] = buffer_[i];
      if (parse_infantry_feedback_packet(candidate.data(), feedback)) {
        if (raw_packet) *raw_packet = candidate;
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
