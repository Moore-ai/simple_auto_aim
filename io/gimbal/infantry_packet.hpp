#ifndef IO__GIMBAL__INFANTRY_PACKET_HPP
#define IO__GIMBAL__INFANTRY_PACKET_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace io
{
// 下位机串口线上的原始数据包。两端均按 IEEE 754 小端序传输 float。
struct __attribute__((packed)) InfantryCommandPacket
{
  uint8_t start = 0xFF;
  uint8_t fire = 0;
  float pitch_abs = 0;
  float yaw_abs = 0;
  float distance = 0;
  float pitch_vel = 0;
  float yaw_vel = 0;
  float pitch_acc = 0;
  float yaw_acc = 0;
  uint8_t crc8 = 0;
  uint8_t end = 0x0D;
};

struct __attribute__((packed)) InfantryFeedbackPacket
{
  uint8_t start = 0xFF;
  uint8_t mode = 0;
  float roll = 0;
  float pitch = 0;
  float yaw = 0;
  uint8_t reserved[8]{};
  uint8_t crc8 = 0;
  uint8_t end = 0x0D;
};

constexpr size_t kInfantryCommandPacketSize = sizeof(InfantryCommandPacket);
constexpr size_t kInfantryFeedbackPacketSize = sizeof(InfantryFeedbackPacket);

static_assert(std::is_standard_layout_v<InfantryCommandPacket>);
static_assert(std::is_standard_layout_v<InfantryFeedbackPacket>);
static_assert(kInfantryCommandPacketSize == 32);
static_assert(kInfantryFeedbackPacketSize == 24);
static_assert(offsetof(InfantryCommandPacket, pitch_abs) == 2);
static_assert(offsetof(InfantryCommandPacket, yaw_abs) == 6);
static_assert(offsetof(InfantryFeedbackPacket, roll) == 2);
static_assert(offsetof(InfantryFeedbackPacket, crc8) == 22);
}  // namespace io

#endif  // IO__GIMBAL__INFANTRY_PACKET_HPP
