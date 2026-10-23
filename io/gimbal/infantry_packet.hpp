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
  // bit0=开火建议。
  uint8_t mode = 0;
  float pitch = 0;
  float yaw = 0;
  float distance = 0;
  int32_t target_id = -1;
  float pitch_vel = 0;
  float yaw_vel = 0;
  // 偏移 26..61：导航、裁判系统和保留字段，暂时不使用。
  uint8_t reserved[36]{};
  uint8_t crc8 = 0;
  uint8_t end = 0x0D;
};

struct __attribute__((packed)) InfantryFeedbackPacket
{
  uint8_t start = 0xFF;
  uint8_t mode = 0;
  float roll = 0;
  float yaw = 0;
  float pitch = 0;
  // 偏移 14..61：导航、裁判系统和保留字段，暂时不使用。
  uint8_t reserved[48]{};
  uint8_t crc8 = 0;
  uint8_t end = 0x0D;
};

constexpr size_t kInfantryCommandPacketSize = sizeof(InfantryCommandPacket);
constexpr size_t kInfantryFeedbackPacketSize = sizeof(InfantryFeedbackPacket);

static_assert(std::is_standard_layout_v<InfantryCommandPacket>);
static_assert(std::is_standard_layout_v<InfantryFeedbackPacket>);
static_assert(kInfantryCommandPacketSize == 64);
static_assert(kInfantryFeedbackPacketSize == 64);
static_assert(offsetof(InfantryCommandPacket, pitch) == 2);
static_assert(offsetof(InfantryCommandPacket, yaw) == 6);
static_assert(offsetof(InfantryCommandPacket, target_id) == 14);
static_assert(offsetof(InfantryCommandPacket, pitch_vel) == 18);
static_assert(offsetof(InfantryCommandPacket, yaw_vel) == 22);
static_assert(offsetof(InfantryCommandPacket, crc8) == 62);
static_assert(offsetof(InfantryCommandPacket, end) == 63);
static_assert(offsetof(InfantryFeedbackPacket, roll) == 2);
static_assert(offsetof(InfantryFeedbackPacket, yaw) == 6);
static_assert(offsetof(InfantryFeedbackPacket, pitch) == 10);
static_assert(offsetof(InfantryFeedbackPacket, crc8) == 62);
static_assert(offsetof(InfantryFeedbackPacket, end) == 63);
}  // namespace io

#endif  // IO__GIMBAL__INFANTRY_PACKET_HPP
