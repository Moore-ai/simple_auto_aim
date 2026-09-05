#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "serial/serial.h"
#include "io/gimbal/infantry_protocol.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct GimbalState
{
  // 下位机反馈包 byte[1] 的原始值；不参与上位机业务模式切换。
  uint8_t mode = 0;
  // yaw/pitch 为 SP 内部坐标系中的绝对反馈角；pitch 向上为负。
  float yaw = 0;
  float yaw_vel = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float bullet_speed = 0;
  uint16_t bullet_count = 0;
  float roll = 0;
};

struct GimbalCommand
{
  bool control = false;
  bool fire = false;
  // yaw/pitch 为 SP 内部坐标系中的绝对命令角；pitch 向上为负。
  float yaw = 0;
  float yaw_vel = 0;
  float yaw_acc = 0;
  float pitch = 0;
  float pitch_vel = 0;
  float pitch_acc = 0;
  float distance = -1;
};

struct GimbalStatePacket
{
  GimbalState state;
  std::array<uint8_t, kInfantryFeedbackPacketSize> packet;
};

struct GimbalCommandPacket
{
  GimbalCommand command;
  std::array<uint8_t, kInfantryCommandPacketSize> packet;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalState state() const;
  GimbalStatePacket state_with_packet() const;
  GimbalCommandPacket command_with_packet() const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  void send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc, float distance = 0);

private:
  serial::Serial serial_;
  bool virtual_serial_{false};
  bool command_angles_in_degrees_{false};
  bool feedback_angles_in_degrees_{false};
  std::array<uint8_t, kInfantryFeedbackPacketSize> virtual_feedback_packet_{};
  std::array<uint8_t, kInfantryCommandPacketSize> command_packet_{};
  std::array<uint8_t, kInfantryFeedbackPacketSize> feedback_packet_{};

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;
  InfantryFeedbackStreamParser rx_parser_;

  GimbalState state_;
  GimbalCommand command_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  void read_thread();
  void reconnect();
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
