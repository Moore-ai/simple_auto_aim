#include "gimbal.hpp"

#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");
  const auto baudrate = yaml["baudrate"] ? yaml["baudrate"].as<uint32_t>()
                                          : kInfantryDefaultBaudrate;
  const auto bytesize_node = yaml["bytesize"]
                               ? yaml["bytesize"]
                               : (yaml["byte_size"] ? yaml["byte_size"] : yaml["data_bits"]);
  const auto bytesize = bytesize_node ? bytesize_node.as<int>() : kInfantryDefaultBytesize;

  try {
    if (!is_supported_infantry_bytesize(bytesize)) {
      throw std::invalid_argument("bytesize must be between 5 and 8");
    }
    serial_.setPort(com_port);
    serial_.setBaudrate(baudrate);
    serial_.setBytesize(static_cast<serial::bytesize_t>(bytesize));
    serial::Timeout timeout(serial::Timeout::max(), kInfantryReadTimeoutMs, 0,
                            kInfantryWriteTimeoutMs, 0);
    serial_.setTimeout(timeout);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  send(
    VisionToGimbal.mode != 0, VisionToGimbal.mode == 2, VisionToGimbal.yaw,
    VisionToGimbal.yaw_vel, VisionToGimbal.yaw_acc, VisionToGimbal.pitch,
    VisionToGimbal.pitch_vel, VisionToGimbal.pitch_acc, VisionToGimbal.distance);
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc, float distance)
{
  if (!control) {
    const auto current = state();
    yaw = current.yaw;
    yaw_vel = 0;
    yaw_acc = 0;
    pitch = current.pitch;
    pitch_vel = 0;
    pitch_acc = 0;
    distance = -1;
  }

  const auto packet = make_infantry_command_packet(
    control, fire, pitch, yaw, distance, pitch_vel, yaw_vel, pitch_acc, yaw_acc);

  try {
    serial_.write(packet.data(), packet.size());
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    if (error_count > 5000) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Too many errors, attempting to reconnect...");
      reconnect();
      continue;
    }

    uint8_t buffer[kInfantryFeedbackPacketSize]{};
    size_t received = 0;
    try {
      received = serial_.read(buffer, sizeof(buffer));
    } catch (const std::exception & e) {
      error_count++;
      continue;
    }
    if (received == 0) continue;
    rx_parser_.push(buffer, received);

    InfantryFeedback feedback;
    while (rx_parser_.pop(feedback)) {
      const auto t = std::chrono::steady_clock::now();
      const auto q = infantry_feedback_quaternion(feedback.roll, feedback.pitch, feedback.yaw);
      queue_.push({q, t});

      std::lock_guard<std::mutex> lock(mutex_);

      state_.yaw = feedback.yaw;
      state_.yaw_vel = 0;
      state_.pitch = infantry_feedback_pitch_to_sp(feedback.pitch);
      state_.pitch_vel = 0;
      state_.bullet_speed = 0;
      state_.bullet_count = 0;

      switch (feedback.mode) {
        case 0:
          mode_ = GimbalMode::IDLE;
          break;
        case 1:
          mode_ = GimbalMode::AUTO_AIM;
          break;
        case 2:
          mode_ = GimbalMode::SMALL_BUFF;
          break;
        case 3:
          mode_ = GimbalMode::BIG_BUFF;
          break;
        default:
          mode_ = GimbalMode::IDLE;
          tools::logger()->warn("[Gimbal] Invalid mode: {}", feedback.mode);
          break;
      }
    }
    error_count = 0;
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (...) {
    }

    try {
      serial_.open();  // 尝试重新打开
      queue_.clear();
      rx_parser_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io
