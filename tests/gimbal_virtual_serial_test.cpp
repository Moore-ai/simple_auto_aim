#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include "io/gimbal/gimbal.hpp"

int main()
{
  auto config = YAML::LoadFile("configs/standard.yaml");
  config["virtual_serial"]["enable"] = true;
  config["virtual_serial"]["feedback"]["mode"] = 1;
  config["virtual_serial"]["feedback"]["roll"] = 0.1F;
  config["virtual_serial"]["feedback"]["pitch"] = 0.2F;
  config["virtual_serial"]["feedback"]["yaw"] = -0.3F;

  const auto path = std::filesystem::temp_directory_path() / "simple_auto_aim_virtual_serial.yaml";
  {
    std::ofstream output(path);
    assert(output);
    output << config;
  }

  {
    io::Gimbal gimbal(path.string());
    const auto initial_q = gimbal.q(std::chrono::steady_clock::now());
    assert(initial_q.coeffs().allFinite());
    const auto configured_state = gimbal.state();
    assert(configured_state.mode == 1);
    assert(configured_state.roll == 0.1F);
    assert(configured_state.pitch == -0.2F);
    assert(configured_state.yaw == 0.3F);

    gimbal.send(true, false, 0.3F, 0.1F, 0.0F, -0.2F, 0.0F, 0.0F, 5.0F);
    const auto command = gimbal.command_with_packet().command;
    assert(command.control);
    assert(command.distance == 5.0F);

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const auto state = gimbal.state();
    assert(state.mode == 1);
    assert(state.roll == 0.1F);
    assert(state.pitch == -0.2F);
    assert(state.yaw == 0.3F);
  }

  config["command_angles_in_degrees"] = true;
  config["feedback_angles_in_degrees"] = true;
  config["virtual_serial"]["feedback"]["roll"] = 10.0F;
  config["virtual_serial"]["feedback"]["pitch"] = 20.0F;
  config["virtual_serial"]["feedback"]["yaw"] = -30.0F;
  {
    std::ofstream degree_output(path);
    assert(degree_output);
    degree_output << config;
  }

  {
    const auto packet = io::make_infantry_feedback_packet(1, 10.0F, 20.0F, -30.0F);
    io::InfantryFeedbackStreamParser parser;
    parser.set_feedback_angles_in_degrees(true);
    parser.push(packet.data(), packet.size());
    io::InfantryFeedback feedback{};
    assert(parser.pop(feedback));
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kRadToDeg = 180.0F / kPi;
    assert(std::abs(feedback.roll - 10.0F / kRadToDeg) < 1e-6F);
    assert(std::abs(feedback.pitch + 20.0F / kRadToDeg) < 1e-6F);
    assert(std::abs(feedback.yaw - 30.0F / kRadToDeg) < 1e-6F);
  }

  {
    io::Gimbal gimbal(path.string());
    constexpr float kPi = 3.14159265358979323846F;
    constexpr float kRadToDeg = 180.0F / kPi;
    const auto state = gimbal.state();
    assert(std::abs(state.roll - 10.0F / kRadToDeg) < 1e-6F);
    assert(std::abs(state.pitch + 20.0F / kRadToDeg) < 1e-6F);
    assert(std::abs(state.yaw - 30.0F / kRadToDeg) < 1e-6F);

    gimbal.send(true, false, 0.3F, 0.1F, 0.2F, -0.2F, -0.1F, -0.4F, 5.0F);
    const auto packet = gimbal.command_with_packet().packet;
    io::InfantryCommandPacket raw_command;
    std::memcpy(&raw_command, packet.data(), sizeof(raw_command));
    assert(std::abs(raw_command.pitch - 0.2F * kRadToDeg) < 1e-5F);
    assert(std::abs(raw_command.yaw + 0.3F * kRadToDeg) < 1e-5F);
    assert(std::abs(raw_command.pitch_vel - 0.1F * kRadToDeg) < 1e-5F);
    assert(std::abs(raw_command.yaw_vel + 0.1F * kRadToDeg) < 1e-5F);
    assert(std::abs(raw_command.pitch_acc - 0.4F * kRadToDeg) < 1e-5F);
    assert(std::abs(raw_command.yaw_acc + 0.2F * kRadToDeg) < 1e-5F);
  }

  std::filesystem::remove(path);
  return 0;
}
