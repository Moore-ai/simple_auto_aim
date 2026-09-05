#include <cassert>
#include <chrono>
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

  std::filesystem::remove(path);
  return 0;
}
