#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <thread>

#include <fcntl.h>
#include <pty.h>
#include <signal.h>
#include <unistd.h>

#include "io/gimbal/infantry_protocol.hpp"

namespace
{
std::atomic<bool> quit = false;

void stop(int)
{
  quit = true;
}
}  // namespace

int main(int argc, char * argv[])
{
  const std::string link_path = argc > 1 ? argv[1] : "/tmp/sp_vision_25_gimbal";

  int master_fd = -1;
  int slave_fd = -1;
  char slave_name[256]{};
  if (openpty(&master_fd, &slave_fd, slave_name, nullptr, nullptr) != 0) {
    std::cerr << "Failed to create virtual serial port: " << std::strerror(errno) << '\n';
    return 1;
  }
  close(slave_fd);

  const std::filesystem::path link(link_path);
  if (std::filesystem::exists(link) && !std::filesystem::is_symlink(link)) {
    std::cerr << "Refusing to replace non-symlink path: " << link_path << '\n';
    close(master_fd);
    return 1;
  }
  std::filesystem::remove(link);
  if (symlink(slave_name, link_path.c_str()) != 0) {
    std::cerr << "Failed to create serial symlink: " << std::strerror(errno) << '\n';
    close(master_fd);
    return 1;
  }

  fcntl(master_fd, F_SETFL, fcntl(master_fd, F_GETFL) | O_NONBLOCK);
  signal(SIGINT, stop);
  signal(SIGTERM, stop);

  std::cout << "Virtual gimbal serial: " << link_path << " -> " << slave_name << std::endl;
  const auto packet = io::make_infantry_feedback_packet(0, 0.0F, 0.0F, 0.0F);
  std::array<uint8_t, 256> commands{};
  while (!quit) {
    const auto written = write(master_fd, packet.data(), packet.size());
    if (written < 0 && errno != EAGAIN && errno != EIO) {
      std::cerr << "Failed to write feedback: " << std::strerror(errno) << '\n';
      break;
    }
    while (read(master_fd, commands.data(), commands.size()) > 0) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  close(master_fd);
  std::filesystem::remove(link);
}
