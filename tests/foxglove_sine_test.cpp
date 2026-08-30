#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <iostream>
#include <string>
#include <thread>

#include <foxglove/channel.hpp>
#include <foxglove/error.hpp>
#include <foxglove/server.hpp>

namespace
{
std::atomic<bool> quit = false;
constexpr double kTwoPi = 6.28318530717958647692;

void stop(int)
{
  quit = true;
}
}  // namespace

int main()
{
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);

  foxglove::WebSocketServerOptions options;
  options.host = "127.0.0.1";
  options.port = 8765;
  options.name = "simple_auto_aim sine test";
  auto server_result = foxglove::WebSocketServer::create(std::move(options));
  if (!server_result) {
    std::cerr << "Failed to start Foxglove server: " << foxglove::strerror(server_result.error())
              << '\n';
    return 1;
  }
  auto server = std::move(server_result.value());

  auto channel_result = foxglove::RawChannel::create("/sine", "json");
  if (!channel_result) {
    std::cerr << "Failed to create /sine channel: " << foxglove::strerror(channel_result.error())
              << '\n';
    return 1;
  }
  auto channel = std::move(channel_result.value());

  std::cout << "Foxglove server listening at ws://127.0.0.1:" << server.port()
            << ", publishing /sine.value at 50 Hz. Press Ctrl-C to stop.\n";

  const auto start = std::chrono::steady_clock::now();
  while (!quit) {
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    const auto value = std::sin(kTwoPi * elapsed.count());
    const std::string message = "{\"value\":" + std::to_string(value) + "}";
    channel.log(reinterpret_cast<const std::byte *>(message.data()), message.size());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  server.stop();
  return 0;
}
