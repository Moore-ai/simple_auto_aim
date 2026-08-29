#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <cstring>
#include <future>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "io/http_sender.hpp"

namespace
{
std::string receive_request(int server_fd)
{
  const int client_fd = accept(server_fd, nullptr, nullptr);
  assert(client_fd >= 0);

  std::string request;
  char buffer[1024];
  size_t expected_size = std::string::npos;
  while (expected_size == std::string::npos || request.size() < expected_size) {
    const ssize_t count = recv(client_fd, buffer, sizeof(buffer), 0);
    assert(count > 0);
    request.append(buffer, static_cast<size_t>(count));

    const size_t header_end = request.find("\r\n\r\n");
    const size_t length_pos = request.find("Content-Length:");
    if (header_end != std::string::npos && length_pos != std::string::npos) {
      const size_t value_start = length_pos + std::strlen("Content-Length:");
      const size_t value_end = request.find("\r\n", value_start);
      const size_t content_length = std::stoul(request.substr(value_start, value_end - value_start));
      expected_size = header_end + 4 + content_length;
    }
  }

  constexpr char response[] =
    "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
  send(client_fd, response, sizeof(response) - 1, 0);
  close(client_fd);
  return request;
}
}  // namespace

int main()
{
  const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  assert(server_fd >= 0);

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  assert(bind(server_fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0);
  assert(listen(server_fd, 1) == 0);

  socklen_t address_size = sizeof(address);
  assert(getsockname(server_fd, reinterpret_cast<sockaddr *>(&address), &address_size) == 0);
  const auto port = ntohs(address.sin_port);

  auto request_future = std::async(std::launch::async, receive_request, server_fd);
  io::HttpSender sender("http://127.0.0.1:" + std::to_string(port) + "/auto_aim/target", 500);
  assert(sender.send(Eigen::Vector4d{1.25, -2.5, 1.0, 7.0}));

  const std::string request = request_future.get();
  close(server_fd);

  assert(request.rfind("POST /auto_aim/target HTTP/1.1\r\n", 0) == 0);
  assert(request.find("Content-Type: application/json") != std::string::npos);

  const size_t body_start = request.find("\r\n\r\n") + 4;
  const auto body = nlohmann::json::parse(request.substr(body_start));
  const nlohmann::json expected = {
    {"x", 1.25}, {"y", -2.5}, {"detected", true}, {"target_id", 7}};
  assert(body == expected);
  return 0;
}
