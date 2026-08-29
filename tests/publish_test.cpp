#include <thread>

#include "io/http_sender.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

int main(int argc, char ** argv)
{
  tools::Exiter exiter;
  const std::string config_path = argc > 1 ? argv[1] : "configs/sentry.yaml";
  io::HttpSender http_sender(config_path);

  double i = 0;
  while (!exiter.exit()) {
    Eigen::Vector4d data{i, i + 1, 1, auto_aim::ArmorName::sentry + 1};
    http_sender.send(data);
    i++;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (i > 1000) break;
  }
  return 0;
}
