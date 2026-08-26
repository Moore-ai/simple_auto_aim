#include <cassert>

#include <yaml-cpp/yaml.h>

#include "io/hikrobot/hikrobot.hpp"

int main()
{
  const auto yaml = YAML::Load(R"(
exposure_ms: 2.5
gain: 16.9
camera_index: 2
image_width: 1280
image_height: 1024
fps: 120
flip_image: true
)");

  const auto config = io::load_hikrobot_config(yaml);
  assert(config.exposure_us == 2500.0);
  assert(config.gain == 16.9);
  assert(config.device_index == 2);
  assert(config.image_width == 1280);
  assert(config.image_height == 1024);
  assert(config.fps == 120.0);
  assert(config.flip_image);

  const auto defaults = io::load_hikrobot_config(YAML::Load("exposure_ms: 1\ngain: 0\n"));
  assert(defaults.device_index == 0);
  assert(!defaults.flip_image);

  const auto standard3 = io::load_hikrobot_config(YAML::LoadFile("configs/standard3.yaml"));
  assert(standard3.device_index == 0);
  assert(standard3.image_width == 1280);
  assert(standard3.image_height == 1024);
  assert(standard3.fps == 120.0);
  assert(standard3.exposure_us == 5500.0);
  assert(standard3.gain == 16.0);
  assert(standard3.flip_image);
  return 0;
}
