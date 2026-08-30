#include <cassert>

#include <yaml-cpp/yaml.h>

#include "tools/detect_factory.hpp"

int main()
{
  assert(tools::detector_debug_window_enabled(YAML::Load("debug_window: true")));
  assert(!tools::detector_debug_window_enabled(YAML::Load("debug_window: false")));
  assert(tools::detector_debug_window_enabled(YAML::Load("detect_method: yolo")));
}
