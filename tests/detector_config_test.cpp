#include <cassert>

#include <yaml-cpp/yaml.h>

#include "tools/detect_factory.hpp"

int main()
{
  assert(detector_debug_window_enabled(YAML::Load("debug_window: true")));
  assert(!detector_debug_window_enabled(YAML::Load("debug_window: false")));
  assert(detector_debug_window_enabled(YAML::Load("detect_method: yolo")));
}
