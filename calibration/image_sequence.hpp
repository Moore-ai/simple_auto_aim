#pragma once

#include <charconv>
#include <filesystem>

namespace calibration
{

inline int last_image_index(const std::filesystem::path & folder)
{
  int last_index = 0;
  for (const auto & entry : std::filesystem::directory_iterator(folder)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".jpg") continue;

    const auto name = entry.path().stem().string();
    int index = 0;
    const auto [end, error] = std::from_chars(name.data(), name.data() + name.size(), index);
    if (error == std::errc{} && end == name.data() + name.size()) last_index = std::max(last_index, index);
  }
  return last_index;
}

}  // namespace calibration
