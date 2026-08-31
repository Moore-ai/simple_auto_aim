#include <cassert>
#include <filesystem>
#include <fstream>

#include "calibration/image_sequence.hpp"

int main()
{
  const auto folder = std::filesystem::temp_directory_path() / "calibration_image_sequence_test";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directory(folder);
  std::ofstream(folder / "1.jpg");
  std::ofstream(folder / "3.jpg");
  std::ofstream(folder / "not_an_image.jpg");

  assert(calibration::last_image_index(folder) == 3);

  std::filesystem::remove_all(folder);
  return 0;
}
