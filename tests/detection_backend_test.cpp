#include <cassert>
#include <type_traits>

#include "tools/detect_factory.hpp"

namespace
{
class FakeBackend final : public tools::DetectionBackend
{
public:
  auto_aim::DetectionResult detect(const cv::Mat &, int) override
  {
    ++calls;
    return {};
  }

  int calls = 0;
};
}  // namespace

int main()
{
  FakeBackend backend;
  const auto result = backend.detect(cv::Mat::zeros(1, 1, CV_8UC3), 0);
  assert(result.armors.empty());
  assert(backend.calls == 1);
  static_assert(std::has_virtual_destructor_v<tools::DetectionBackend>);
  return 0;
}
