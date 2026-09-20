#ifndef STANDARD__MODE_RUNTIME_HPP
#define STANDARD__MODE_RUNTIME_HPP

#include <functional>
#include <string>

namespace io
{
class Gimbal;
}

namespace standard
{

enum class Mode
{
  auto_aim = 0,
  small_buff = 1,
  big_buff = 2,
};

constexpr bool is_valid_mode(int value) { return value >= 0 && value <= 2; }

constexpr Mode mode_from_value(int value)
{
  return is_valid_mode(value) ? static_cast<Mode>(value) : Mode::auto_aim;
}

class ModeRuntime
{
public:
  using ModeReader = std::function<Mode(const io::Gimbal &)>;

  ModeRuntime(std::string config_path, ModeReader mode_reader);

  int run();

private:
  std::string config_path_;
  ModeReader mode_reader_;
};

}  // namespace standard

#endif  // STANDARD__MODE_RUNTIME_HPP
