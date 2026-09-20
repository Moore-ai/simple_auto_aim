#include "src/mode_runtime.hpp"

static_assert(static_cast<int>(standard::Mode::auto_aim) == 0);
static_assert(static_cast<int>(standard::Mode::small_buff) == 1);
static_assert(static_cast<int>(standard::Mode::big_buff) == 2);
static_assert(standard::is_valid_mode(0));
static_assert(standard::is_valid_mode(1));
static_assert(standard::is_valid_mode(2));
static_assert(!standard::is_valid_mode(-1));
static_assert(!standard::is_valid_mode(3));
static_assert(standard::mode_from_value(0) == standard::Mode::auto_aim);
static_assert(standard::mode_from_value(1) == standard::Mode::small_buff);
static_assert(standard::mode_from_value(2) == standard::Mode::big_buff);
static_assert(standard::mode_from_value(-1) == standard::Mode::auto_aim);
static_assert(standard::mode_from_value(3) == standard::Mode::auto_aim);

int main() { return 0; }
