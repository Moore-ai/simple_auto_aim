#include <cassert>
#include <chrono>
#include <fstream>
#include <list>
#include <sstream>

#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"

int main()
{
  std::ifstream config_input("configs/demo.yaml");
  std::stringstream config_buffer;
  config_buffer << config_input.rdbuf();
  auto config_text = config_buffer.str();
  assert(config_text.find("enemy_color:") == std::string::npos);
  config_text.insert(0, "image_width: 640\nimage_height: 480\n");

  constexpr auto config_path = "/tmp/tracker_test.yaml";
  std::ofstream config_output(config_path);
  config_output << config_text;
  config_output.close();

  auto solver = auto_aim::Solver(config_path);
  auto tracker = auto_aim::Tracker(config_path, solver);
  // 模拟下位机首包：蓝色不能因默认值相同而跳过 ObservationPath 的同步。
  tracker.set_enemy_color(auto_aim::Color::blue);
  const auto points = solver.reproject_armor(
    {2.0, 0.0, 0.0}, 0.0, auto_aim::ArmorType::small, auto_aim::ArmorName::sentry);

  auto configured_center_armor = auto_aim::Armor(0, 0.9F, cv::Rect{}, points);
  configured_center_armor.center = {320.0F, 240.0F};
  auto hard_coded_center_armor = auto_aim::Armor(0, 0.9F, cv::Rect{}, points);
  hard_coded_center_armor.center = {720.0F, 540.0F};

  std::list<auto_aim::Armor> armors{
    hard_coded_center_armor,
    configured_center_armor,
  };
  tracker.track(armors, std::chrono::steady_clock::now());

  assert(armors.front().center == configured_center_armor.center);

  const auto legacy_priority_config =
    "use_priority: true\npriority_list: [one, sentry]\n" + config_text;
  constexpr auto legacy_priority_config_path = "/tmp/tracker_legacy_priority_test.yaml";
  std::ofstream legacy_priority_output(legacy_priority_config_path);
  legacy_priority_output << legacy_priority_config;
  legacy_priority_output.close();

  auto legacy_priority_solver = auto_aim::Solver(legacy_priority_config_path);
  auto legacy_priority_tracker =
    auto_aim::Tracker(legacy_priority_config_path, legacy_priority_solver);
  legacy_priority_tracker.set_enemy_color(auto_aim::Color::blue);

  auto centered_sentry = auto_aim::Armor(0, 0.9F, cv::Rect{}, points);
  centered_sentry.center = {320.0F, 240.0F};
  auto off_center_hero = auto_aim::Armor(3, 0.9F, cv::Rect{}, points);
  off_center_hero.center = {600.0F, 400.0F};
  std::list<auto_aim::Armor> legacy_priority_armors{off_center_hero, centered_sentry};

  legacy_priority_tracker.track(legacy_priority_armors, std::chrono::steady_clock::now());

  assert(legacy_priority_armors.front().name == auto_aim::ArmorName::sentry);
  return 0;
}
