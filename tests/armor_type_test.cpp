#include <cassert>

#include "tasks/auto_aim/armor.hpp"

int main()
{
  const auto expected_type = [](auto_aim::ArmorName name) {
    return name == auto_aim::ArmorName::one ? auto_aim::ArmorType::big
                                             : auto_aim::ArmorType::small;
  };

  for (const auto name : {auto_aim::ArmorName::one, auto_aim::ArmorName::two,
                          auto_aim::ArmorName::three, auto_aim::ArmorName::four,
                          auto_aim::ArmorName::five, auto_aim::ArmorName::sentry,
                          auto_aim::ArmorName::outpost, auto_aim::ArmorName::base}) {
    assert(auto_aim::armor_type_for_name(name) == expected_type(name));
    assert(auto_aim::armor_type_matches_name(expected_type(name), name));
    assert(!auto_aim::armor_type_matches_name(
      expected_type(name) == auto_aim::ArmorType::big ? auto_aim::ArmorType::small
                                                       : auto_aim::ArmorType::big,
      name));
  }

  for (const auto & [color, name, type] : auto_aim::armor_properties)
    assert(type == expected_type(name));
}
