#include "detection_result.hpp"

#include <string>

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
DetectionOptions DetectionOptions::from_yaml(const YAML::Node & yaml)
{
  DetectionOptions options;
  const auto observation_mode =
    yaml["observation_mode"] ? yaml["observation_mode"].as<std::string>() : "pnp";
  const auto filter_method =
    yaml["filter_method"] ? yaml["filter_method"].as<std::string>() : "ekf";

  if (filter_method != "ekf") return options;
  if (observation_mode == "reprojection") {
    options.collect_lightbars = true;
    return options;
  }

  const auto pnp_config = yaml["pnp"];
  if (pnp_config && pnp_config["enable_lightbar_assist"]) {
    options.collect_lightbars = pnp_config["enable_lightbar_assist"].as<bool>();
  }
  return options;
}
}  // namespace auto_aim
