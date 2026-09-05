#include <cassert>
#include <cmath>
#include <cstring>
#include <list>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>

#include "tools/foxglove_visualizer.hpp"

int main()
{
  cv::Mat input(1, 3, CV_8UC3);
  input.at<cv::Vec3b>(0, 0) = {1, 2, 3};
  input.at<cv::Vec3b>(0, 1) = {4, 5, 6};
  input.at<cv::Vec3b>(0, 2) = {7, 8, 9};

  const auto output = tools::detail::prepare_image_for_publish(input);
  assert(cv::norm(input, output, cv::NORM_INF) == 0.0);

  const std::vector<cv::Point2f> red_points = {{10, 10}, {30, 10}, {30, 30}, {10, 30}};
  const std::vector<cv::Point2f> blue_points = {{60, 10}, {80, 10}, {80, 30}, {60, 30}};
  auto red_armor = auto_aim::Armor(1, 0.9F, {10, 10, 20, 20}, red_points);
  auto blue_armor = auto_aim::Armor(0, 0.8F, {60, 10, 20, 20}, blue_points);
  std::list<auto_aim::Armor> armors = {red_armor, blue_armor};

  cv::Mat detection_image = cv::Mat::zeros(40, 90, CV_8UC3);
  tools::detail::draw_aim_overlay(detection_image, armors, nullptr, nullptr);
  const auto detected_corner = detection_image.at<cv::Vec3b>(10, 10);
  assert(detected_corner[0] == 0);
  assert(detected_corner[1] > 0 && detected_corner[2] > 0);
  const auto detected_center = detection_image.at<cv::Vec3b>(20, 20);
  assert(detected_center[0] == 0);
  assert(detected_center[1] > 0);
  assert(detected_center[2] > 0);
  const auto other_color_center = detection_image.at<cv::Vec3b>(20, 70);
  assert(other_color_center[1] > 0 && other_color_center[2] > 0);
  assert(detection_image.at<cv::Vec3b>(30, 20) == cv::Vec3b(0, 0, 0));

  const std::vector<cv::Point2f> second_red_points = {{60, 10}, {80, 10}, {80, 30}, {60, 30}};
  auto second_red_armor = auto_aim::Armor(1, 0.8F, {60, 10, 20, 20}, second_red_points);
  std::list<auto_aim::Armor> red_armors = {red_armor, second_red_armor};
  cv::Mat tracking_image = cv::Mat::zeros(80, 100, CV_8UC3);
  tools::detail::draw_aim_overlay(
    tracking_image, red_armors, &red_armors.front(), nullptr);
  const auto locked_corner = tracking_image.at<cv::Vec3b>(10, 10);
  assert(locked_corner[2] > 0 && locked_corner[1] == 0);
  const auto locked_center = tracking_image.at<cv::Vec3b>(20, 20);
  const auto detected_only_center = tracking_image.at<cv::Vec3b>(20, 70);
  assert(locked_center[2] > 0 && locked_center[1] == 0);
  assert(detected_only_center[1] > 0 && detected_only_center[2] > 0);

  const std::vector<cv::Point2f> predicted_hit = {{35, 45}, {65, 45}, {65, 65}, {35, 65}};
  cv::Mat anti_spin_image = cv::Mat::zeros(80, 100, CV_8UC3);
  tools::detail::draw_aim_overlay(
    anti_spin_image, red_armors, &red_armors.front(), &predicted_hit);
  const auto anti_spin_locked_center = anti_spin_image.at<cv::Vec3b>(20, 20);
  const auto predicted_hit_edge = anti_spin_image.at<cv::Vec3b>(45, 50);
  const auto predicted_hit_center = anti_spin_image.at<cv::Vec3b>(55, 50);
  assert(anti_spin_locked_center[1] > 0 && anti_spin_locked_center[2] > 0);
  assert(predicted_hit_edge[2] > 0 && predicted_hit_edge[1] == 0);
  assert(predicted_hit_center == cv::Vec3b(0, 0, 0));

  auto_aim::Solver projection_solver("configs/standard.yaml");
  auto_aim::Plan inactive_plan;
  inactive_plan.debug_valid = true;
  inactive_plan.debug_xyza = {2.0, 0.0, 0.0, 0.2};
  assert(!tools::detail::anti_spin_hit_armor(
            inactive_plan, 1, 1, auto_aim::ArmorType::small, projection_solver)
            .has_value());

  auto active_plan = inactive_plan;
  active_plan.anti_spin_active = true;
  assert(!tools::detail::anti_spin_hit_armor(
            active_plan, 1, 2, auto_aim::ArmorType::small, projection_solver)
            .has_value());
  const auto small_hit = tools::detail::anti_spin_hit_armor(
    active_plan, 2, 2, auto_aim::ArmorType::small, projection_solver);
  const auto big_hit = tools::detail::anti_spin_hit_armor(
    active_plan, 2, 2, auto_aim::ArmorType::big, projection_solver);
  assert(small_hit && small_hit->size() == 4);
  assert(big_hit && big_hit->size() == 4);
  assert(cv::norm((*big_hit)[0] - (*big_hit)[1]) > cv::norm((*small_hit)[0] - (*small_hit)[1]));

  const double yaw = 0.3;
  const double pitch = CV_PI / 12.0;
  const auto cube = tools::detail::armor_cube({-0.3, 0.0, 0.0}, yaw, pitch, auto_aim::big);
  assert(cube.pose.has_value());
  assert(cube.pose->orientation.has_value());
  const auto & orientation = *cube.pose->orientation;
  const Eigen::Quaterniond visualization_rotation{
    orientation.w, orientation.x, orientation.y, orientation.z};
  const Eigen::Quaterniond expected_rotation{
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())};
  assert(visualization_rotation.angularDistance(expected_rotation) < 1e-12);

  assert(cube.size.has_value());
  assert(std::abs(cube.size->x - 0.020) < 1e-12);
  assert(std::abs(cube.size->y - 0.230) < 1e-12);
  assert(std::abs(cube.size->z - 0.130) < 1e-12);

  auto_aim::TrackerDebugData normal_target;
  normal_target.target_state = auto_aim::TargetState(Eigen::VectorXd::Zero(11));
  normal_target.ekf_converged = true;
  assert(
    tools::detail::target_topic(normal_target) == tools::detail::FoxgloveTargetTopic::normal);
  assert(
    std::strcmp(
      tools::detail::target_topic_name(tools::detail::FoxgloveTargetTopic::normal),
      "/target/scene") == 0);
  const auto normal_scene = tools::detail::target_scene_update(normal_target);
  assert(normal_scene.entities.size() == 1);
  assert(normal_scene.entities.front().metadata.front().key == "model");
  assert(normal_scene.entities.front().metadata.front().value == "normal");
  const auto normal_values = tools::detail::target_values(normal_target);
  assert(normal_values.is_object());
  assert(normal_values.size() == 12);
  assert(normal_values.at("center_x").is_number());
  assert(normal_values.at("velocity_x").is_number());
  assert(normal_values.at("center_y").is_number());
  assert(normal_values.at("velocity_y").is_number());
  assert(normal_values.at("center_z").is_number());
  assert(normal_values.at("velocity_z").is_number());
  assert(normal_values.at("vehicle_yaw").is_number());
  assert(normal_values.at("vehicle_pitch").is_number());
  assert(normal_values.at("yaw_rate").is_number());
  assert(normal_values.at("radius").is_number());
  assert(normal_values.at("radius_diff").is_number());
  assert(normal_values.at("height_diff").is_number());

  auto_aim::TrackerDebugData current_outpost;
  current_outpost.outpost_snapshot = auto_aim::OutpostSnapshot{
    auto_aim::TargetState(Eigen::VectorXd::Zero(11)),
    auto_aim::OutpostState(Eigen::VectorXd::Zero(8)), Eigen::VectorXd::Zero(8), {}, {}, 0.0,
    false, true};
  assert(
    tools::detail::target_topic(current_outpost) ==
    tools::detail::FoxgloveTargetTopic::outpost_current);
  assert(
    std::strcmp(
      tools::detail::target_topic_name(tools::detail::FoxgloveTargetTopic::outpost_current),
      "/outpost/current/scene") == 0);

  auto_aim::TrackerDebugData v2_outpost;
  v2_outpost.target_state = auto_aim::TargetState(Eigen::VectorXd::Ones(11));
  v2_outpost.outpost_snapshot = auto_aim::OutpostSnapshot{
    auto_aim::TargetState(Eigen::VectorXd::Zero(11)),
    auto_aim::OutpostStateV2(Eigen::VectorXd::Zero(10)), Eigen::VectorXd::Zero(10), {}, {}, 0.0,
    false, true};
  v2_outpost.ekf_converged = true;
  const auto v2_scene = tools::detail::target_scene_update(v2_outpost);
  assert(
    tools::detail::target_topic(v2_outpost) == tools::detail::FoxgloveTargetTopic::outpost_v2);
  assert(
    std::strcmp(
      tools::detail::target_topic_name(tools::detail::FoxgloveTargetTopic::outpost_v2),
      "/outpost/v2/scene") == 0);
  assert(v2_scene.entities.size() == 1);
  const auto has_convergence_metadata = [](const auto & metadata) {
    for (const auto & [key, value] : metadata) {
      if (key == "ekf_converged") return value == "true";
    }
    return false;
  };
  assert(has_convergence_metadata(v2_scene.entities.front().metadata));
  const auto has_height_offset = [](const auto & metadata) {
    for (const auto & item : metadata) {
      if (item.key == "height_offset_2") return true;
    }
    return false;
  };
  assert(has_height_offset(v2_scene.entities.front().metadata));

  const auto v2_values = tools::detail::target_values(v2_outpost);
  assert(v2_values.at("center_x") == 0.0);
  assert(v2_values.at("height_offset_1").is_number());
  assert(v2_values.at("height_offset_2").is_number());

  auto normal_channel_result =
    tools::detail::create_target_values_channel(tools::detail::FoxgloveTargetTopic::normal);
  assert(normal_channel_result.has_value());
  auto normal_channel = std::move(normal_channel_result.value());
  assert(normal_channel.topic() == "/target");
  assert(normal_channel.message_encoding() == "json");
  const auto normal_schema = normal_channel.schema();
  assert(normal_schema.has_value());
  assert(normal_schema->encoding == "jsonschema");
  const auto normal_schema_json = nlohmann::json::parse(
    reinterpret_cast<const char *>(normal_schema->data),
    reinterpret_cast<const char *>(normal_schema->data) + normal_schema->data_len);
  assert(normal_schema_json.at("properties").contains("center_x"));
  assert(normal_schema_json.at("properties").contains("radius"));

  auto v2_channel_result =
    tools::detail::create_target_values_channel(tools::detail::FoxgloveTargetTopic::outpost_v2);
  assert(v2_channel_result.has_value());
  auto v2_channel = std::move(v2_channel_result.value());
  assert(v2_channel.topic() == "/outpost/v2");
  const auto v2_schema = v2_channel.schema();
  assert(v2_schema.has_value());
  assert(v2_schema->encoding == "jsonschema");
  const auto v2_schema_json = nlohmann::json::parse(
    reinterpret_cast<const char *>(v2_schema->data),
    reinterpret_cast<const char *>(v2_schema->data) + v2_schema->data_len);
  assert(v2_schema_json.at("properties").contains("height_offset_1"));
  assert(v2_schema_json.at("properties").contains("height_offset_2"));
  return 0;
}
