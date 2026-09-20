#include "foxglove_visualizer.hpp"
#include "tools/math_tools.hpp"

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <thread>
#include <utility>

#include <foxglove/channel.hpp>
#include <foxglove/error.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

#include "tools/yaml.hpp"

namespace tools
{
namespace
{
using Json = nlohmann::json;

foxglove::schemas::Color color(double r, double g, double b)
{
  return {r, g, b, 1.0};
}

foxglove::schemas::Pose pose(const Eigen::Vector3d & center, double yaw, double pitch)
{
  const Eigen::Quaterniond q{Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())};
  foxglove::schemas::Pose result;
  result.position = {center.x(), center.y(), center.z()};
  result.orientation = {q.x(), q.y(), q.z(), q.w()};
  return result;
}

void draw_polygon(cv::Mat & image, const std::vector<cv::Point2f> & points, const cv::Scalar & color)
{
  if (points.size() < 2) return;
  std::vector<std::vector<cv::Point>> polygon(1);
  polygon.front().reserve(points.size());
  for (const auto & point : points) polygon.front().emplace_back(cvRound(point.x), cvRound(point.y));
  cv::polylines(image, polygon, true, color, 2, cv::LINE_AA);
}

void draw_cross(
  cv::Mat & image, const std::vector<cv::Point2f> & corners, const cv::Scalar & color)
{
  if (corners.size() != 4) return;
  cv::line(image, corners[0], corners[2], color, 2, cv::LINE_AA);
  cv::line(image, corners[1], corners[3], color, 2, cv::LINE_AA);
}

foxglove::schemas::CompressedImage compressed_image(const cv::Mat & image)
{
  std::vector<uint8_t> encoded;
  cv::imencode(".jpg", image, encoded);
  foxglove::schemas::CompressedImage result;
  result.frame_id = "camera";
  result.format = "jpeg";
  result.data.assign(
    reinterpret_cast<const std::byte *>(encoded.data()),
    reinterpret_cast<const std::byte *>(encoded.data() + encoded.size()));
  return result;
}

template <typename Channel>
bool create(
  std::optional<Channel> & result, foxglove::FoxgloveResult<Channel> && channel, const char * topic)
{
  if (channel) {
    result.emplace(std::move(channel.value()));
    return true;
  }
  std::cerr << "Failed to create Foxglove channel " << topic << ": "
            << foxglove::strerror(channel.error()) << '\n';
  return false;
}

template <typename Channel>
void log_json(std::optional<Channel> & channel, const Json & message, uint64_t log_time)
{
  if (!channel) return;
  const auto payload = message.dump();
  channel->log(reinterpret_cast<const std::byte *>(payload.data()), payload.size(), log_time);
}

const char * target_values_topic_name(detail::FoxgloveTargetTopic topic)
{
  switch (topic) {
    case detail::FoxgloveTargetTopic::normal: return "/target";
    case detail::FoxgloveTargetTopic::outpost_current: return "/outpost/current";
    case detail::FoxgloveTargetTopic::outpost_v2: return "/outpost/v2";
  }
  return "/target";
}

const char * target_values_schema_name(detail::FoxgloveTargetTopic topic)
{
  switch (topic) {
    case detail::FoxgloveTargetTopic::normal: return "simple_auto_aim.TargetState";
    case detail::FoxgloveTargetTopic::outpost_current:
      return "simple_auto_aim.OutpostState";
    case detail::FoxgloveTargetTopic::outpost_v2: return "simple_auto_aim.OutpostStateV2";
  }
  return "simple_auto_aim.TargetState";
}

const std::string & target_values_schema_data(detail::FoxgloveTargetTopic topic)
{
  static const std::array<std::string, 3> schemas = [] {
    const auto number = Json{{"type", "number"}};
    Json common_properties = {
      {"center_x", number},       {"velocity_x", number}, {"center_y", number},
      {"velocity_y", number},    {"center_z", number},   {"velocity_z", number},
      {"vehicle_yaw", number},   {"vehicle_pitch", number},
      {"yaw_rate", number}};
    auto make_schema = [](const Json & properties) {
      return Json{{"$schema", "http://json-schema.org/draft-07/schema#"},
                  {"type", "object"},
                  {"properties", properties},
                  {"additionalProperties", false}}
        .dump();
    };

    Json normal_properties = common_properties;
    normal_properties.update(
      {{"radius", number}, {"radius_diff", number}, {"height_diff", number}});
    Json v2_properties = common_properties;
    v2_properties.update({{"height_offset_1", number}, {"height_offset_2", number}});
    return std::array<std::string, 3>{
      make_schema(normal_properties), make_schema(common_properties), make_schema(v2_properties)};
  }();
  return schemas[static_cast<std::size_t>(topic)];
}

const std::string & angular_acceleration_schema_data()
{
  static const auto schema = Json{
    {"$schema", "http://json-schema.org/draft-07/schema#"},
    {"type", "object"},
    {"properties",
     {{"yaw_acc", {{"type", "number"}, {"description", "yaw angular acceleration (rad/s^2)"}}},
      {"pitch_acc",
       {{"type", "number"}, {"description", "pitch angular acceleration (rad/s^2)"}}}}},
    {"required", Json::array({"yaw_acc", "pitch_acc"})},
    {"additionalProperties", false}}
    .dump();
  return schema;
}

const std::string & angular_error_schema_data()
{
  static const auto schema = Json{
    {"$schema", "http://json-schema.org/draft-07/schema#"},
    {"type", "object"},
    {"properties",
     {{"yaw_planner_error", {{"type", "number"}, {"description", "yaw_ref - yaw_mpc (rad)"}}},
      {"pitch_planner_error",
       {{"type", "number"}, {"description", "pitch_ref - pitch_mpc (rad)"}}},
      {"yaw_tracking_error",
       {{"type", "number"}, {"description", "yaw_mpc - yaw_gimbal (rad)"}}},
      {"pitch_tracking_error",
       {{"type", "number"}, {"description", "pitch_mpc - pitch_gimbal (rad)"}}}}},
    {"required", Json::array({
       "yaw_planner_error", "pitch_planner_error", "yaw_tracking_error", "pitch_tracking_error"})},
    {"additionalProperties", false}}
    .dump();
  return schema;
}
}  // namespace

detail::FoxgloveConfig detail::load_foxglove_config(const YAML::Node & yaml)
{
  FoxgloveConfig config;
  const auto foxglove = yaml["foxglove"];
  if (foxglove) {
    if (foxglove["enable"]) config.enable = foxglove["enable"].as<bool>();
    if (foxglove["image_fps"]) config.image_fps = foxglove["image_fps"].as<double>();
  }
  if (!std::isfinite(config.image_fps) || config.image_fps <= 0.0) {
    throw std::invalid_argument("foxglove.image_fps must be positive");
  }
  return config;
}

detail::ImagePublishLimiter::ImagePublishLimiter(double fps)
: period_{std::chrono::duration_cast<FrameSnapshot::Timestamp::duration>(
    std::chrono::duration<double>(1.0 / fps))}
{
}

bool detail::ImagePublishLimiter::should_publish(FrameSnapshot::Timestamp timestamp)
{
  if (
    last_publish_time_ && timestamp >= *last_publish_time_ &&
    timestamp - *last_publish_time_ < period_) {
    return false;
  }
  last_publish_time_ = timestamp;
  return true;
}

void detail::LatestFrameQueue::push(FrameSnapshot frame)
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) return;
    latest_frame_ = std::move(frame);
  }
  ready_.notify_one();
}

bool detail::LatestFrameQueue::wait_and_pop(FrameSnapshot & frame)
{
  std::unique_lock<std::mutex> lock(mutex_);
  ready_.wait(lock, [this] { return stopped_ || latest_frame_.has_value(); });
  if (!latest_frame_) return false;
  frame = std::move(*latest_frame_);
  latest_frame_.reset();
  return true;
}

void detail::LatestFrameQueue::stop()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    latest_frame_.reset();
  }
  ready_.notify_all();
}

cv::Mat detail::prepare_image_for_publish(const cv::Mat & image)
{
  return image;
}

foxglove::schemas::CubePrimitive detail::armor_cube(
  const Eigen::Vector3d & center, double yaw, double pitch, auto_aim::ArmorType armor_type)
{
  foxglove::schemas::CubePrimitive cube;
  cube.pose = pose(center, yaw, pitch);
  cube.size = {0.020, armor_type == auto_aim::big ? 0.230 : 0.135, 0.130};
  return cube;
}

detail::FoxgloveTargetTopic detail::target_topic(const auto_aim::TrackerDebugData & target_data)
{
  if (!target_data.outpost_snapshot) return FoxgloveTargetTopic::normal;
  if (std::holds_alternative<auto_aim::OutpostStateV2>(
        target_data.outpost_snapshot->debug_state)) {
    return FoxgloveTargetTopic::outpost_v2;
  }
  return FoxgloveTargetTopic::outpost_current;
}

const char * detail::target_topic_name(FoxgloveTargetTopic topic)
{
  switch (topic) {
    case FoxgloveTargetTopic::normal: return "/target/scene";
    case FoxgloveTargetTopic::outpost_current: return "/outpost/current/scene";
    case FoxgloveTargetTopic::outpost_v2: return "/outpost/v2/scene";
  }
  return "/target/scene";
}

foxglove::FoxgloveResult<foxglove::RawChannel> detail::create_target_values_channel(
  FoxgloveTargetTopic topic)
{
  const auto & schema_data = target_values_schema_data(topic);
  foxglove::Schema schema{
    target_values_schema_name(topic), "jsonschema",
    reinterpret_cast<const std::byte *>(schema_data.data()), schema_data.size()};
  return foxglove::RawChannel::create(target_values_topic_name(topic), "json", std::move(schema));
}

nlohmann::json detail::angular_acceleration_values(const io::GimbalCommand & command)
{
  return Json{{"yaw_acc", command.yaw_acc}, {"pitch_acc", command.pitch_acc}};
}

foxglove::FoxgloveResult<foxglove::RawChannel> detail::create_angular_acceleration_channel()
{
  const auto & schema_data = angular_acceleration_schema_data();
  foxglove::Schema schema{
    "simple_auto_aim.AngularAcceleration", "jsonschema",
    reinterpret_cast<const std::byte *>(schema_data.data()), schema_data.size()};
  return foxglove::RawChannel::create(
    "/planner/angular_acceleration", "json", std::move(schema));
}

nlohmann::json detail::command_packet_values(
  const std::array<uint8_t, io::kInfantryCommandPacketSize> & bytes)
{
  io::InfantryCommandPacket packet;
  std::memcpy(&packet, bytes.data(), bytes.size());
  const auto pitch = packet.pitch;
  const auto yaw = packet.yaw;
  const auto distance = packet.distance;
  const auto pitch_vel = packet.pitch_vel;
  const auto yaw_vel = packet.yaw_vel;
  const auto pitch_acc = packet.pitch_acc;
  const auto yaw_acc = packet.yaw_acc;
  return Json{
    {"fire", packet.fire},          {"pitch", pitch},
    {"yaw", yaw},                   {"distance", distance},
    {"pitch_vel", pitch_vel},       {"yaw_vel", yaw_vel},
    {"pitch_acc", pitch_acc},       {"yaw_acc", yaw_acc}};
}

nlohmann::json detail::feedback_packet_values(
  const std::array<uint8_t, io::kInfantryFeedbackPacketSize> & bytes)
{
  io::InfantryFeedbackPacket packet;
  std::memcpy(&packet, bytes.data(), bytes.size());
  const auto roll = packet.roll;
  const auto pitch = packet.pitch;
  const auto yaw = packet.yaw;
  return Json{
    {"mode", packet.mode},          {"roll", roll},
    {"pitch", pitch},               {"yaw", yaw},
    {"reserved", std::vector<uint8_t>(
                    packet.reserved, packet.reserved + sizeof(packet.reserved))}};
}

nlohmann::json detail::angular_error_values(
  const auto_aim::Plan & plan, const io::GimbalState & gimbal_state)
{
  return Json{
    {"yaw_planner_error", tools::limit_rad(plan.target_yaw - plan.yaw)},
    {"pitch_planner_error", plan.target_pitch - plan.pitch},
    {"yaw_tracking_error", tools::limit_rad(plan.yaw - gimbal_state.yaw)},
    {"pitch_tracking_error", plan.pitch - gimbal_state.pitch}};
}

foxglove::FoxgloveResult<foxglove::RawChannel> detail::create_angular_error_channel()
{
  const auto & schema_data = angular_error_schema_data();
  foxglove::Schema schema{
    "simple_auto_aim.AngularError", "jsonschema",
    reinterpret_cast<const std::byte *>(schema_data.data()), schema_data.size()};
  return foxglove::RawChannel::create("/planner/angular_error", "json", std::move(schema));
}

Json detail::target_values(const auto_aim::TrackerDebugData & target_data)
{
  Json values = Json::object();
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double vehicle_yaw = 0.0;
  double yaw_rate = 0.0;

  const auto topic = detail::target_topic(target_data);
  if (topic == FoxgloveTargetTopic::normal && target_data.target_state) {
    const auto & target = *target_data.target_state;
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    values = {{"center_x", target.center_x()}, {"velocity_x", target.velocity_x()},
              {"center_y", target.center_y()}, {"velocity_y", target.velocity_y()},
              {"center_z", target.center_z()}, {"velocity_z", target.velocity_z()},
              {"vehicle_yaw", vehicle_yaw},       {"yaw_rate", yaw_rate},
              {"radius", target.radius()},         {"radius_diff", target.radius_diff()},
              {"height_diff", target.height_diff()}};
  } else if (
    topic == FoxgloveTargetTopic::outpost_current && target_data.outpost_snapshot &&
    std::holds_alternative<auto_aim::OutpostState>(target_data.outpost_snapshot->debug_state)) {
    const auto & target =
      std::get<auto_aim::OutpostState>(target_data.outpost_snapshot->debug_state);
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    values = {{"center_x", target.center_x()}, {"velocity_x", target.velocity_x()},
              {"center_y", target.center_y()}, {"velocity_y", target.velocity_y()},
              {"center_z", target.center_z()}, {"velocity_z", target.velocity_z()},
              {"vehicle_yaw", vehicle_yaw},       {"yaw_rate", yaw_rate}};
  } else if (
    topic == FoxgloveTargetTopic::outpost_v2 && target_data.outpost_snapshot &&
    std::holds_alternative<auto_aim::OutpostStateV2>(target_data.outpost_snapshot->debug_state)) {
    const auto & target =
      std::get<auto_aim::OutpostStateV2>(target_data.outpost_snapshot->debug_state);
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    values = {{"center_x", target.center_x()}, {"velocity_x", target.velocity_x()},
              {"center_y", target.center_y()}, {"velocity_y", target.velocity_y()},
              {"center_z", target.center_z()}, {"velocity_z", target.velocity_z()},
              {"vehicle_yaw", vehicle_yaw},       {"yaw_rate", yaw_rate},
              {"height_offset_1", target.height_offset_1()},
              {"height_offset_2", target.height_offset_2()}};
  } else {
    return values;
  }

  values["vehicle_pitch"] = -std::atan2(center.z(), std::hypot(center.x(), center.y()));
  return values;
}

foxglove::schemas::SceneUpdate detail::target_scene_update(
  const auto_aim::TrackerDebugData & target_data)
{
  foxglove::schemas::SceneUpdate update;
  update.deletions.push_back({std::nullopt,
                              foxglove::schemas::SceneEntityDeletion::SceneEntityDeletionType::ALL,
                              ""});

  if (target_data.locked_armor) {
    foxglove::schemas::SceneEntity detected;
    detected.id = "locked_armor";
    detected.frame_id = "world";
    auto cube = detail::armor_cube(
      target_data.locked_armor->xyz_in_world, target_data.locked_armor->ypr_in_world[0],
      target_data.locked_armor->ypr_in_world[1], target_data.locked_armor->type);
    cube.color = color(1.0, 0.0, 0.0);
    detected.cubes.push_back(std::move(cube));
    update.entities.push_back(std::move(detected));
  }

  for (size_t i = 0; i < target_data.predicted_world_armors.size(); ++i) {
    const auto & armor = target_data.predicted_world_armors[i];
    foxglove::schemas::SceneEntity predicted;
    predicted.id = "predicted_armor_" + std::to_string(i);
    predicted.frame_id = "world";
    auto cube = detail::armor_cube(armor.center, armor.yaw, armor.pitch, target_data.armor_type);
    cube.color = color(0.0, 0.0, 1.0);
    predicted.cubes.push_back(std::move(cube));
    update.entities.push_back(std::move(predicted));
  }

  foxglove::schemas::SceneEntity state;
  state.id = "target_state";
  state.frame_id = "world";
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  double vehicle_yaw = 0.0;
  double yaw_rate = 0.0;
  bool has_state = false;

  const auto topic = detail::target_topic(target_data);
  if (topic == FoxgloveTargetTopic::normal && target_data.target_state) {
    const auto & target = *target_data.target_state;
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    state.metadata = {{"model", "normal"}};
    state.metadata.insert(state.metadata.end(),
                          {{"center_x", std::to_string(target.center_x())},
                           {"velocity_x", std::to_string(target.velocity_x())},
                           {"center_y", std::to_string(target.center_y())},
                           {"velocity_y", std::to_string(target.velocity_y())},
                           {"center_z", std::to_string(target.center_z())},
                           {"velocity_z", std::to_string(target.velocity_z())},
                           {"vehicle_yaw", std::to_string(target.yaw())},
                           {"yaw_rate", std::to_string(target.yaw_rate())},
                           {"radius", std::to_string(target.radius())},
                           {"radius_diff", std::to_string(target.radius_diff())},
                           {"height_diff", std::to_string(target.height_diff())}});
    has_state = true;
  } else if (
    topic == FoxgloveTargetTopic::outpost_current && target_data.outpost_snapshot &&
    std::holds_alternative<auto_aim::OutpostState>(target_data.outpost_snapshot->debug_state)) {
    const auto & target = std::get<auto_aim::OutpostState>(target_data.outpost_snapshot->debug_state);
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    state.metadata = {{"model", "current"}};
    state.metadata.insert(state.metadata.end(),
                          {{"center_x", std::to_string(target.center_x())},
                           {"velocity_x", std::to_string(target.velocity_x())},
                           {"center_y", std::to_string(target.center_y())},
                           {"velocity_y", std::to_string(target.velocity_y())},
                           {"center_z", std::to_string(target.center_z())},
                           {"velocity_z", std::to_string(target.velocity_z())},
                           {"vehicle_yaw", std::to_string(target.yaw())},
                           {"yaw_rate", std::to_string(target.yaw_rate())}});
    has_state = true;
  } else if (
    topic == FoxgloveTargetTopic::outpost_v2 && target_data.outpost_snapshot &&
    std::holds_alternative<auto_aim::OutpostStateV2>(target_data.outpost_snapshot->debug_state)) {
    const auto & target =
      std::get<auto_aim::OutpostStateV2>(target_data.outpost_snapshot->debug_state);
    center = {target.center_x(), target.center_y(), target.center_z()};
    vehicle_yaw = target.yaw();
    yaw_rate = target.yaw_rate();
    state.metadata = {{"model", "v2"}};
    state.metadata.insert(state.metadata.end(),
                          {{"center_x", std::to_string(target.center_x())},
                           {"velocity_x", std::to_string(target.velocity_x())},
                           {"center_y", std::to_string(target.center_y())},
                           {"velocity_y", std::to_string(target.velocity_y())},
                           {"center_z", std::to_string(target.center_z())},
                           {"velocity_z", std::to_string(target.velocity_z())},
                           {"vehicle_yaw", std::to_string(target.yaw())},
                           {"yaw_rate", std::to_string(target.yaw_rate())},
                           {"height_offset_1", std::to_string(target.height_offset_1())},
                           {"height_offset_2", std::to_string(target.height_offset_2())}});
    has_state = true;
  }

  if (!has_state) return update;

  const auto vehicle_pitch = -std::atan2(center.z(), std::hypot(center.x(), center.y()));
  state.metadata.push_back({"vehicle_pitch", std::to_string(vehicle_pitch)});
  state.metadata.push_back({"ekf_converged", target_data.ekf_converged ? "true" : "false"});
  if (target_data.locked_armor) {
    state.metadata.push_back(
      {"selected_armor_yaw", std::to_string(target_data.locked_armor->ypr_in_world[0])});
    state.metadata.push_back(
      {"selected_armor_pitch", std::to_string(target_data.locked_armor->ypr_in_world[1])});
  }
  foxglove::schemas::TextPrimitive text;
  text.billboard = true;
  text.scale_invariant = true;
  text.font_size = 18;
  text.color = color(1.0, 1.0, 1.0);
  text.pose = pose(center, 0, 0);
  text.text = "vehicle yaw=" + std::to_string(vehicle_yaw) +
    " pitch=" + std::to_string(vehicle_pitch) + " yaw_rate=" + std::to_string(yaw_rate) +
    "\nekf converged=" + (target_data.ekf_converged ? "true" : "false");
  if (target_data.locked_armor) {
    text.text += "\narmor yaw=" + std::to_string(target_data.locked_armor->ypr_in_world[0]) +
      " pitch=" + std::to_string(target_data.locked_armor->ypr_in_world[1]);
  }
  state.texts.push_back(std::move(text));
  update.entities.push_back(std::move(state));
  return update;
}

void detail::draw_aim_overlay(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors,
  auto_aim::Color enemy_color,
  const auto_aim::Armor * locked_armor, const std::vector<cv::Point2f> * anti_spin_hit_armor)
{
  for (const auto & armor : armors) {
    if (armor.color != enemy_color) continue;
    draw_cross(image, armor.points, {0, 0, 255});
  }

  if (anti_spin_hit_armor) {
    draw_polygon(image, *anti_spin_hit_armor, {0, 0, 255});
  }
  if (locked_armor) draw_polygon(image, locked_armor->points, {0, 255, 255});
}

void detail::draw_buff_overlay(cv::Mat & image, const BuffDebugData & debug_data)
{
  const cv::Scalar yellow{0, 255, 255};
  const cv::Scalar green{0, 255, 0};
  for (const auto & detection : debug_data.detections) {
    cv::circle(image, detection.point, 5, green, 2, cv::LINE_AA);
    for (const auto & corner : detection.corners)
      cv::circle(image, corner, 3, green, 1, cv::LINE_AA);
    if (!detection.label.empty())
      cv::putText(image, detection.label, detection.point, cv::FONT_HERSHEY_SIMPLEX, 0.45, green,
                  1, cv::LINE_AA);
  }
  for (const auto & point : debug_data.reprojected_features)
    cv::circle(image, point, 4, yellow, 2, cv::LINE_AA);

  if (debug_data.blade_polygon) {
    std::vector<cv::Point2f> blades(
      debug_data.blade_polygon->begin(), debug_data.blade_polygon->end());
    draw_polygon(image, blades, yellow);
    if (debug_data.icon) {
      cv::Point2f center;
      for (const auto & blade : *debug_data.blade_polygon) center += blade;
      center *= 1.0F / static_cast<float>(debug_data.blade_polygon->size());
      cv::line(image, *debug_data.icon, center, yellow, 2, cv::LINE_AA);
    }
  }
  if (debug_data.info_anchor && !debug_data.info.empty())
    cv::putText(image, debug_data.info, *debug_data.info_anchor, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                {255, 255, 255}, 1, cv::LINE_AA);
  if (debug_data.aimpoint) {
    const cv::Scalar aimpoint_color = debug_data.aimpoint_fire ? cv::Scalar{0, 0, 255} : green;
    cv::circle(image, *debug_data.aimpoint, 5, aimpoint_color, -1, cv::LINE_AA);
  }
}

std::optional<cv::Point2f> detail::buff_aimpoint(
  const auto_aim::Plan & plan, std::uint64_t plan_target_generation,
  std::uint64_t current_target_generation, auto_aim::Solver & solver)
{
  if (plan_target_generation != current_target_generation || !plan.control || !plan.debug_valid ||
      !plan.debug_xyza.allFinite()) {
    return std::nullopt;
  }
  const auto pixels = solver.world2pixel(
    {{static_cast<float>(plan.debug_xyza.x()), static_cast<float>(plan.debug_xyza.y()),
      static_cast<float>(plan.debug_xyza.z())}});
  if (pixels.size() != 1) return std::nullopt;
  return pixels.front();
}

std::optional<std::vector<cv::Point2f>> detail::anti_spin_hit_armor(
  const auto_aim::Plan & plan, std::uint64_t plan_target_generation,
  std::uint64_t current_target_generation, auto_aim::ArmorType armor_type,
  const auto_aim::Solver & solver)
{
  if (
    plan_target_generation != current_target_generation || !plan.anti_spin_active ||
    !plan.debug_valid || !plan.debug_xyza.allFinite()) {
    return std::nullopt;
  }

  auto points = solver.reproject_armor(
    plan.debug_xyza.head<3>(), plan.debug_xyza[3], plan.debug_armor_pitch, armor_type);
  if (points.size() != 4) return std::nullopt;
  return points;
}

class FoxgloveVisualizer::Impl
{
public:
  Impl(auto_aim::Solver & solver, detail::FoxgloveConfig config)
  : solver{solver}, enabled{config.enable}, image_limiter{config.image_fps}
  {
  }

  auto_aim::Solver solver;
  bool enabled;
  detail::ImagePublishLimiter image_limiter;
  detail::LatestFrameQueue frames;
  std::thread worker;
  std::mutex plan_mutex;
  std::optional<std::pair<std::uint64_t, auto_aim::Plan>> latest_plan;
  std::optional<foxglove::WebSocketServer> server;
  std::optional<foxglove::RawChannel> serial_receive;
  std::optional<foxglove::RawChannel> serial_send;
  std::optional<foxglove::RawChannel> angular_acceleration;
  std::optional<foxglove::RawChannel> angular_error;
  std::optional<foxglove::schemas::CompressedImageChannel> image_raw;
  std::optional<foxglove::schemas::CompressedImageChannel> image;
  std::optional<foxglove::schemas::CompressedImageChannel> image_detection;
  std::optional<foxglove::RawChannel> target_values;
  std::optional<foxglove::RawChannel> outpost_current_values;
  std::optional<foxglove::RawChannel> outpost_v2_values;
  std::optional<foxglove::schemas::SceneUpdateChannel> target;
  std::optional<foxglove::schemas::SceneUpdateChannel> outpost_current;
  std::optional<foxglove::schemas::SceneUpdateChannel> outpost_v2;
  auto_aim::Color enemy_color = auto_aim::Color::blue;
  const std::chrono::steady_clock::time_point steady_origin = std::chrono::steady_clock::now();
  const std::chrono::system_clock::time_point system_origin = std::chrono::system_clock::now();

  uint64_t log_time(FrameSnapshot::Timestamp timestamp) const
  {
    const auto elapsed = timestamp - steady_origin;
    const auto wall_time = system_origin +
      std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed);
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_time.time_since_epoch()).count());
  }
};

FoxgloveVisualizer::FoxgloveVisualizer(
  auto_aim::Solver & solver, const std::string & config_path)
: impl_(std::make_unique<Impl>(
    solver, detail::load_foxglove_config(tools::load(config_path))))
{
  if (!impl_->enabled) return;

  foxglove::WebSocketServerOptions options;
  options.host = "0.0.0.0";
  options.port = 8765;
  options.name = "simple_auto_aim standard";
  auto server = foxglove::WebSocketServer::create(std::move(options));
  if (!server) {
    std::cerr << "Failed to start Foxglove server: " << foxglove::strerror(server.error()) << '\n';
    return;
  }
  impl_->server.emplace(std::move(server.value()));

  create(impl_->serial_receive, foxglove::RawChannel::create("/serial/receive", "json"),
         "/serial/receive");
  create(impl_->serial_send, foxglove::RawChannel::create("/serial/send", "json"), "/serial/send");
  create(
    impl_->angular_acceleration, detail::create_angular_acceleration_channel(),
    "/planner/angular_acceleration");
  create(
    impl_->angular_error, detail::create_angular_error_channel(), "/planner/angular_error");
  create(impl_->image_raw, foxglove::schemas::CompressedImageChannel::create("/image_raw"),
         "/image_raw");
  create(impl_->image, foxglove::schemas::CompressedImageChannel::create("/image"), "/image");
  create(
    impl_->image_detection,
    foxglove::schemas::CompressedImageChannel::create("/image_detection"), "/image_detection");
  create(impl_->target_values,
         detail::create_target_values_channel(detail::FoxgloveTargetTopic::normal), "/target");
  create(impl_->outpost_current_values,
         detail::create_target_values_channel(detail::FoxgloveTargetTopic::outpost_current),
         "/outpost/current");
  create(impl_->outpost_v2_values,
         detail::create_target_values_channel(detail::FoxgloveTargetTopic::outpost_v2),
         "/outpost/v2");
  create(
    impl_->target,
    foxglove::schemas::SceneUpdateChannel::create(
      detail::target_topic_name(detail::FoxgloveTargetTopic::normal)),
    detail::target_topic_name(detail::FoxgloveTargetTopic::normal));
  create(
    impl_->outpost_current,
    foxglove::schemas::SceneUpdateChannel::create(
      detail::target_topic_name(detail::FoxgloveTargetTopic::outpost_current)),
    detail::target_topic_name(detail::FoxgloveTargetTopic::outpost_current));
  create(
    impl_->outpost_v2,
    foxglove::schemas::SceneUpdateChannel::create(
      detail::target_topic_name(detail::FoxgloveTargetTopic::outpost_v2)),
    detail::target_topic_name(detail::FoxgloveTargetTopic::outpost_v2));

  impl_->worker = std::thread([this] {
    FrameSnapshot frame;
    while (impl_->frames.wait_and_pop(frame)) publish_frame(frame);
  });
  std::cout << "Foxglove server listening at ws://127.0.0.1:" << impl_->server->port() << '\n';
}

FoxgloveVisualizer::~FoxgloveVisualizer()
{
  impl_->frames.stop();
  if (impl_->worker.joinable()) impl_->worker.join();
  if (impl_->server) impl_->server->stop();
}

void FoxgloveVisualizer::update_plan(
  std::uint64_t target_generation, const auto_aim::Plan & plan)
{
  if (!impl_->server) return;
  std::lock_guard<std::mutex> lock(impl_->plan_mutex);
  impl_->latest_plan = std::make_pair(target_generation, plan);
}

void FoxgloveVisualizer::publish(FrameSnapshot frame)
{
  if (!impl_->server) return;
  impl_->frames.push(std::move(frame));
}

void FoxgloveVisualizer::publish_frame(const FrameSnapshot & frame)
{
  if (!impl_->server) return;

  const auto log_time = impl_->log_time(frame.timestamp);
  const auto & serial_receive = frame.gimbal_state;
  const auto & serial_send = frame.gimbal_command;
  const auto & target_data = frame.tracker;

  if (const auto color = io::infantry_enemy_color(serial_receive.mode)) {
    impl_->enemy_color = *color == io::InfantryEnemyColor::red ?
      auto_aim::Color::red : auto_aim::Color::blue;
  }

  log_json(
    impl_->serial_receive,
    detail::feedback_packet_values(frame.serial_receive_packet),
    log_time);
  log_json(
    impl_->serial_send,
    detail::command_packet_values(frame.serial_send_packet),
    log_time);
  log_json(
    impl_->angular_acceleration, detail::angular_acceleration_values(serial_send), log_time);

  std::optional<std::pair<std::uint64_t, auto_aim::Plan>> latest_plan;
  {
    std::lock_guard<std::mutex> lock(impl_->plan_mutex);
    latest_plan = impl_->latest_plan;
  }
  if (latest_plan && latest_plan->second.control) {
    log_json(
      impl_->angular_error,
      detail::angular_error_values(latest_plan->second, serial_receive), log_time);
  }
  if (impl_->image_limiter.should_publish(frame.timestamp)) {
    const auto * locked_armor =
      target_data.locked_armor ? &target_data.locked_armor.value() : nullptr;
    auto buff_debug = frame.buff_debug;
    std::optional<std::vector<cv::Point2f>> hit_armor;
    if (latest_plan) {
      impl_->solver.set_R_gimbal2world(frame.gimbal_orientation);
      hit_armor = detail::anti_spin_hit_armor(
        latest_plan->second, latest_plan->first, frame.target_generation, target_data.armor_type,
        impl_->solver);
      if (buff_debug.is_buff_mode) {
        if (const auto aimpoint = detail::buff_aimpoint(
              latest_plan->second, latest_plan->first, frame.target_generation, impl_->solver)) {
          buff_debug.aimpoint = *aimpoint;
          buff_debug.aimpoint_fire = latest_plan->second.fire;
        }
      }
    }
    const auto * anti_spin_hit_armor = hit_armor ? &hit_armor.value() : nullptr;
    cv::Mat image = frame.image.clone();
    for (const auto & polygon : target_data.predicted_image_armors) {
      draw_polygon(image, polygon, {255, 0, 0});
    }
    detail::draw_aim_overlay(
      image, frame.detections.armors, impl_->enemy_color, locked_armor, anti_spin_hit_armor);
    detail::draw_buff_overlay(image, buff_debug);

    cv::Mat detection_image = frame.image.clone();
    detail::draw_aim_overlay(
      detection_image, frame.detections.armors, impl_->enemy_color, locked_armor,
      anti_spin_hit_armor);

    if (impl_->image_raw) {
      impl_->image_raw->log(
        compressed_image(detail::prepare_image_for_publish(frame.image)), log_time);
    }
    if (impl_->image) {
      impl_->image->log(compressed_image(detail::prepare_image_for_publish(image)), log_time);
    }
    if (impl_->image_detection) {
      impl_->image_detection->log(
        compressed_image(detail::prepare_image_for_publish(detection_image)), log_time);
    }
  }

  const auto active_topic = detail::target_topic(target_data);
  const auto active_values = detail::target_values(target_data);
  const Json empty_values = Json::object();
  log_json(impl_->target_values,
           active_topic == detail::FoxgloveTargetTopic::normal ? active_values : empty_values,
           log_time);
  log_json(impl_->outpost_current_values,
           active_topic == detail::FoxgloveTargetTopic::outpost_current ? active_values :
                                                                         empty_values,
           log_time);
  log_json(impl_->outpost_v2_values,
           active_topic == detail::FoxgloveTargetTopic::outpost_v2 ? active_values : empty_values,
           log_time);
  const auto active_update = detail::target_scene_update(target_data);
  const auto clear_update = detail::target_scene_update({});
  if (impl_->target) {
    impl_->target->log(
      active_topic == detail::FoxgloveTargetTopic::normal ? active_update : clear_update, log_time);
  }
  if (impl_->outpost_current) {
    impl_->outpost_current->log(
      active_topic == detail::FoxgloveTargetTopic::outpost_current ? active_update : clear_update,
      log_time);
  }
  if (impl_->outpost_v2) {
    impl_->outpost_v2->log(
      active_topic == detail::FoxgloveTargetTopic::outpost_v2 ? active_update : clear_update,
      log_time);
  }
}
}  // namespace tools
