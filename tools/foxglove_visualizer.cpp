#include "foxglove_visualizer.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>

#include <foxglove/channel.hpp>
#include <foxglove/error.hpp>
#include <foxglove/schemas.hpp>
#include <foxglove/server.hpp>

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
}  // namespace

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

void detail::draw_detected_armors(
  cv::Mat & image, const std::list<auto_aim::Armor> & armors, auto_aim::Color target_color)
{
  for (const auto & armor : armors) {
    if (armor.color != target_color) continue;

    const cv::Scalar color =
      armor.color == auto_aim::red ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
    draw_polygon(image, armor.points, color);
    cv::putText(
      image,
      std::string(auto_aim::ARMOR_NAMES[armor.name]) + cv::format(" %.2f", armor.confidence),
      armor.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
  }
}

class FoxgloveVisualizer::Impl
{
public:
  std::optional<foxglove::WebSocketServer> server;
  std::optional<foxglove::RawChannel> serial_receive;
  std::optional<foxglove::RawChannel> serial_send;
  std::optional<foxglove::schemas::CompressedImageChannel> image_raw;
  std::optional<foxglove::schemas::CompressedImageChannel> image;
  std::optional<foxglove::schemas::CompressedImageChannel> image_detection;
  std::optional<foxglove::schemas::SceneUpdateChannel> target;
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

FoxgloveVisualizer::FoxgloveVisualizer() : impl_(std::make_unique<Impl>())
{
  foxglove::WebSocketServerOptions options;
  options.host = "0.0.0.0";
  options.port = 8765;
  options.name = "simple_auto_aim standard_mpc";
  auto server = foxglove::WebSocketServer::create(std::move(options));
  if (!server) {
    std::cerr << "Failed to start Foxglove server: " << foxglove::strerror(server.error()) << '\n';
    return;
  }
  impl_->server.emplace(std::move(server.value()));

  create(impl_->serial_receive, foxglove::RawChannel::create("/serial/receive", "json"),
         "/serial/receive");
  create(impl_->serial_send, foxglove::RawChannel::create("/serial/send", "json"), "/serial/send");
  create(impl_->image_raw, foxglove::schemas::CompressedImageChannel::create("/image_raw"),
         "/image_raw");
  create(impl_->image, foxglove::schemas::CompressedImageChannel::create("/image"), "/image");
  create(
    impl_->image_detection,
    foxglove::schemas::CompressedImageChannel::create("/image_detection"), "/image_detection");
  create(impl_->target, foxglove::schemas::SceneUpdateChannel::create("/target"), "/target");

  std::cout << "Foxglove server listening at ws://127.0.0.1:" << impl_->server->port() << '\n';
}

FoxgloveVisualizer::~FoxgloveVisualizer()
{
  if (impl_->server) impl_->server->stop();
}

void FoxgloveVisualizer::publish(const FrameSnapshot & frame)
{
  if (!impl_->server) return;

  const auto log_time = impl_->log_time(frame.timestamp);
  const auto & serial_receive = frame.gimbal_state;
  const auto & serial_send = frame.gimbal_command;
  const auto & target_color = frame.target_color;
  const auto & target_data = frame.tracker;
  cv::Mat image = frame.image.clone();

  log_json(
    impl_->serial_receive,
    Json{{"mode", serial_receive.mode}, {"roll", serial_receive.roll},
         {"yaw", serial_receive.yaw}, {"yaw_vel", serial_receive.yaw_vel},
         {"pitch", serial_receive.pitch}, {"pitch_vel", serial_receive.pitch_vel},
         {"bullet_speed", serial_receive.bullet_speed},
         {"bullet_count", serial_receive.bullet_count}},
    log_time);
  log_json(
    impl_->serial_send,
    Json{{"control", serial_send.control}, {"fire", serial_send.fire},
         {"yaw", serial_send.yaw}, {"yaw_vel", serial_send.yaw_vel},
         {"yaw_acc", serial_send.yaw_acc}, {"pitch", serial_send.pitch},
         {"pitch_vel", serial_send.pitch_vel}, {"pitch_acc", serial_send.pitch_acc},
         {"distance", serial_send.distance}},
    log_time);

  const auto * locked_armor =
    target_data.locked_armor && target_data.locked_armor->color == target_color ?
    &target_data.locked_armor.value() : nullptr;
  if (locked_armor) draw_polygon(image, locked_armor->points, {0, 0, 255});
  for (const auto & polygon : target_data.predicted_image_armors) draw_polygon(image, polygon, {255, 0, 0});

  cv::Mat detection_image = frame.image.clone();
  detail::draw_detected_armors(detection_image, frame.detections.armors, target_color);

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

  if (!impl_->target) return;
  foxglove::schemas::SceneUpdate update;
  update.deletions.push_back({std::nullopt,
                              foxglove::schemas::SceneEntityDeletion::SceneEntityDeletionType::ALL,
                              ""});
  if (locked_armor) {
    foxglove::schemas::SceneEntity detected;
    detected.id = "locked_armor";
    detected.frame_id = "world";
    auto cube = detail::armor_cube(
      locked_armor->xyz_in_world, locked_armor->ypr_in_world[0], locked_armor->ypr_in_world[1],
      locked_armor->type);
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

  if (target_data.target_state) {
    const auto & target_state = *target_data.target_state;
    foxglove::schemas::SceneEntity state;
    state.id = "target_state";
    state.frame_id = "world";
    const Eigen::Vector3d center(
      target_state.center_x(), target_state.center_y(), target_state.center_z());
    const auto vehicle_pitch = -std::atan2(center.z(), std::hypot(center.x(), center.y()));
    state.metadata = {{"center_x", std::to_string(target_state.center_x())},
                      {"velocity_x", std::to_string(target_state.velocity_x())},
                      {"center_y", std::to_string(target_state.center_y())},
                      {"velocity_y", std::to_string(target_state.velocity_y())},
                      {"center_z", std::to_string(target_state.center_z())},
                      {"velocity_z", std::to_string(target_state.velocity_z())},
                      {"vehicle_yaw", std::to_string(target_state.yaw())},
                      {"vehicle_pitch", std::to_string(vehicle_pitch)},
                      {"yaw_rate", std::to_string(target_state.yaw_rate())},
                      {"radius", std::to_string(target_state.radius())},
                      {"radius_diff", std::to_string(target_state.radius_diff())},
                      {"height_diff", std::to_string(target_state.height_diff())}};
    if (locked_armor) {
      state.metadata.push_back(
        {"selected_armor_yaw", std::to_string(locked_armor->ypr_in_world[0])});
      state.metadata.push_back(
        {"selected_armor_pitch", std::to_string(locked_armor->ypr_in_world[1])});
    }
    foxglove::schemas::TextPrimitive text;
    text.billboard = true;
    text.scale_invariant = true;
    text.font_size = 18;
    text.color = color(1.0, 1.0, 1.0);
    text.pose = pose(center, 0, 0);
    text.text = "vehicle yaw=" + std::to_string(target_state.yaw()) +
      " pitch=" + std::to_string(vehicle_pitch);
    if (locked_armor) {
      text.text += "\narmor yaw=" + std::to_string(locked_armor->ypr_in_world[0]) +
        " pitch=" + std::to_string(locked_armor->ypr_in_world[1]);
    }
    state.texts.push_back(std::move(text));
    update.entities.push_back(std::move(state));
  }
  impl_->target->log(update, log_time);
}
}  // namespace tools
