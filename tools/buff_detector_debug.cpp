#include "buff_detector_debug.hpp"

#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include <fmt/format.h>
#include <foxglove/error.hpp>
#include <foxglove/server.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/exiter.hpp"
#include "tools/foxglove_visualizer.hpp"
#include "tools/yaml.hpp"

namespace tools
{
using Json = nlohmann::json;

BuffDetectorDebug::BuffDetectorDebug(
  auto_buff_v2::BuffConfig::Detector config, BuffDetectorDebugTarget target)
: target_(target)
{
  static_cast<auto_buff_v2::BuffConfig::Detector &>(detector_.config) = std::move(config);
}

BuffDetectorDebugFrame BuffDetectorDebug::process(const FrameFacts & facts)
{
  if (const auto color = facts.enemy_color())
    detector_.config.enemy_red = *color == io::InfantryEnemyColor::red;
  BuffDetectorDebugFrame frame;
  frame.target = target_;
  const auto elements = detector_.detect(facts.image, &frame.measurements);
  auto & candidates = frame.measurements.candidates;
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const auto & item) {
    return item.radius_pass != (target_ == BuffDetectorDebugTarget::bullseye);
  }), candidates.end());

  BuffDebugData debug;
  if (target_ == BuffDetectorDebugTarget::icon) {
    debug.icon_count = elements.icons.size();
    for (const auto & icon : elements.icons)
      debug.detections.push_back({icon.center, {}, fmt::format("R: {:.3f}", icon.score)});
  } else {
    debug.bullseye_count = elements.bullseyes.size();
    for (const auto & bull : elements.bullseyes)
      debug.detections.push_back(
        {bull.center, bull.active ? std::vector<cv::Point2f>{} :
          std::vector<cv::Point2f>(bull.corners.begin(), bull.corners.end()),
         fmt::format("B: {:.3f}", bull.score)});
  }
  frame.snapshot = facts.snapshot({}, {}, {}, std::move(debug));
  return frame;
}
nlohmann::json detail::buff_detector_values(const auto_buff_v2::RuneDetectorMeasurements & debug)
{
  Json candidates = Json::array();
  for (const auto & candidate : debug.candidates) {
    candidates.push_back({
      {"radius", candidate.radius}, {"area", candidate.area},
      {"center_x", candidate.center.x}, {"center_y", candidate.center.y},
      {"radius_pass", candidate.radius_pass}, {"icon_area_pass", candidate.icon_area_pass}});
  }
  return {{"min_radius", debug.min_radius}, {"max_radius", debug.max_radius},
          {"min_icon_area", debug.min_icon_area}, {"max_icon_area", debug.max_icon_area},
          {"candidates", std::move(candidates)}};
}

foxglove::FoxgloveResult<foxglove::RawChannel> detail::create_buff_detector_channel()
{
  Json properties;
  for (const auto * name : {"min_radius", "max_radius", "min_icon_area", "max_icon_area"})
    properties[name] = {{"type", "number"}};
  Json candidate_properties;
  for (const auto * name : {"radius", "area", "center_x", "center_y"})
    candidate_properties[name] = {{"type", "number"}};
  for (const auto * name : {"radius_pass", "icon_area_pass"})
    candidate_properties[name] = {{"type", "boolean"}};
  properties["candidates"] = {
    {"type", "array"},
    {"items", {{"type", "object"}, {"properties", candidate_properties}}}};
  const auto schema_data = Json{
    {"$schema", "http://json-schema.org/draft-07/schema#"},
    {"type", "object"}, {"properties", properties}}.dump();
  foxglove::Schema schema{
    "simple_auto_aim.BuffDetector", "jsonschema",
    reinterpret_cast<const std::byte *>(schema_data.data()), schema_data.size()};
  return foxglove::RawChannel::create("/buff_v2/detector", "json", std::move(schema));
}

cv::Mat detail::buff_detector_debug_image(const BuffDetectorDebugFrame & frame)
{
  auto image = frame.snapshot.image.clone();
  const cv::Scalar yellow{0, 255, 255};
  const auto & candidates = frame.measurements.candidates;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto & candidate = candidates[i];
    cv::circle(image, candidate.center, cvRound(candidate.radius), yellow, 1, cv::LINE_AA);
    cv::putText(image, fmt::format("#{} r={} a={}", i, cvRound(candidate.radius),
                                  cvRound(candidate.area)),
                candidate.center + cv::Point2f(0, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                yellow, 1, cv::LINE_AA);
  }
  const auto & debug = frame.snapshot.buff_debug;
  const auto status = frame.target == BuffDetectorDebugTarget::icon ?
    fmt::format("Buff R:{}", debug.icon_count) : fmt::format("Buff B:{}/5", debug.bullseye_count);
  cv::putText(image, status, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2, cv::LINE_AA);
  // Use the main chain's green observation drawing, without its joint-model status.
  detail::draw_buff_overlay(image, debug);
  return image;
}

class BuffDetectorDebugVisualizer::Impl
{
public:
  explicit Impl(detail::FoxgloveConfig config) : limiter(config.image_fps) {}

  std::optional<foxglove::WebSocketServer> server;
  std::optional<foxglove::RawChannel> detector;
  std::optional<foxglove::schemas::CompressedImageChannel> image;
  std::optional<foxglove::schemas::CompressedImageChannel> image_raw;
  detail::ImagePublishLimiter limiter;
  std::mutex mutex;
  std::condition_variable ready;
  std::optional<BuffDetectorDebugFrame> latest;
  bool stopped = false;
  std::thread worker;
  const FrameSnapshot::Timestamp steady_origin = std::chrono::steady_clock::now();
  const std::chrono::system_clock::time_point system_origin = std::chrono::system_clock::now();

  void publish_frame(const BuffDetectorDebugFrame & frame)
  {
    const auto elapsed = frame.snapshot.timestamp - steady_origin;
    const auto wall_time = system_origin +
      std::chrono::duration_cast<std::chrono::system_clock::duration>(elapsed);
    const auto log_time = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(wall_time.time_since_epoch()).count());
    if (detector) {
      const auto payload = detail::buff_detector_values(frame.measurements).dump();
      detector->log(reinterpret_cast<const std::byte *>(payload.data()), payload.size(), log_time);
    }
    if (!limiter.should_publish(frame.snapshot.timestamp)) return;
    const auto log_image = [&](auto & channel, const cv::Mat & pixels) {
      if (!channel) return;
      std::vector<uint8_t> encoded;
      cv::imencode(".jpg", detail::prepare_image_for_publish(pixels), encoded);
      foxglove::schemas::CompressedImage message;
      message.frame_id = "camera";
      message.format = "jpeg";
      message.data.assign(reinterpret_cast<const std::byte *>(encoded.data()),
                          reinterpret_cast<const std::byte *>(encoded.data() + encoded.size()));
      channel->log(message, log_time);
    };
    log_image(image_raw, frame.snapshot.image);
    log_image(image, detail::buff_detector_debug_image(frame));
  }
};

BuffDetectorDebugVisualizer::BuffDetectorDebugVisualizer(const std::string & config_path)
{
  const auto config = detail::load_foxglove_config(tools::load(config_path));
  impl_ = std::make_unique<Impl>(config);
  if (!config.enable) return;
  foxglove::WebSocketServerOptions options;
  options.host = "0.0.0.0";
  options.port = 8765;
  options.name = "simple_auto_aim buff detector debug";
  auto server = foxglove::WebSocketServer::create(std::move(options));
  if (!server) {
    std::cerr << "Failed to start Foxglove server: " << foxglove::strerror(server.error()) << '\n';
    return;
  }
  impl_->server.emplace(std::move(server.value()));
  const auto create_channel = [](auto & destination, auto result) {
    if (result) destination.emplace(std::move(result.value()));
    else
      std::cerr << "Failed to create debug channel: " << foxglove::strerror(result.error()) << '\n';
  };
  create_channel(impl_->detector, detail::create_buff_detector_channel());
  create_channel(impl_->image_raw, foxglove::schemas::CompressedImageChannel::create("/image_raw"));
  create_channel(impl_->image, foxglove::schemas::CompressedImageChannel::create("/image"));
  impl_->worker = std::thread([this] {
    while (true) {
      std::optional<BuffDetectorDebugFrame> frame;
      {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->ready.wait(lock, [this] { return impl_->stopped || impl_->latest.has_value(); });
        if (!impl_->latest) return;
        frame = std::move(impl_->latest);
        impl_->latest.reset();
      }
      impl_->publish_frame(*frame);
    }
  });
}

BuffDetectorDebugVisualizer::~BuffDetectorDebugVisualizer()
{
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->stopped = true;
  }
  impl_->ready.notify_one();
  if (impl_->worker.joinable()) impl_->worker.join();
}

void BuffDetectorDebugVisualizer::publish(BuffDetectorDebugFrame frame)
{
  if (!impl_->server) return;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->stopped) return;
    impl_->latest = std::move(frame);
  }
  impl_->ready.notify_one();
}

int run_buff_detector_debug(int argc, char * argv[], BuffDetectorDebugTarget target)
{
  const std::string keys =
    "{help h usage ? | | 输出命令行参数说明}"
    "{@config-path   | | 与主链路相同的 yaml 配置文件路径}"
    "{mode           |1| 1=小符，2=大符}"
    "{target-color   | | 必填：red、blue 或 none（由下位机决定）}";
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }
  const auto config_path = cli.get<std::string>("@config-path");
  const int mode = cli.get<int>("mode");
  const auto color = cli.get<std::string>("target-color");
  if (!cli.check() || (mode != 1 && mode != 2)) {
    cli.printErrors();
    std::cerr << "mode 必须为 1（小符）或 2（大符）\n";
    return 2;
  }
  std::optional<io::InfantryEnemyColor> target_color_override;
  if (color == "red") target_color_override = io::InfantryEnemyColor::red;
  else if (color == "blue") target_color_override = io::InfantryEnemyColor::blue;
  else if (color != "none") {
    std::cerr << "target-color 为必填参数，必须为 red、blue 或 none\n";
    return 2;
  }

  Exiter exiter;
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);
  const auto config = auto_buff_v2::BuffConfig::load(config_path);
  FrameCapture frames(camera, gimbal, target_color_override);
  BuffDetectorDebug detector(config.detector, target);
  BuffDetectorDebugVisualizer visualizer(config_path);
  while (!exiter.exit()) {
    FrameFacts facts;
    if (!frames.next(facts)) break;
    visualizer.publish(detector.process(facts));
  }
  return 0;
}
}  // namespace tools
