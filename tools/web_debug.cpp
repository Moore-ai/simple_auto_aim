#include "web_debug.hpp"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tools/logger.hpp"

namespace tools
{
namespace
{

nlohmann::json point_json(const cv::Point2f & point) { return {point.x, point.y}; }

nlohmann::json points_json(const std::vector<cv::Point2f> & points)
{
  nlohmann::json result = nlohmann::json::array();
  for (const auto & point : points) result.push_back(point_json(point));
  return result;
}

nlohmann::json vector_json(const Eigen::VectorXd & vector)
{
  nlohmann::json result = nlohmann::json::array();
  for (Eigen::Index i = 0; i < vector.size(); ++i) result.push_back(vector[i]);
  return result;
}

const char * mode_string(io::GimbalMode mode)
{
  switch (mode) {
    case io::GimbalMode::IDLE:
      return "IDLE";
    case io::GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case io::GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case io::GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
  }
  return "UNKNOWN";
}

const char * color_string(auto_aim::Color color)
{
  switch (color) {
    case auto_aim::red:
      return "red";
    case auto_aim::blue:
      return "blue";
    case auto_aim::extinguish:
      return "extinguish";
    case auto_aim::purple:
      return "purple";
  }
  return "unknown";
}

const char * armor_type_string(auto_aim::ArmorType type)
{
  switch (type) {
    case auto_aim::big:
      return "big";
    case auto_aim::small:
      return "small";
  }
  return "unknown";
}

const char * armor_name_string(auto_aim::ArmorName name)
{
  switch (name) {
    case auto_aim::one:
      return "one";
    case auto_aim::two:
      return "two";
    case auto_aim::three:
      return "three";
    case auto_aim::four:
      return "four";
    case auto_aim::five:
      return "five";
    case auto_aim::sentry:
      return "sentry";
    case auto_aim::outpost:
      return "outpost";
    case auto_aim::base:
      return "base";
    case auto_aim::not_armor:
      return "not_armor";
  }
  return "unknown";
}

nlohmann::json armor_json(const auto_aim::Armor & armor)
{
  return {
    {"color", color_string(armor.color)},
    {"name", armor_name_string(armor.name)},
    {"type", armor_type_string(armor.type)},
    {"confidence", armor.confidence},
    {"box", {armor.box.x, armor.box.y, armor.box.width, armor.box.height}},
    {"points", points_json(armor.points)},
    {"xyz_in_gimbal", vector_json(armor.xyz_in_gimbal)},
    {"xyz_in_world", vector_json(armor.xyz_in_world)},
    {"ypr_in_gimbal", vector_json(armor.ypr_in_gimbal)},
    {"ypr_in_world", vector_json(armor.ypr_in_world)},
    {"yaw_raw", armor.yaw_raw},
  };
}

nlohmann::json lightbar_json(const auto_aim::Lightbar & lightbar)
{
  return {
    {"color", color_string(lightbar.color)},
    {"center", point_json(lightbar.center)},
    {"top", point_json(lightbar.top)},
    {"bottom", point_json(lightbar.bottom)},
    {"length", lightbar.length},
    {"angle", lightbar.angle},
  };
}

nlohmann::json gimbal_json(const io::GimbalState & gimbal)
{
  return {
    {"yaw", gimbal.yaw},
    {"yaw_vel", gimbal.yaw_vel},
    {"pitch", gimbal.pitch},
    {"pitch_vel", gimbal.pitch_vel},
    {"bullet_speed", gimbal.bullet_speed},
    {"bullet_count", gimbal.bullet_count},
  };
}

nlohmann::json plan_json(const auto_aim::Plan & plan)
{
  return {
    {"debug_xyza", vector_json(plan.debug_xyza)},
    {"debug_valid", plan.debug_valid},
    {"fly_time", plan.fly_time},
    {"target_yaw", plan.target_yaw},
    {"target_pitch", plan.target_pitch},
    {"yaw", plan.yaw},
    {"yaw_vel", plan.yaw_vel},
    {"yaw_acc", plan.yaw_acc},
    {"pitch", plan.pitch},
    {"pitch_vel", plan.pitch_vel},
    {"pitch_acc", plan.pitch_acc},
    {"fire_error", {{"yaw", plan.target_yaw - plan.yaw}, {"pitch", plan.target_pitch - plan.pitch}}},
  };
}

void draw_polygon(cv::Mat & image, const std::vector<cv::Point2f> & points, const cv::Scalar & color)
{
  if (points.empty()) return;
  std::vector<cv::Point> pixels;
  pixels.reserve(points.size());
  for (const auto & point : points) pixels.emplace_back(cvRound(point.x), cvRound(point.y));
  cv::polylines(image, pixels, true, color, 2, cv::LINE_AA);
}

void draw_arrow(
  cv::Mat & image, const std::optional<std::pair<cv::Point2f, cv::Point2f>> & arrow,
  const cv::Scalar & color)
{
  if (!arrow) return;
  cv::arrowedLine(image, arrow->first, arrow->second, color, 2, cv::LINE_AA, 0, 0.2);
}

}  // namespace

WebDebug::WebDebug(WebDebugPaths paths)
: paths_(std::move(paths)),
  data_({
    {"time", nlohmann::json::array()},
    {"yaw", nlohmann::json::array()},
    {"pitch", nlohmann::json::array()},
    {"target_yaw", nlohmann::json::array()},
    {"target_pitch", nlohmann::json::array()},
    {"gimbal_yaw", nlohmann::json::array()},
    {"gimbal_pitch", nlohmann::json::array()},
    {"control_v_yaw", nlohmann::json::array()},
    {"control_v_pitch", nlohmann::json::array()},
    {"control_a_yaw", nlohmann::json::array()},
    {"control_a_pitch", nlohmann::json::array()},
    {"fly_time", nlohmann::json::array()},
    {"target_v_yaw", nlohmann::json::array()},
    {"yaw_diff", nlohmann::json::array()},
    {"pitch_diff", nlohmann::json::array()},
  })
{
}

WebDebug::~WebDebug()
{
  if (shm_data_) ::munmap(shm_data_, kSharedMemorySize);
  if (shm_fd_ >= 0) ::close(shm_fd_);
  if (shm_fd_ >= 0) ::shm_unlink(paths_.shm_name.c_str());
}

nlohmann::json WebDebug::make_log_json(const WebDebugContext & context)
{
  nlohmann::json armors = nlohmann::json::array();
  for (const auto & armor : context.armors) armors.push_back(armor_json(armor));
  nlohmann::json lightbars = nlohmann::json::array();
  for (const auto & lightbar : context.lightbars) lightbars.push_back(lightbar_json(lightbar));

  return {
    {"frame_id", context.frame_id},
    {"time", context.elapsed_seconds},
    {"latency_ms", context.latency_ms},
    {"mode", mode_string(context.mode)},
    {"gimbal", gimbal_json(context.gimbal)},
    {"detector", {
      {"count", context.armors.size()},
      {"lightbar_count", context.lightbars.size()},
      {"armors", std::move(armors)},
      {"lightbars", std::move(lightbars)}}},
    {"tracker", context.tracker_json},
    {"mpc", plan_json(context.plan)},
    {"command", {{"control", context.plan.control}, {"fire", context.plan.fire}, {"yaw", context.plan.yaw}, {"pitch", context.plan.pitch}}},
  };
}

nlohmann::json WebDebug::data_json() const { return data_; }

void WebDebug::publish(const WebDebugContext & context, cv::Mat & image)
{
  try {
    draw(image, context);
    append_data(context);
    write_json_atomically(paths_.data_path, data_);
    write_json_atomically(paths_.log_path, make_log_json(context));
    write_frame(image);
  } catch (const std::exception & error) {
    logger()->warn("[WebDebug] publish failed: {}", error.what());
  }
}

void WebDebug::draw(cv::Mat & image, const WebDebugContext & context) const
{
  if (image.empty()) return;

  for (const auto & armor : context.armors) {
    const cv::Scalar color = armor.color == auto_aim::red ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
    draw_polygon(image, armor.points, color);
    cv::putText(
      image, std::string(armor_name_string(armor.name)) + cv::format(" %.2f", armor.confidence),
      armor.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
  }

  for (const auto & lightbar : context.lightbars) {
    const cv::Scalar color = lightbar.color == auto_aim::red ? cv::Scalar(0, 0, 255) :
      cv::Scalar(255, 0, 0);
    cv::line(image, lightbar.top, lightbar.bottom, color, 2, cv::LINE_AA);
  }

  for (const auto & polygon : context.projected_target_armors) draw_polygon(image, polygon, {0, 255, 0});
  draw_polygon(image, context.projected_aim_armor, {0, 255, 255});
  draw_arrow(image, context.target_velocity_arrow, {255, 255, 0});
  draw_arrow(image, context.target_yaw_rate_arrow, {255, 0, 255});

  const cv::Point center(image.cols / 2, image.rows / 2);
  cv::drawMarker(image, center, {255, 255, 255}, cv::MARKER_CROSS, 18, 1, cv::LINE_AA);
  cv::putText(
    image, cv::format("t %.3f s  latency %.2f ms", context.elapsed_seconds, context.latency_ms), {10, 22},
    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
  cv::putText(
    image,
    cv::format(
      "MPC target %.3f %.3f cmd %.3f %.3f v %.3f %.3f a %.3f %.3f control %d",
      context.plan.target_yaw, context.plan.target_pitch, context.plan.yaw, context.plan.pitch,
      context.plan.yaw_vel, context.plan.pitch_vel, context.plan.yaw_acc, context.plan.pitch_acc,
      context.plan.control),
    {10, 44}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1, cv::LINE_AA);
  cv::putText(
    image, cv::format("V_norm %.3f  V_yaw %.3f", context.target_velocity_norm, context.target_yaw_rate),
    {10, 62}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {50, 255, 50}, 1, cv::LINE_AA);
  if (context.plan.fire) {
    cv::putText(image, "Fire!", {10, 72}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2, cv::LINE_AA);
  }
}

void WebDebug::append_data(const WebDebugContext & context)
{
  const double target_v_yaw = context.target_yaw_rate;
  const std::initializer_list<std::pair<const char *, double>> values = {
    {"time", context.elapsed_seconds},
    {"yaw", context.plan.yaw},
    {"pitch", context.plan.pitch},
    {"target_yaw", context.plan.target_yaw},
    {"target_pitch", context.plan.target_pitch},
    {"gimbal_yaw", context.gimbal.yaw},
    {"gimbal_pitch", context.gimbal.pitch},
    {"control_v_yaw", context.plan.yaw_vel},
    {"control_v_pitch", context.plan.pitch_vel},
    {"control_a_yaw", context.plan.yaw_acc},
    {"control_a_pitch", context.plan.pitch_acc},
    {"fly_time", context.plan.debug_valid ? context.plan.fly_time : 0.0},
    {"target_v_yaw", target_v_yaw},
    {"yaw_diff", context.plan.yaw - context.gimbal.yaw},
    {"pitch_diff", context.plan.pitch - context.gimbal.pitch},
  };
  for (const auto & [name, value] : values) {
    auto & series = data_.at(name);
    series.push_back(value);
    if (series.size() > kMaxPoints) series.erase(series.begin());
  }
}

void WebDebug::write_json_atomically(const std::string & path, const nlohmann::json & json) const
{
  const std::filesystem::path destination(path);
  const std::filesystem::path temporary = destination.string() + ".tmp";
  {
    std::ofstream output(temporary);
    if (!output) throw std::runtime_error("cannot open " + temporary.string());
    output << json.dump();
    if (!output) throw std::runtime_error("cannot write " + temporary.string());
  }
  std::error_code error;
  std::filesystem::rename(temporary, destination, error);
  if (error) throw std::system_error(error, "cannot rename " + temporary.string());
}

void WebDebug::write_frame(const cv::Mat & image)
{
  std::vector<uchar> jpeg;
  if (!cv::imencode(".jpg", image, jpeg, {cv::IMWRITE_JPEG_QUALITY, 75})) {
    throw std::runtime_error("JPEG encoding failed");
  }
  if (jpeg.size() > kSharedMemorySize - sizeof(uint32_t)) {
    throw std::runtime_error("JPEG frame exceeds shared-memory slot");
  }

  if (shm_fd_ < 0) {
    shm_fd_ = ::shm_open(paths_.shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) throw std::system_error(errno, std::generic_category(), "shm_open failed");
    if (::ftruncate(shm_fd_, kSharedMemorySize) != 0) {
      const int error = errno;
      ::close(shm_fd_);
      shm_fd_ = -1;
      throw std::system_error(error, std::generic_category(), "ftruncate failed");
    }
    void * mapping = ::mmap(nullptr, kSharedMemorySize, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (mapping == MAP_FAILED) {
      const int error = errno;
      ::close(shm_fd_);
      shm_fd_ = -1;
      shm_data_ = nullptr;
      throw std::system_error(error, std::generic_category(), "mmap failed");
    }
    shm_data_ = static_cast<uint8_t *>(mapping);
  }

  const uint32_t size = static_cast<uint32_t>(jpeg.size());
  std::memcpy(shm_data_, &size, sizeof(size));
  std::memcpy(shm_data_ + sizeof(size), jpeg.data(), jpeg.size());
}

}  // namespace tools
