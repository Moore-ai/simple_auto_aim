#include "hikrobot.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace io
{
HikRobotConfig load_hikrobot_config(const YAML::Node & yaml)
{
  HikRobotConfig config;
  config.exposure_us = tools::read<double>(yaml, "exposure_ms") * 1e3;
  config.gain = tools::read<double>(yaml, "gain");
  if (yaml["camera_index"]) config.device_index = yaml["camera_index"].as<int>();
  if (yaml["image_width"]) config.image_width = yaml["image_width"].as<int>();
  if (yaml["image_height"]) config.image_height = yaml["image_height"].as<int>();
  if (yaml["fps"]) config.fps = yaml["fps"].as<double>();
  if (yaml["flip_image"]) config.flip_image = yaml["flip_image"].as<bool>();
  return config;
}

HikRobot::HikRobot(HikRobotConfig config) : config_(config), exposure_us_(config.exposure_us)
{
  if (!open()) throw std::runtime_error("Failed to open HikRobot camera");
}

HikRobot::~HikRobot()
{
  close();
  tools::logger()->info("HikRobot destructed.");
}

bool HikRobot::read(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  if (!handle_ || !grabbing_) return false;
  if (timeout.count() <= 0 || timeout.count() > std::numeric_limits<unsigned int>::max()) return false;

  MV_FRAME_OUT_INFO_EX frame_info{};
  const auto status = MV_CC_GetOneFrameTimeout(
    handle_, frame_buffer_.data(), static_cast<unsigned int>(frame_buffer_.size()), &frame_info,
    static_cast<unsigned int>(timeout.count()));
  if (status != MV_OK) {
    tools::logger()->warn("MV_CC_GetOneFrameTimeout failed: {:#x}", status);
    return false;
  }

  if (frame_info.nWidth == 0 || frame_info.nHeight == 0 || frame_info.nFrameLen == 0 ||
      frame_info.nFrameLen > frame_buffer_.size()) {
    tools::logger()->warn(
      "Invalid Hik frame: width={}, height={}, length={}", frame_info.nWidth, frame_info.nHeight,
      frame_info.nFrameLen);
    return false;
  }

  const auto bgr_size = static_cast<size_t>(frame_info.nWidth) * frame_info.nHeight * 3;
  if (bgr_size > std::numeric_limits<unsigned int>::max()) {
    tools::logger()->warn("Hik frame is too large to convert: {}x{}", frame_info.nWidth, frame_info.nHeight);
    return false;
  }
  bgr_buffer_.resize(bgr_size);

  MV_CC_PIXEL_CONVERT_PARAM convert{};
  convert.nWidth = frame_info.nWidth;
  convert.nHeight = frame_info.nHeight;
  convert.enSrcPixelType = frame_info.enPixelType;
  convert.pSrcData = frame_buffer_.data();
  convert.nSrcDataLen = frame_info.nFrameLen;
  convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
  convert.pDstBuffer = bgr_buffer_.data();
  convert.nDstBufferSize = static_cast<unsigned int>(bgr_buffer_.size());

  const auto convert_status = MV_CC_ConvertPixelType(handle_, &convert);
  if (convert_status != MV_OK || convert.nDstLen != bgr_buffer_.size()) {
    tools::logger()->warn(
      "MV_CC_ConvertPixelType failed: status={:#x}, output={}/{}", convert_status, convert.nDstLen,
      bgr_buffer_.size());
    return false;
  }

  const cv::Mat bgr(frame_info.nHeight, frame_info.nWidth, CV_8UC3, bgr_buffer_.data());
  if (config_.flip_image)
    cv::flip(bgr, img, -1);
  else
    img = bgr.clone();
  if (img.empty()) return false;

  timestamp = std::chrono::steady_clock::now();
  return true;
}

double HikRobot::get_exposure_us() const { return exposure_us_.load(); }

void HikRobot::set_exposure_us(double exposure_us)
{
  if (!std::isfinite(exposure_us) || exposure_us <= 0.0) return;
  if (set_float_value("ExposureTime", exposure_us)) exposure_us_.store(exposure_us);
}

bool HikRobot::open()
{
  if (config_.device_index < 0 || config_.image_width <= 0 || config_.image_height <= 0 ||
      !std::isfinite(config_.fps) || config_.fps <= 0.0) {
    tools::logger()->error("Invalid HikRobot configuration");
    return false;
  }

  MV_CC_DEVICE_INFO_LIST device_list{};
  auto status = MV_CC_EnumDevices(MV_USB_DEVICE, &device_list);
  if (status != MV_OK) {
    tools::logger()->error("MV_CC_EnumDevices failed: {:#x}", status);
    return false;
  }
  if (static_cast<unsigned int>(config_.device_index) >= device_list.nDeviceNum) {
    tools::logger()->error(
      "Hik camera index {} is unavailable ({} USB device(s) found)", config_.device_index,
      device_list.nDeviceNum);
    return false;
  }

  status = MV_CC_CreateHandle(&handle_, device_list.pDeviceInfo[config_.device_index]);
  if (status != MV_OK) {
    tools::logger()->error("MV_CC_CreateHandle failed: {:#x}", status);
    handle_ = nullptr;
    return false;
  }
  status = MV_CC_OpenDevice(handle_);
  if (status != MV_OK) {
    tools::logger()->error("MV_CC_OpenDevice failed: {:#x}", status);
    close();
    return false;
  }
  if (!configure()) {
    close();
    return false;
  }

  status = MV_CC_StartGrabbing(handle_);
  if (status != MV_OK) {
    tools::logger()->error("MV_CC_StartGrabbing failed: {:#x}", status);
    close();
    return false;
  }
  grabbing_ = true;
  if (!prepare_frame_buffer()) {
    close();
    return false;
  }

  tools::logger()->info(
    "HikRobot opened: index={}, {}x{}, {:.1f} fps, BGR8", config_.device_index, config_.image_width,
    config_.image_height, config_.fps);
  return true;
}

void HikRobot::close()
{
  if (!handle_) return;
  if (grabbing_) {
    const auto status = MV_CC_StopGrabbing(handle_);
    if (status != MV_OK) tools::logger()->warn("MV_CC_StopGrabbing failed: {:#x}", status);
    grabbing_ = false;
  }
  const auto close_status = MV_CC_CloseDevice(handle_);
  if (close_status != MV_OK) tools::logger()->warn("MV_CC_CloseDevice failed: {:#x}", close_status);
  const auto destroy_status = MV_CC_DestroyHandle(handle_);
  if (destroy_status != MV_OK)
    tools::logger()->warn("MV_CC_DestroyHandle failed: {:#x}", destroy_status);
  handle_ = nullptr;
  frame_buffer_.clear();
  bgr_buffer_.clear();
}

bool HikRobot::configure()
{
  const auto frame_rate_enable_status = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
  if (frame_rate_enable_status != MV_OK) {
    tools::logger()->error(
      "MV_CC_SetBoolValue(AcquisitionFrameRateEnable) failed: {:#x}", frame_rate_enable_status);
    return false;
  }
  return set_enum_value("TriggerMode", MV_TRIGGER_MODE_OFF) &&
    set_enum_value("AcquisitionMode", "Continuous") &&
    set_enum_value("PixelFormat", PixelType_Gvsp_BayerRG8) &&
    set_enum_value("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF) &&
    set_enum_value("GainAuto", MV_GAIN_MODE_OFF) &&
    set_float_value("ExposureTime", exposure_us_.load()) && set_float_value("Gain", config_.gain) &&
    set_float_value("AcquisitionFrameRate", config_.fps) &&
    set_int_value("OffsetX", 0) && set_int_value("OffsetY", 0) &&
    set_int_value("Width", config_.image_width) && set_int_value("Height", config_.image_height);
}

bool HikRobot::prepare_frame_buffer()
{
  MVCC_INTVALUE payload_size{};
  const auto status = MV_CC_GetIntValue(handle_, "PayloadSize", &payload_size);
  if (status != MV_OK || payload_size.nCurValue == 0) {
    tools::logger()->error("MV_CC_GetIntValue(PayloadSize) failed: {:#x}", status);
    return false;
  }
  frame_buffer_.resize(payload_size.nCurValue);
  return true;
}

bool HikRobot::set_float_value(const std::string & name, double value)
{
  const auto status = MV_CC_SetFloatValue(handle_, name.c_str(), value);
  if (status != MV_OK) tools::logger()->error("MV_CC_SetFloatValue({}) failed: {:#x}", name, status);
  return status == MV_OK;
}

bool HikRobot::set_int_value(const std::string & name, unsigned int value)
{
  const auto status = MV_CC_SetIntValue(handle_, name.c_str(), value);
  if (status != MV_OK) tools::logger()->error("MV_CC_SetIntValue({}) failed: {:#x}", name, status);
  return status == MV_OK;
}

bool HikRobot::set_enum_value(const std::string & name, unsigned int value)
{
  const auto status = MV_CC_SetEnumValue(handle_, name.c_str(), value);
  if (status != MV_OK) tools::logger()->error("MV_CC_SetEnumValue({}) failed: {:#x}", name, status);
  return status == MV_OK;
}

bool HikRobot::set_enum_value(const std::string & name, const std::string & value)
{
  const auto status = MV_CC_SetEnumValueByString(handle_, name.c_str(), value.c_str());
  if (status != MV_OK)
    tools::logger()->error("MV_CC_SetEnumValueByString({}) failed: {:#x}", name, status);
  return status == MV_OK;
}

}  // namespace io
