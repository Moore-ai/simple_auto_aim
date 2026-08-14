#include <cassert>

#include "tools/web_debug.hpp"

int main()
{
  tools::WebDebugContext context;
  context.frame_id = 17;
  context.elapsed_seconds = 1.25;
  context.latency_ms = 3.5;
  context.mode = io::GimbalMode::AUTO_AIM;
  context.gimbal.yaw = 0.1F;
  context.gimbal.pitch = -0.2F;
  context.plan.debug_valid = false;

  const auto log = tools::WebDebug::make_log_json(context);
  assert(log.at("frame_id") == 17);
  assert(log.at("mode") == "AUTO_AIM");
  assert(log.at("gimbal").at("yaw") == 0.1F);
  assert(log.at("mpc").at("debug_valid") == false);

  tools::WebDebug web({
    "/sp_vision_25_web_debug_test",
    "/tmp/sp_vision_25_data_test.json",
    "/tmp/sp_vision_25_log_test.json"
  });
  web.publish(context, cv::Mat::zeros(8, 8, CV_8UC3));
  const auto data = web.data_json();
  assert(data.at("time").size() == 1);
  assert(data.at("yaw").size() == 1);

  for (uint64_t frame_id = 0; frame_id < 100; ++frame_id) {
    context.frame_id = frame_id;
    context.elapsed_seconds = static_cast<double>(frame_id);
    web.publish(context, cv::Mat::zeros(8, 8, CV_8UC3));
  }
  const auto rolling_data = web.data_json();
  assert(rolling_data.at("time").size() == 100);
  assert(rolling_data.at("time").front() == 0.0);
  return 0;
}
