#include "tools/web_debug.hpp"

int main()
{
  tools::WebDebugContext context;
  context.gimbal.mode = 7;
  tools::WebDebug web;
  web.publish(context, cv::Mat::zeros(4, 4, CV_8UC3));
  return 0;
}
