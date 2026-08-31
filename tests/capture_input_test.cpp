#include <opencv2/highgui.hpp>

bool should_save_capture(int event);

int main()
{
  return should_save_capture(cv::EVENT_LBUTTONDOWN) &&
      !should_save_capture(cv::EVENT_MOUSEMOVE) && !should_save_capture(cv::EVENT_RBUTTONDOWN)
    ? 0
    : 1;
}
