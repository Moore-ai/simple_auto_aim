#include <opencv2/opencv.hpp>

#include "calibration/pattern.hpp"

int main()
{
  constexpr int square_size = 60;
  cv::Mat image(9 * square_size, 12 * square_size, CV_8UC1, cv::Scalar::all(255));
  for (int row = 0; row < 9; ++row)
    for (int col = 0; col < 12; ++col)
      if ((row + col) % 2 == 0)
        cv::rectangle(
          image, {col * square_size, row * square_size},
          {(col + 1) * square_size - 1, (row + 1) * square_size - 1}, cv::Scalar::all(0), cv::FILLED);

  std::vector<cv::Point2f> corners;
  return calibration::find_chessboard_pattern(image, {11, 8}, corners) && corners.size() == 88 ? 0 : 1;
}
