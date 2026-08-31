#ifndef CALIBRATION__PATTERN_HPP
#define CALIBRATION__PATTERN_HPP

#include <opencv2/calib3d.hpp>

namespace calibration
{
inline bool find_chessboard_pattern(
  const cv::Mat & image, const cv::Size & pattern_size, std::vector<cv::Point2f> & corners)
{
  return cv::findChessboardCornersSB(image, pattern_size, corners, cv::CALIB_CB_NORMALIZE_IMAGE);
}
}  // namespace calibration

#endif  // CALIBRATION__PATTERN_HPP
