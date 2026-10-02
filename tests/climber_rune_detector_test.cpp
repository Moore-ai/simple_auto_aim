#include <cassert>
#include <cmath>
#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/pass/serialize.hpp>

#include "tasks/auto_buff_v2/detectors/rune_detector_factory.hpp"

int main()
{
  // A real inference fixture: red, duplicate red, second red, and blue candidates.
  std::vector<float> data(33 * 4, 0);
  auto put = [&](int row, int col, float value) { data[row * 4 + col] = value; };
  const std::array<cv::Point2f, 9> points = {
    cv::Point2f(100, 100), {195, 220}, {220, 205}, {220, 195}, {205, 180},
    {195, 180}, {180, 195}, {180, 205}, {205, 220}};
  for (int col = 0; col < 4; ++col) {
    const float dx = col == 1 ? 2 : col == 2 ? 100 : col == 3 ? 200 : 0;
    put(0, col, 200 + dx);
    put(1, col, 200);
    put(2, col, 40);
    put(3, col, 40);
    put(col == 3 ? 5 : 4, col, 0.9f - col * 0.1f);
    for (int p = 0; p < 9; ++p) {
      put(6 + p * 3, col, points[p].x + (p == 0 ? 0 : dx));
      put(7 + p * 3, col, points[p].y);
      put(8 + p * 3, col, 1);
    }
  }
  namespace op = ov::opset13;
  auto input = std::make_shared<op::Parameter>(ov::element::f32, ov::Shape{1, 3, 640, 640});
  auto output = op::Constant::create(ov::element::f32, ov::Shape{1, 33, 4}, data);
  auto sum = std::make_shared<op::ReduceSum>(input,
    op::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3}), false);
  auto zero = std::make_shared<op::Multiply>(sum,
    op::Constant::create(ov::element::f32, ov::Shape{}, {0}));
  auto result = std::make_shared<op::Add>(output, zero);
  const auto dir = std::filesystem::temp_directory_path() / "climber_rune_detector_test";
  std::filesystem::create_directories(dir);
  ov::serialize(std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input}),
                (dir / "model.xml").string(), (dir / "model.bin").string());

  auto_buff_v2::BuffConfig::Detector config;
  config.type = "climber";
  config.parameters["climber"]["model"] = (dir / "model.xml").string();
  config.parameters["climber"]["IouThreshold"] = 1.0;
  auto detector = auto_buff_v2::make_rune_detector(config);
  cv::Mat image = cv::Mat::zeros(320, 320, CV_8UC3);
  const auto original = image.clone();
  detector->set_enemy_red(true);
  auto red = detector->detect(image);
  assert(red.bullseyes.size() == 2);  // Deduplicate even when NMS retains overlapping boxes.
  assert(red.icons.size() == 1);
  assert(cv::norm(red.icons[0].center - cv::Point2f(50, 50)) < 0.01);
  assert(cv::norm(red.bullseyes[0].center - cv::Point2f(100, 100)) < 0.01);
  assert(cv::norm(red.bullseyes[1].center - cv::Point2f(150, 100)) < 0.01);
  assert(cv::norm(red.bullseyes[0].corners[0] - cv::Point2f(100, 90)) < 0.01);
  assert(cv::norm(red.bullseyes[0].corners[1] - cv::Point2f(90, 100)) < 0.01);
  assert(!red.bullseyes[0].active);
  assert(std::abs(red.bullseyes[0].score - 0.9) < 1e-5);
  assert(cv::norm(image, original, cv::NORM_INF) == 0);
  detector->set_enemy_red(false);
  auto blue = detector->detect(image);
  assert(blue.bullseyes.size() == 1);
  assert(cv::norm(blue.bullseyes[0].center - cv::Point2f(200, 100)) < 0.01);
  assert(detector->detect(cv::Mat{}).bullseyes.empty());

  config.parameters["climber"]["max_bullseyes"] = 1;
  auto single = auto_buff_v2::make_rune_detector(config);
  single->set_enemy_red(true);
  const auto single_result = single->detect(image);
  assert(single_result.bullseyes.size() == 1);
  assert(cv::norm(single_result.bullseyes[0].center - cv::Point2f(100, 100)) < 0.01);
  config.parameters["climber"]["max_bullseyes"] = 2;

  // Preprocessing dimensions follow the model, including rectangular inputs.
  auto rectangular_model = std::make_shared<ov::Model>(
    ov::OutputVector{result}, ov::ParameterVector{input});
  rectangular_model->reshape(ov::PartialShape{1, 3, 320, 640});
  ov::serialize(rectangular_model, (dir / "rectangular.xml").string(),
                (dir / "rectangular.bin").string());
  auto_buff_v2::BuffConfig::Detector rectangular_config;
  rectangular_config.type = "climber";
  rectangular_config.parameters = YAML::Clone(config.parameters);
  rectangular_config.parameters["climber"]["model"] = (dir / "rectangular.xml").string();
  auto rectangular = auto_buff_v2::make_rune_detector(rectangular_config);
  rectangular->set_enemy_red(true);
  const auto scaled = rectangular->detect(cv::Mat::zeros(160, 320, CV_8UC3));
  assert(scaled.bullseyes.size() == 2);
  assert(cv::norm(scaled.bullseyes[0].center - cv::Point2f(100, 100)) < 0.01);
  assert(cv::norm(scaled.icons[0].center - cv::Point2f(50, 50)) < 0.01);

  config.parameters["climber"]["device"] = "INVALID_DEVICE";
  bool invalid_device_rejected = false;
  try {
    auto_buff_v2::make_rune_detector(config);
  } catch (const ov::Exception &) {
    invalid_device_rejected = true;
  }
  assert(invalid_device_rejected);
  config.parameters["climber"]["device"] = "CPU";

  // Contour correction is bounded relative to the network's R center.
  assert(config.parameters["climber"]["model"].as<std::string>() ==
         (dir / "model.xml").string());
  detector->set_enemy_red(true);
  cv::rectangle(image, {52, 48}, {54, 52}, {255, 255, 255}, cv::FILLED);
  config.parameters["climber"]["r_center_refine_max_shift"] = 1.0;
  auto refined = auto_buff_v2::make_rune_detector(config);
  refined->set_enemy_red(true);
  const auto corrected = refined->detect(image);
  assert(cv::norm(corrected.icons[0].center - cv::Point2f(51, 50)) < 0.01);

  config.parameters["climber"]["ConfidenceThreshold"] = 0.95;
  auto strict = auto_buff_v2::make_rune_detector(config);
  strict->set_enemy_red(true);
  const auto below_threshold = strict->detect(image);
  assert(below_threshold.bullseyes.empty() && below_threshold.icons.empty());

  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
  const auto real_config =
    auto_buff_v2::BuffConfig::load((root / "configs/standard.yaml").string());
  auto real_detector_config = real_config.detector;
  real_detector_config.type = "climber";
  real_detector_config.parameters["climber"]["model"] = (root / "assets/buff_repvgg.xml").string();
  auto real_detector = auto_buff_v2::make_rune_detector(real_detector_config);
  const auto blank = real_detector->detect(original);
  assert(blank.bullseyes.empty() && blank.icons.empty());
  std::filesystem::remove_all(dir);
}
