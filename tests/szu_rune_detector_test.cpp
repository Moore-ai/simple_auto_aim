#include <cassert>
#include <cmath>
#include <filesystem>
#include <limits>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/pass/serialize.hpp>

#include "tasks/auto_buff_v2/detectors/rune_detector_factory.hpp"

// 验证五点网络解析、预处理、颜色轮廓精修、配置校验及附带模型的实际检测结果。
int main()
{
  // 构造候选以验证输出通道偏移、补边坐标还原、非极大值抑制和状态映射。
  // 原图 320×160 转为模型输入 640×480：缩放两倍，上下各补边 80 像素。
  constexpr int count = 6;
  std::vector<float> data(18 * count, 0);
  const std::array<cv::Point2f, 5> points = {
    cv::Point2f(100, 60), {80, 80}, {40, 80}, {120, 80}, {100, 100}};
  for (int i = 0; i < count; ++i) {
    const float dx = i == 1 ? 2 : i >= 2 ? (i - 1) * 60 : 0;
    data[(i == 2 ? 1 : i == 3 ? 2 : 0) * count + i] = 0.95f - i * 0.02f;
    for (int p = 0; p < 5; ++p) {
      data[(3 + p * 3) * count + i] = 2 * (points[p].x + (p == 2 ? 0 : dx));
      data[(4 + p * 3) * count + i] = 2 * points[p].y + 80;
      data[(5 + p * 3) * count + i] = 0.9;
    }
  }
  data[5 * count + 4] = 0.1;  // 类别置信度高也不能保留关键点不可靠的候选。
  data[3 * count + 5] = std::numeric_limits<float>::quiet_NaN();
  data[1] = 0.99;  // 重复候选的类别置信度更高，但综合质量更低。
  for (int p = 0; p < 5; ++p) data[(5 + p * 3) * count + 1] = 0.8;
  namespace op = ov::opset13;
  auto input = std::make_shared<op::Parameter>(ov::element::f32, ov::Shape{1, 3, 480, 640});
  auto sum = std::make_shared<op::ReduceSum>(input,
    op::Constant::create(ov::element::i64, ov::Shape{4}, {0, 1, 2, 3}), false);
  auto zero = std::make_shared<op::Multiply>(sum,
    op::Constant::create(ov::element::f32, ov::Shape{}, {0}));
  const auto dir = std::filesystem::temp_directory_path() / "szu_rune_detector_test";
  std::filesystem::create_directories(dir);
  // 保存指定输出布局的测试模型，可让类别置信度依赖归一化后的颜色通道。
  auto save = [&](const std::string & name, bool transpose, bool check_rgb = false,
                  int channel = 0) {
    std::vector<float> values = data;
    if (transpose)
      for (int c = 0; c < 18; ++c)
        for (int i = 0; i < count; ++i) values[i * 18 + c] = data[c * count + i];
    auto output = op::Constant::create(ov::element::f32,
      transpose ? ov::Shape{1, count, 18} : ov::Shape{1, 18, count}, values);
    ov::Output<ov::Node> decoded = output;
    if (check_rgb) {
      // 类别置信度依赖归一化后的指定通道，用于检验颜色顺序、归一化和补边。
      std::vector<float> classes(18 * count, 0), coordinates = values;
      for (int i = 0; i < 3 * count; ++i) {
        classes[i] = coordinates[i];
        coordinates[i] = 0;
      }
      auto red = std::make_shared<op::Gather>(input,
        op::Constant::create(ov::element::i64, ov::Shape{}, {channel}),
        op::Constant::create(ov::element::i64, ov::Shape{}, {1}));
      auto maximum = std::make_shared<op::ReduceMax>(red,
        op::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2}), false);
      auto scores = std::make_shared<op::Multiply>(maximum,
        op::Constant::create(ov::element::f32, ov::Shape{1, 18, count}, classes));
      decoded = std::make_shared<op::Add>(scores,
        op::Constant::create(ov::element::f32, ov::Shape{1, 18, count}, coordinates));
    }
    auto result = std::make_shared<op::Add>(decoded, zero);
    ov::serialize(std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{input}),
                  (dir / (name + ".xml")).string(), (dir / (name + ".bin")).string());
  };
  save("nca", false);
  save("nac", true);
  save("rgb", false, true);
  save("padding", false, true, 2);

  cv::Mat image = cv::Mat::zeros(160, 320, CV_8UC3);
  for (int x : {102, 162, 222, 282})
    cv::ellipse(image, {x, 80}, {20, 20}, 0, 0, 360, {0, 0, 255}, cv::FILLED);
  cv::ellipse(image, {42, 80}, {7, 7}, 0, 0, 360, {0, 0, 255}, cv::FILLED);
  const auto original = image.clone();
  auto_buff_v2::BuffConfig::Detector config;
  config.type = "szu";
  for (const auto & layout : {"nca", "nac"}) {
    config.parameters["szu"]["model"] = (dir / (std::string(layout) + ".xml")).string();
    auto detector = auto_buff_v2::make_rune_detector(config);
    detector->set_enemy_red(true);
    const auto elements = detector->detect(image);
    assert(elements.bullseyes.size() == 3);
    assert(elements.icons.size() == 1);
    assert(cv::norm(elements.icons[0].center - cv::Point2f(42, 80)) < 1);
    for (int i = 0; i < 3; ++i) {
      assert(cv::norm(elements.bullseyes[i].center - cv::Point2f(102 + i * 60, 80)) < 1);
      assert(elements.bullseyes[i].active == (i > 0));
    }
    assert(cv::norm(elements.bullseyes[0].corners[0] - cv::Point2f(102, 60)) < 2);
    assert(cv::norm(elements.bullseyes[0].corners[1] - cv::Point2f(82, 80)) < 2);
    assert(cv::norm(elements.bullseyes[0].corners[2] - cv::Point2f(102, 100)) < 2);
    assert(cv::norm(elements.bullseyes[0].corners[3] - cv::Point2f(122, 80)) < 2);
    assert(std::abs(elements.bullseyes[0].score - 0.95) < 1e-5);
    assert(cv::norm(image, original, cv::NORM_INF) == 0);
    detector->set_enemy_red(false);
    assert(detector->detect(image).bullseyes.empty());
    cv::Mat blue;
    cv::cvtColor(image, blue, cv::COLOR_BGR2RGB);
    assert(detector->detect(blue).bullseyes.size() == 3);
    detector->set_enemy_red(true);
    auto no_r = image.clone();
    cv::rectangle(no_r, {20, 50}, {60, 110}, {0, 0, 0}, cv::FILLED);
    assert(detector->detect(no_r).bullseyes.empty());
    cv::Mat tilted = cv::Mat::zeros(image.size(), CV_8UC3);
    cv::ellipse(tilted, {102, 80}, {23, 15}, 30, 0, 360, {0, 0, 255}, cv::FILLED);
    cv::ellipse(tilted, {42, 80}, {7, 7}, 0, 0, 360, {0, 0, 255}, cv::FILLED);
    const auto elliptical = detector->detect(tilted);
    assert(elliptical.bullseyes.size() == 1);
    assert(cv::norm(elliptical.bullseyes[0].corners[0] - cv::Point2f(102, 63.8)) < 2);
    assert(cv::norm(elliptical.bullseyes[0].corners[3] - cv::Point2f(121.9, 80)) < 2);
    assert(detector->detect(cv::Mat{}).icons.empty());
    assert(detector->detect(cv::Mat::zeros(image.size(), CV_8UC3)).bullseyes.empty());
  }
  // 增大边缘裕量后，原先通过验证的轮廓应被过滤。
  config.parameters["szu"]["model"] = (dir / "nca.xml").string();
  config.parameters["szu"]["armor_border_margin"] = 80;
  auto border = auto_buff_v2::make_rune_detector(config);
  border->set_enemy_red(true);
  assert(border->detect(image).bullseyes.empty());
  config.parameters["szu"].remove("armor_border_margin");

  // 强模糊会抹去较小的 R 标，因此不应输出精修结果。
  config.parameters["szu"]["gaussian_kernel_size"] = 51;
  config.parameters["szu"]["gaussian_sigma"] = 100;
  auto blurred = auto_buff_v2::make_rune_detector(config);
  blurred->set_enemy_red(true);
  assert(blurred->detect(image).bullseyes.empty());
  config.parameters["szu"]["gaussian_sigma"] = 1;
  auto narrow_blur = auto_buff_v2::make_rune_detector(config);
  narrow_blur->set_enemy_red(true);
  assert(narrow_blur->detect(image).bullseyes.size() == 3);
  config.parameters["szu"]["gaussian_kernel_size"] = 1;
  config.parameters["szu"]["gaussian_sigma"] = 100;
  auto no_blur = auto_buff_v2::make_rune_detector(config);
  no_blur->set_enemy_red(true);
  assert(no_blur->detect(image).bullseyes.size() == 3);
  config.parameters["szu"].remove("gaussian_kernel_size");
  config.parameters["szu"].remove("gaussian_sigma");

  // 此输出依赖蓝色通道强度；红色测试图中只有灰色补边提供蓝色分量。
  config.parameters["szu"]["model"] = (dir / "padding.xml").string();
  config.parameters["szu"]["confidence_threshold"] = 0.3;
  auto padded = auto_buff_v2::make_rune_detector(config);
  padded->set_enemy_red(true);
  assert(padded->detect(image).bullseyes.size() == 3);
  config.parameters["szu"]["letterbox_value"] = 0;
  auto black_padding = auto_buff_v2::make_rune_detector(config);
  black_padding->set_enemy_red(true);
  assert(black_padding->detect(image).bullseyes.empty());
  config.parameters["szu"].remove("letterbox_value");
  config.parameters["szu"].remove("confidence_threshold");

  for (const auto & invalid : {"letterbox_value: 256", "letterbox_value: -1",
                             "gaussian_kernel_size: 0", "gaussian_kernel_size: 4",
                             "gaussian_sigma: -1", "gaussian_sigma: .nan",
                             "armor_border_margin: -1"}) {
    auto_buff_v2::BuffConfig::Detector parameters;
    parameters.type = config.type;
    parameters.parameters = YAML::Clone(config.parameters);
    const auto value = YAML::Load(invalid);
    const auto key = value.begin()->first.as<std::string>();
    parameters.parameters["szu"][key] = value[key];
    bool rejected = false;
    try { auto_buff_v2::make_rune_detector(parameters); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
  }

  config.parameters["szu"]["model"] = (dir / "rgb.xml").string();
  auto rgb = auto_buff_v2::make_rune_detector(config);
  rgb->set_enemy_red(true);
  const auto normalized = rgb->detect(image);
  assert(normalized.bullseyes.size() == 3);
  assert(std::abs(normalized.bullseyes[0].score - 0.95) < 1e-5);
  rgb->set_enemy_red(false);
  cv::Mat blue;
  cv::cvtColor(image, blue, cv::COLOR_BGR2RGB);
  assert(rgb->detect(blue).bullseyes.empty());
  config.parameters["szu"]["confidence_threshold"] = 1.0;
  auto strict = auto_buff_v2::make_rune_detector(config);
  strict->set_enemy_red(true);
  assert(strict->detect(image).bullseyes.empty());
  config.parameters["szu"]["confidence_threshold"] = -1;
  bool rejected = false;
  try { auto_buff_v2::make_rune_detector(config); }
  catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);

  // 使用附带的 ONNX 模型和源仓库样例，在比赛图像尺度下验证实际推理。
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path();
  config.parameters["szu"]["confidence_threshold"] = 0.65;
  config.parameters["szu"]["model"] = (root / "assets/buff_repvgg.xml").string();
  rejected = false;
  try { auto_buff_v2::make_rune_detector(config); }
  catch (const std::invalid_argument &) { rejected = true; }
  assert(rejected);  // 在推理前拒绝不兼容的九关键点 YOLO 输出。
  auto real_config = auto_buff_v2::BuffConfig::load((root / "configs/standard.yaml").string());
  real_config.detector.type = "szu";
  real_config.detector.parameters["szu"]["model"] = (root / "assets/szu_rune.onnx").string();
  auto real = auto_buff_v2::make_rune_detector(real_config.detector);
  real->set_enemy_red(false);
  const auto sample = cv::imread((root / "tests/data/szu_inactive.png").string());
  assert(!sample.empty());
  cv::Mat small, canvas = cv::Mat::zeros(1080, 1440, CV_8UC3);
  cv::resize(sample, small, {}, 0.25, 0.25);
  small.copyTo(canvas(cv::Rect(200, 200, small.cols, small.rows)));
  const auto real_result = real->detect(canvas);
  assert(real_result.bullseyes.size() == 1 && real_result.icons.size() == 1);
  assert(!real_result.bullseyes[0].active);
  assert(cv::norm(real_result.bullseyes[0].center - cv::Point2f(516, 282)) < 10);
  assert(cv::norm(real_result.icons[0].center - cv::Point2f(234, 422)) < 10);
  assert(real->detect(cv::Mat::zeros(480, 640, CV_8UC3)).bullseyes.empty());
  std::filesystem::remove_all(dir);
}
