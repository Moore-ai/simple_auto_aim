#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
{
  std::vector<cv::Point3f> centers_3d;

  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      centers_3d.push_back({j * center_distance, i * center_distance, 0});

  return centers_3d;
}

Eigen::Quaterniond read_q(const std::string & q_path)
{
  std::ifstream q_file(q_path);
  double w, x, y, z;
  q_file >> w >> x >> y >> z;
  return {w, x, y, z};
}

void load(
  const std::string & input_folder, const std::string & config_path,
  Eigen::Matrix3d & R_gimbal2imubody, std::vector<cv::Mat> & R_gimbal2world_list,
  std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
  std::vector<cv::Mat> & tvecs)
{
  // 读取yaml参数
  auto yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto center_distance_mm = yaml["center_distance_mm"].as<double>();
  auto rpy_gi = yaml["gimbal2imubody"]["rpy"].as<std::vector<double>>();
  R_gimbal2imubody = tools::rotation_matrix(Eigen::Vector3d(rpy_gi.data()));
  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

  cv::Size pattern_size(pattern_cols, pattern_rows);
  cv::Matx33d camera_matrix(camera_matrix_data.data());
  cv::Mat distort_coeffs(distort_coeffs_data);

  for (int i = 1; true; i++) {
    // 读取图片和对应四元数
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);
    auto img = cv::imread(img_path);
    Eigen::Quaterniond q = read_q(q_path);
    if (img.empty()) break;

    // 计算云台的欧拉角
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // degree

    // 在图片上显示云台的欧拉角，用来检验R_gimbal2imubody是否正确
    auto drawing = img.clone();
    tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

    // 识别标定板
    std::vector<cv::Point2f> centers_2d;
    auto success = cv::findCirclesGrid(img, pattern_size, centers_2d);  // 默认是对称圆点图案

    // 显示识别结果
    cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
    cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
    cv::imshow("Press any to continue", drawing);
    cv::waitKey(0);

    // 输出识别结果
    fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
    if (!success) continue;

    // 计算所需的数据
    cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat R_gimbal2world_cv;
    cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
    cv::Mat rvec, tvec;
    auto centers_3d_ = centers_3d(pattern_size, center_distance_mm);
    cv::solvePnP(
      centers_3d_, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

    // 记录所需的数据
    R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
    t_gimbal2world_list.emplace_back(t_gimbal2world);
    rvecs.emplace_back(rvec);
    tvecs.emplace_back(tvec);
  }
}

void print_yaml(
  const Eigen::Matrix3d & R_gimbal2imubody, const cv::Mat & R_camera2gimbal,
  const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
{
  // 从旋转矩阵反算 rpy
  Eigen::Vector3d rpy_gi = tools::eulers(R_gimbal2imubody, 2, 1, 0);

  // 标定结果 R_camera2gimbal 是 camera_optical_frame → gimbal 的组合变换
  // 分解出去除光轴旋转的 camera_joint（gimbal → camera_link）:
  // R_cj = R_combined * R_op^T
  Eigen::Matrix3d R_cg_eigen;
  cv::cv2eigen(R_camera2gimbal, R_cg_eigen);
  const Eigen::Matrix3d R_optical = tools::rotation_matrix(Eigen::Vector3d(-CV_PI / 2, 0, -CV_PI / 2));
  Eigen::Matrix3d R_cj = R_cg_eigen * R_optical.transpose();
  Eigen::Vector3d rpy_cj = tools::eulers(R_cj, 2, 1, 0);

  // t_camera2gimbal → xyz（不变，因为 t_optical = 0）
  std::vector<double> xyz_cj(3);
  for (int i = 0; i < 3; i++) xyz_cj[i] = t_camera2gimbal.at<double>(i);

  YAML::Emitter result;
  result << YAML::BeginMap;

  result << YAML::Key << "gimbal2imubody";
  result << YAML::Value << YAML::BeginMap;
  result << YAML::Key << "xyz" << YAML::Value << YAML::Flow << std::vector<double>{0, 0, 0};
  result << YAML::Key << "rpy" << YAML::Value << YAML::Flow
         << std::vector<double>{rpy_gi[0], rpy_gi[1], rpy_gi[2]};
  result << YAML::EndMap;
  result << YAML::Newline;
  result << YAML::Newline;
  result << YAML::Comment(fmt::format(
    "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));

  result << YAML::Key << "camera_joint";
  result << YAML::Value << YAML::BeginMap;
  result << YAML::Key << "xyz" << YAML::Value << YAML::Flow << xyz_cj;
  result << YAML::Key << "rpy" << YAML::Value << YAML::Flow
         << std::vector<double>{rpy_cj[0], rpy_cj[1], rpy_cj[2]};
  result << YAML::EndMap;
  result << YAML::Newline;

  result << YAML::EndMap;

  fmt::print("\n{}\n", result.c_str());
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  // 从输入文件夹中加载标定所需的数据
  Eigen::Matrix3d R_gimbal2imubody;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;
  load(
    input_folder, config_path, R_gimbal2imubody, R_gimbal2world_list, t_gimbal2world_list,
    rvecs, tvecs);

  // 手眼标定
  cv::Mat R_camera2gimbal, t_camera2gimbal;
  cv::calibrateHandEye(
    R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
  t_camera2gimbal /= 1e3;  // mm to m

  // 计算相机同理想情况的偏角
  Eigen::Matrix3d R_camera2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

  // 输出yaml
  print_yaml(R_gimbal2imubody, R_camera2gimbal, t_camera2gimbal, ypr);
}
