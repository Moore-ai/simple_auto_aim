# simple_auto_aim

RoboMaster 自瞄与打符程序，支持普通装甲板、前哨站、小能量机关和大能量机关。
统一入口为 `src/standard.cpp`，通过串口与下位机通信，通过 Foxglove 查看图像与运行数据。

## 1. 环境配置

环境配置参考 [同济大学 SuperPower 的 sp_vision_25](https://github.com/TongjiSuperPower/sp_vision_25#32-编译方式)，并按本仓库当前依赖调整。

### 系统与依赖

使用 Ubuntu 22.04，支持 x86_64 和 ARM64。WSL 开发时，运行实机程序需要先将相机和串口设备接入上位机。

```bash
sudo apt update
sudo apt install -y \
    git \
    g++ \
    cmake \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    openssh-server \
    screen
```

根据使用的工业相机安装对应 SDK：

- [海康 HikRobot SDK](https://www.hikrobotics.com/cn2/service/download/)
- [迈德威 MindVision SDK](https://mindvision.com.cn/category/software/sdk-installation-package/)

另需安装 [OpenVINO Runtime](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)。
当前 CMake 中的 `OpenVINO_DIR` 指向 `/usr/local/runtime/cmake`；若安装路径不同，需同步修改
顶层和 `tasks/auto_aim/CMakeLists.txt` 中的路径。

Foxglove SDK 已包含在 `thirdparty/foxglove_x64` 和 `thirdparty/foxglove_arm64`，构建时会自动选择对应架构，无需手动设置。

### 串口权限与固定设备名（可选）

将当前用户加入 `dialout` 组，注销并重新登录后生效：

```bash
sudo usermod -a -G dialout "$USER"
```

可直接在配置中使用 `/dev/ttyACM0` 等设备名。如需固定为 `/dev/gimbal`，先查询实际设备信息：

```bash
udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
sudo touch /etc/udev/rules.d/99-usb-serial.rules
```

在规则文件中写入以下内容，将示例 ID 替换为查询结果：

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="1234", ATTRS{serial}=="A1234567", SYMLINK+="gimbal"
```

加载规则并检查设备：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/gimbal
```

使用固定设备名时，将 `configs/standard.yaml` 的 `com_port` 改为 `/dev/gimbal`。

## 2. 构建与运行

在仓库根目录执行：

```bash
cmake -S . -B build
cmake --build build --target standard -j2
```

运行前，先检查 `configs/standard.yaml` 中的相机、串口、模型路径及标定参数。
以下示例由下位机反馈决定目标颜色：

```bash
./build/standard configs/standard.yaml --mode=0 --target-color=none # 自瞄与前哨站
./build/standard configs/standard.yaml --mode=1 --target-color=none # 小符
./build/standard configs/standard.yaml --mode=2 --target-color=none # 大符
```

`--target-color` 为必填参数：`red` 检测红色目标，`blue` 检测蓝色目标，`none` 使用下位机反馈。
例如固定检测蓝色小符：

```bash
./build/standard configs/standard.yaml --mode=1 --target-color=blue
```

使用 `./build/standard --help` 查看命令行说明。

## 3. 相机与手眼标定

标定使用 [configs/calibration.yaml](configs/calibration.yaml)，请先检查其中的相机与标定板参数。

```yaml
# 对称圆点阵标定板
pattern_cols: 11
pattern_rows: 8
center_distance_mm: 20
```

流程为先启动./build/capture采集图片，然后用./build/calibrate_camera标定相机或者./build/calibrate_handeye手眼标定。手眼标定程序可能存在问题，几次标定的外参里 t_gimbal_to_camera 总是接近零，因此建议直接从机械结构获取此变换

```bash
cmake --build build --target capture calibrate_camera calibrate_handeye -j2

# 采集图像与云台姿态
./build/capture -c configs/calibration.yaml -o assets/img_with_q

# 相机内参标定
./build/calibrate_camera -c configs/calibration.yaml assets/img_with_q

# 手眼标定
./build/calibrate_handeye -c configs/calibration.yaml assets/img_with_q
```

完成后，将相机内参、畸变和手眼外参更新到 `configs/standard.yaml`。
当前使用 `camera2gimbal_mode: xyz_ypr`；若使用标定输出的矩阵外参，改为 `matrix`，
并填写 `R_camera2gimbal` 和 `t_camera2gimbal`。

## 4. Foxglove 调试

开启 `foxglove.enable`，启动程序后，在 Foxglove 中建立 WebSocket 连接：

```text
ws://localhost:8765
```

远程连接时，将 `localhost` 替换为运行程序的主机 IP。
Image 面板可选择 `/image_raw`（原始图像）、`/image_detection`（检测图像）和 `/image`（运行标注）。
Plot 面板可选择目标状态、规划和云台相关数据。

### Njust 检测专用入口

这部分主要用于调试南京理工的纯传统大符检测器，**效果不好不打算使用，可以跳过**

两个调试程序分别查看 R 标和靶心检测，仅运行检测与数据发布。
使用前将 `buff_v2.detector.type` 改为 `njust`，并开启 `foxglove.enable`。

```bash
cmake --build build --target buff_v2_njust_r_detector_debug buff_v2_njust_bullseye_detector_debug -j2

# R 标检测
./build/buff_v2_njust_r_detector_debug configs/standard.yaml --mode=1 --target-color=blue

# 靶心检测
./build/buff_v2_njust_bullseye_detector_debug configs/standard.yaml --mode=1 --target-color=blue
```

Image 面板选择 `/buff_v2/njust/image` 或 `/buff_v2/njust/image_raw`。
Plot 面板常用路径：

| 数据 | 路径 |
| --- | --- |
| 候选半径 | `/buff_v2/njust/detector.candidates[:].radius` |
| 候选面积 | `/buff_v2/njust/detector.candidates[:].area` |
| 靶心半径范围 | `/buff_v2/njust/detector.min_radius`、`/buff_v2/njust/detector.max_radius` |
| R 标面积范围 | `/buff_v2/njust/detector.min_icon_area`、`/buff_v2/njust/detector.max_icon_area` |
| R 标评分 | `/buff_v2/njust/detector.icon_scores[:].score` |
| R 标评分阈值 | `/buff_v2/njust/detector.match_threshold` |

主入口与两个检测调试入口共用相机、串口和 Foxglove 端口，每次运行其中一个。

## 5. 项目结构

```text
simple_auto_aim/
├── CMakeLists.txt                # 构建配置
├── src/
│   ├── standard.cpp              # 自瞄与打符统一入口
│   ├── mode_runtime.*            # 模式运行时
│   ├── buff_v2_njust_r_detector_debug.cpp
│   └── buff_v2_njust_bullseye_detector_debug.cpp
├── configs/
│   ├── standard.yaml             # 运行配置，参数说明见行尾注释
│   └── calibration.yaml          # 标定配置
├── tasks/
│   ├── auto_aim/                 # 装甲板检测与目标估计
│   │   └── planner/              # MPC 轨迹规划
│   ├── auto_buff_v2/             # 当前打符链路
│   │   ├── detectors/            # 检测器
│   │   ├── models/               # 状态模型与运动预测
│   │   ├── aimers/               # 瞄准器
│   │   └── planners/             # 规划器
│   └── auto_buff/                # 保留的旧打符代码（目前暂未启用）
├── io/                           # 相机、串口与云台通信
├── tools/                        # 弹道、滤波、日志、录像与 Foxglove 工具
├── calibration/                  # 标定数据采集、相机与手眼标定
├── assets/                       # 模型权重与示例素材
├── thirdparty/                   # Foxglove SDK
└── tests/                        # 测试与回归检查，主要由 gpt 编写
```
