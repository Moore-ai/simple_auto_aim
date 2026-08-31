# simple_auto_aim 环境配置

## 1 项目环境

- 操作系统：Ubuntu 22.04
- 运算平台：NUC12WSKI7（i7-1260P，16GB）
- 相机型号：海康 MV-CS016-10UC
- 镜头型号：海康官方 6mm 镜头
- 下位机型号：RoboMaster 开发板 C 型（STM32F407）
- IMU 型号：C 板内置 BMI088
- 通信方式：USB2CAN 或 MicroUSB 虚拟串口

## 2 依赖安装

### 2.1 第三方 SDK

根据相机型号安装以下 SDK 之一：

- [MindVision SDK](https://mindvision.com.cn/category/software/sdk-installation-package/)
- [HikRobot SDK](https://www.hikrobotics.com/cn2/source/support/software/MVS_STD_GML_V2.1.2_231116.zip)

安装以下库：

- [OpenVINO](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html)
- [Ceres Solver](http://ceres-solver.org/installation.html)

### 2.2 系统软件包

```bash
sudo apt install -y \
    git \
    g++ \
    cmake \
    can-utils \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libusb-1.0-0-dev \
    nlohmann-json3-dev \
    openssh-server \
    screen
```

## 3 编译与运行

在项目根目录执行：

```bash
cmake -S . -B build
cmake --build build --parallel 1
```

运行自瞄测试 demo：

```bash
./build/auto_aim_test
```

其他常用可执行文件：

```bash
./build/camera_test
./build/detector_video_test
./build/planner_test
./build/auto_buff_test
```

## 4 开机自启

确保已安装 `screen`：

```bash
sudo apt install screen
```

创建自启文件：

```bash
mkdir -p ~/.config/autostart/
touch ~/.config/autostart/simple_auto_aim.desktop
```

写入以下内容。`Exec` 必须使用项目的绝对路径：

```ini
[Desktop Entry]
Type=Application
Exec=/home/rm/Desktop/simple_auto_aim/autostart.sh
Name=simple_auto_aim
```

授予启动脚本执行权限：

```bash
chmod +x autostart.sh
```

## 5 USB2CAN 设置（可选）

创建 udev 规则文件：

```bash
sudo touch /etc/udev/rules.d/99-can-up.rules
```

写入：

```udev
ACTION=="add", KERNEL=="can0", RUN+="/sbin/ip link set can0 up type can bitrate 1000000"
ACTION=="add", KERNEL=="can1", RUN+="/sbin/ip link set can1 up type can bitrate 1000000"
```

## 6 GPU 推理（可选）

以下示例安装 Intel GPU 运行时依赖：

```bash
mkdir -p neo
cd neo

wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-core_1.0.13463.18_amd64.deb
wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.13463.18/intel-igc-opencl_1.0.13463.18_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu-dbgsym_1.3.25812.14_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-level-zero-gpu_1.3.25812.14_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd-dbgsym_23.09.25812.14_amd64.ddeb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/intel-opencl-icd_23.09.25812.14_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/libigdgmm12_22.3.0_amd64.deb
wget https://github.com/intel/compute-runtime/releases/download/23.09.25812.14/ww09.sum

sha256sum -c ww09.sum
sudo dpkg -i *.deb
```

使用 GPU 异步推理（`async-infer`）时，最高显示分辨率限制为 1920×1080（24 Hz）。

## 7 串口设置

将当前用户加入 `dialout` 组：

```bash
sudo usermod -a -G dialout $USER
```

获取设备信息（将设备名替换为实际端口）：

```bash
udevadm info -a -n /dev/ttyACM0 | grep -E '({serial}|{idVendor}|{idProduct})'
```

创建规则文件：

```bash
sudo touch /etc/udev/rules.d/99-usb-serial.rules
```

写入规则，并将示例 ID 替换为实际值：

```udev
SUBSYSTEM=="tty", ATTRS{idVendor}=="1234", ATTRS{idProduct}=="1234", ATTRS{serial}=="A1234567", SYMLINK+="gimbal"
```

重新加载规则并检查软链接：

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
ls -l /dev/gimbal
```

## 8 手眼标定

本机
```powershell
$env:DISPLAY = "127.0.0.1:0.0"
ssh -Y hfut@192.168.xxx.xx
```
进入远程后
```bash
cd ~/simple_auto_aim
./build/capture
```
