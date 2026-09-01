cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT
    adaptive_delay_controller.cpp
    camera2gimbal_extrinsic.cpp
    exiter.cpp
    extended_kalman_filter.cpp
    filter_base.cpp
    invariant_pose_filter.cpp
    unscented_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
)

target_include_directories(tools PRIVATE ${PROJECT_SOURCE_DIR}/io/serial/include)
