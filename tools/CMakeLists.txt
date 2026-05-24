cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT
    adaptive_delay_controller.cpp
    exiter.cpp
    extended_kalman_filter.cpp
    filter_base.cpp
    invariant_pose_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    plotter.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
    pid.cpp
    crc.cpp
)