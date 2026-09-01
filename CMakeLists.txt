cmake_minimum_required(VERSION 3.16.3)

project(simple_auto_aim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(DEFAULT_FOXGLOVE_PLATFORM arm64)
else()
  set(DEFAULT_FOXGLOVE_PLATFORM x64)
endif()

set(FOXGLOVE_PLATFORM "${DEFAULT_FOXGLOVE_PLATFORM}" CACHE STRING
    "Foxglove SDK platform: x64 or arm64")
set_property(CACHE FOXGLOVE_PLATFORM PROPERTY STRINGS x64 arm64)
set(FOXGLOVE_SDK_DIR "${PROJECT_SOURCE_DIR}/thirdparty/foxglove_${FOXGLOVE_PLATFORM}")

if(NOT EXISTS "${FOXGLOVE_SDK_DIR}/include" OR
   NOT EXISTS "${FOXGLOVE_SDK_DIR}/lib/libfoxglove.a")
  message(FATAL_ERROR "Foxglove SDK is incomplete: ${FOXGLOVE_SDK_DIR}")
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
set(OpenVINO_DIR "/usr/local/runtime/cmake/")
find_package(OpenVINO REQUIRED)

file(GLOB FOXGLOVE_SDK_SOURCES CONFIGURE_DEPENDS
  "${FOXGLOVE_SDK_DIR}/src/*.cpp"
  "${FOXGLOVE_SDK_DIR}/src/server/*.cpp")
add_library(foxglove_sdk STATIC ${FOXGLOVE_SDK_SOURCES})
target_compile_features(foxglove_sdk PUBLIC cxx_std_20)
target_include_directories(foxglove_sdk PUBLIC "${FOXGLOVE_SDK_DIR}/include")
target_link_libraries(foxglove_sdk PUBLIC
  "${FOXGLOVE_SDK_DIR}/lib/libfoxglove.a"
  Threads::Threads)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(tools)
add_subdirectory(io)
add_subdirectory(tasks/auto_aim)
add_subdirectory(tasks/auto_buff)

add_executable(standard
  src/standard.cpp
  tools/detect_factory.cpp
  tools/foxglove_visualizer.cpp)
target_link_libraries(standard PRIVATE
  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp
  auto_aim
  tools
  io
  foxglove_sdk)

# 相机与手眼标定工具。
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
target_link_libraries(capture PRIVATE ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
