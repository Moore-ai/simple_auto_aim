cmake_minimum_required(VERSION 3.16.3)

project(simple_auto_aim)

enable_testing()

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
add_subdirectory(tasks/auto_buff_v2)

add_executable(standard
  src/standard.cpp
  src/mode_runtime.cpp
  tools/detect_factory.cpp
  tools/foxglove_visualizer.cpp)
target_link_libraries(standard PRIVATE
  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp
  auto_aim
  auto_buff_v2
  tools
  io
  foxglove_sdk)

add_test(NAME standard_cli_test
  COMMAND ${PROJECT_SOURCE_DIR}/tests/standard_cli_test.sh $<TARGET_FILE:standard>)

add_test(NAME legacy_buff_ownership_test
  COMMAND ${PROJECT_SOURCE_DIR}/tests/legacy_buff_ownership_test.sh ${CMAKE_BINARY_DIR})

add_executable(mode_runtime_test tests/mode_runtime_test.cpp)
target_compile_options(mode_runtime_test PRIVATE -UNDEBUG)
add_test(NAME mode_runtime_test COMMAND mode_runtime_test)

add_executable(processed_frame_test tests/processed_frame_test.cpp)
target_compile_options(processed_frame_test PRIVATE -UNDEBUG)
target_link_libraries(processed_frame_test PRIVATE io)
add_test(NAME processed_frame_test COMMAND processed_frame_test)

add_executable(frame_facts_test tests/frame_facts_test.cpp)
target_compile_options(frame_facts_test PRIVATE -UNDEBUG)
target_link_libraries(frame_facts_test PRIVATE ${OpenCV_LIBS} io)
add_test(NAME frame_facts_test COMMAND frame_facts_test)

add_executable(auto_aim_frame_runtime_test tests/auto_aim_frame_runtime_test.cpp)
target_compile_options(auto_aim_frame_runtime_test PRIVATE -UNDEBUG)
target_link_libraries(auto_aim_frame_runtime_test PRIVATE io)
add_test(NAME auto_aim_frame_runtime_test COMMAND auto_aim_frame_runtime_test)

add_executable(buff_v2_model_test tests/buff_v2_model_test.cpp)
target_compile_options(buff_v2_model_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_model_test PRIVATE auto_buff_v2)

add_executable(buff_v2_config_test tests/buff_v2_config_test.cpp)
target_compile_options(buff_v2_config_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_config_test PRIVATE auto_buff_v2)
add_test(NAME buff_v2_config_test COMMAND buff_v2_config_test)

add_executable(buff_v2_assignment_test tests/buff_v2_assignment_test.cpp)
target_compile_options(buff_v2_assignment_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_assignment_test PRIVATE auto_buff_v2)

add_executable(buff_v2_detector_test tests/buff_v2_detector_test.cpp)
target_compile_options(buff_v2_detector_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_detector_test PRIVATE auto_buff_v2)

add_executable(buff_v2_frame_processor_test tests/buff_v2_frame_processor_test.cpp)
target_compile_options(buff_v2_frame_processor_test PRIVATE -UNDEBUG)
target_link_libraries(
  buff_v2_frame_processor_test PRIVATE auto_buff_v2)
add_test(NAME buff_v2_frame_processor_test COMMAND buff_v2_frame_processor_test)

add_executable(buff_v2_pnp_test tests/buff_v2_pnp_test.cpp)
target_compile_options(buff_v2_pnp_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_pnp_test PRIVATE auto_buff_v2)

add_executable(buff_v2_planner_test tests/buff_v2_planner_test.cpp)
target_compile_options(buff_v2_planner_test PRIVATE -UNDEBUG)
target_link_libraries(buff_v2_planner_test PRIVATE auto_buff_v2 fmt::fmt)

# 相机与手眼标定工具。
add_executable(capture calibration/capture.cpp)
add_executable(calibrate_camera calibration/calibrate_camera.cpp)
add_executable(calibrate_handeye calibration/calibrate_handeye.cpp)
target_link_libraries(capture PRIVATE ${OpenCV_LIBS} fmt::fmt tools io)
target_link_libraries(calibrate_camera PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)
target_link_libraries(calibrate_handeye PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools)

add_executable(planner_fly_time_iteration_test tests/planner_fly_time_iteration_test.cpp)
target_compile_options(planner_fly_time_iteration_test PRIVATE -UNDEBUG)
target_link_libraries(planner_fly_time_iteration_test PRIVATE
  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp
  auto_aim
  tools
  io)

add_executable(planner_anti_spin_test tests/planner_anti_spin_test.cpp)
target_compile_options(planner_anti_spin_test PRIVATE -UNDEBUG)
target_link_libraries(planner_anti_spin_test PRIVATE
  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp
  auto_aim
  tools
  io)

add_executable(gimbal_virtual_serial_test tests/gimbal_virtual_serial_test.cpp)
target_compile_options(gimbal_virtual_serial_test PRIVATE -UNDEBUG)
target_link_libraries(gimbal_virtual_serial_test PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp tools io)

add_executable(outpost_snapshot_test tests/outpost_snapshot_test.cpp)
target_link_libraries(outpost_snapshot_test PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(outpost_target_test tests/outpost_target_test.cpp)
target_link_libraries(outpost_target_test PRIVATE ${OpenCV_LIBS} fmt::fmt yaml-cpp auto_aim tools io)

add_executable(foxglove_visualizer_image_test
  tests/foxglove_visualizer_image_test.cpp
  tools/foxglove_visualizer.cpp)
target_compile_options(foxglove_visualizer_image_test PRIVATE -UNDEBUG)
target_link_libraries(foxglove_visualizer_image_test PRIVATE
  ${OpenCV_LIBS}
  fmt::fmt
  yaml-cpp
  auto_aim
  tools
  io
  foxglove_sdk)
