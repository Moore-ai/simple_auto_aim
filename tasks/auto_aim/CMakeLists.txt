cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/usr/local/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    lightbar_detector.cpp
    solver.cpp
    aimer.cpp
    detection_result.cpp
    observation_geometry.cpp
    observation_path.cpp
    target_estimator.cpp
    target_state.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(auto_aim io openvino::runtime tinympcstatic)
