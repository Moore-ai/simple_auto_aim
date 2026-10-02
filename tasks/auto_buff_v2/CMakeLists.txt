add_library(auto_buff_v2 STATIC
  buff_config.cpp
  buff_frame_processor.cpp
  detectors/njust_rune_detector.cpp
  detectors/climber_rune_detector.cpp
  detectors/szu_rune_detector.cpp
  rune_energy_fitter.cpp
  rune_model.cpp
  rune_predictor.cpp
  buff_planner.cpp
  buff_planner_factory.cpp)
target_link_libraries(auto_buff_v2
  PUBLIC ${OpenCV_LIBS} yaml-cpp tools io auto_aim
  PRIVATE fmt::fmt openvino::runtime)
