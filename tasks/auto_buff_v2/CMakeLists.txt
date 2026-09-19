add_library(auto_buff_v2 STATIC
  buff_config.cpp
  rune_detector.cpp
  rune_energy_fitter.cpp
  rune_model.cpp
  rune_predictor.cpp
  buff_planner.cpp)
target_link_libraries(auto_buff_v2 PUBLIC ${OpenCV_LIBS} yaml-cpp tools io auto_aim)
