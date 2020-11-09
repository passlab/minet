aux_source_directory(. CPP_SRCS)
add_library(vitis_dpu SHARED ${CPP_SRCS})
target_link_libraries(vitis_dpu glog json-c dl)
message("MY_DEPS is ${MY_DEPS}")

set_target_properties(vitis_dpu PROPERTIES
  VERSION "${PROJECT_VERSION}"
  SOVERSION "${PROJECT_VERSION_MAJOR}")
install(TARGETS vitis_dpu DESTINATION lib)
