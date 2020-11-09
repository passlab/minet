aux_source_directory(. CPP_SRCS)

add_library(dpu_test SHARED ${CPP_SRCS})
target_link_libraries(dpu_test vitis_dpu pthread)
