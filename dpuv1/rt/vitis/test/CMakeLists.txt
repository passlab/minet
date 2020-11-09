aux_source_directory(. SRC_FILS)

foreach(FIL ${SRC_FILS})
  get_filename_component(FIL_NAME ${FIL} NAME_WE)
  set(BIN ${FIL_NAME}.bin)

  add_executable(${BIN} ${FIL})
  target_link_libraries(${BIN} dpu_test)
endforeach()
