cmake_minimum_required(VERSION 3.5)
project(vitis-runtime-api VERSION 0.0.1 LANGUAGES C CXX)

include_directories(${CMAKE_SOURCE_DIR}/include/dpu)
if(DEFINED ENV{CONDA_PREFIX})
  include_directories($ENV{CONDA_PREFIX}/include)
  link_directories($ENV{CONDA_PREFIX}/lib)
endif()
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/dpu 
    ${CMAKE_SOURCE_DIR}/include/vip
 DESTINATION include/vitis)
include(cmake/XilinxCommon.cmake)

set(CMAKE_MACOSX_RPATH 1)

add_subdirectory(src)
add_subdirectory(test)
