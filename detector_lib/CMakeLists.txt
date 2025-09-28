cmake_minimum_required(VERSION 3.10)
project(DetectorLib VERSION 1.0.4 LANGUAGES C CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

# 输出信息
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")

# 查找依赖库
find_package(OpenCV 4.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
endif()

# RKNN库配置
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 优先使用系统RKNN库
find_library(RKNN_RT_LIB rknnrt HINTS /usr/lib /lib)
if(RKNN_RT_LIB)
    message(STATUS "Found system RKNN runtime: ${RKNN_RT_LIB}")
else()
    set(RKNN_RT_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/librknnrt.so)
    if(EXISTS ${RKNN_RT_LIB})
        message(STATUS "Found project RKNN runtime: ${RKNN_RT_LIB}")
    else()
        message(FATAL_ERROR "RKNN runtime library not found")
    endif()
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
    ${RKNN_API_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/rknpu2/include
)

# 源文件定义
# 核心检测器库源文件
set(DETECTOR_LIB_SOURCES
    src/PoseDetectorLib.cpp
    src/RimBasketballDetectorLib.cpp
    src/detector_lib.cpp
    src/detector_path_utils.cpp
    src/detector_file_utils.c
    src/rim_basketball_postprocess_simple.cpp
    src/internal/detector_common.cpp
)

# 合并所有源文件
set(ALL_SOURCES ${DETECTOR_LIB_SOURCES})

# 创建主静态库（包含detector + pose_analysis）
add_library(detector_lib_static STATIC ${ALL_SOURCES})
target_link_libraries(detector_lib_static
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    ${RKNN_RT_LIB}
    pthread
)

# 创建主动态库
add_library(detector_lib SHARED ${ALL_SOURCES})
target_link_libraries(detector_lib
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
    ${RKNN_RT_LIB}
    pthread
)


# 安装规则
install(TARGETS detector_lib detector_lib_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# 选项
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
