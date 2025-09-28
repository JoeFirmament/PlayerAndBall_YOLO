cmake_minimum_required(VERSION 3.10)
project(PoseAnalysis VERSION 1.0.0 LANGUAGES C CXX)

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
message(STATUS "PoseAnalysis Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "PoseAnalysis C++ standard: ${CMAKE_CXX_STANDARD}")

# 查找依赖库
find_package(OpenCV 4.0 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

if(OpenCV_FOUND)
    message(STATUS "PoseAnalysis OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "PoseAnalysis OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
endif()

# 依赖detector_lib
set(DETECTOR_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../detector_lib)
find_library(DETECTOR_LIB detector_lib_static HINTS ${DETECTOR_LIB_PATH}/build)
if(NOT DETECTOR_LIB)
    message(STATUS "detector_lib_static not found, will try to find detector_lib")
    find_library(DETECTOR_LIB detector_lib HINTS ${DETECTOR_LIB_PATH}/build)
endif()

if(NOT DETECTOR_LIB)
    message(WARNING "detector_lib not found. Please build detector_lib first.")
    set(DETECTOR_LIB "")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DETECTOR_LIB_PATH}/include
    ${OpenCV_INCLUDE_DIRS}
)

# 源文件
set(POSE_ANALYSIS_SOURCES
    src/pose_analyzer.cpp
    src/height_detector.cpp
    src/ball_request_detector.cpp
    src/id_priority_manager.cpp
)

# 创建静态库
add_library(pose_analysis_static STATIC ${POSE_ANALYSIS_SOURCES})
target_link_libraries(pose_analysis_static 
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)

# 创建动态库
add_library(pose_analysis SHARED ${POSE_ANALYSIS_SOURCES})
target_link_libraries(pose_analysis 
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)

# 设置库的输出版本
set_target_properties(pose_analysis PROPERTIES 
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# 示例程序
add_subdirectory(examples)

# 测试程序（可选）
option(BUILD_POSE_ANALYSIS_TESTS "Build pose analysis tests" ON)
if(BUILD_POSE_ANALYSIS_TESTS)
    add_subdirectory(tests)
endif()

# 安装配置
install(TARGETS pose_analysis pose_analysis_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/pose_analysis
    FILES_MATCHING PATTERN "*.h"
)