# 分发包独立编译CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DetectorLibExamples)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找OpenCV
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)

# 如果OpenCV4不存在，尝试opencv
if(NOT OPENCV_FOUND)
    pkg_check_modules(OPENCV REQUIRED opencv)
endif()

# 设置库路径（相对于examples目录）
set(DETECTOR_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
set(DETECTOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")

# 检查库文件是否存在
if(NOT EXISTS "${DETECTOR_LIB_DIR}/libdetector_lib.so")
    message(FATAL_ERROR "❌ 找不到detector_lib库文件: ${DETECTOR_LIB_DIR}/libdetector_lib.so")
endif()

# 示例程序源文件
set(EXAMPLE_SOURCES
    test_detector_lib.cpp
    pose_image.cpp
    pose_image_with_homography.cpp
    pose_image_with_polar.cpp
    rim_basketball_image.cpp
    pose_camera_bytetrack_homography.cpp
    # pose_image_smart_path.cpp  # 保留源码但不编译，供用户参考
)

# 为每个示例创建可执行文件
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_src})
        add_executable(${example_name} ${example_src})

        # 包含目录
        target_include_directories(${example_name} PRIVATE
            ${DETECTOR_INCLUDE_DIR}
            ${OPENCV_INCLUDE_DIRS}
        )

        # 编译选项
        target_compile_options(${example_name} PRIVATE
            ${OPENCV_CFLAGS_OTHER}
        )

        # 链接库
        target_link_libraries(${example_name}
            ${DETECTOR_LIB_DIR}/libdetector_lib.so
            ${DETECTOR_LIB_DIR}/librknnrt.so
            ${OPENCV_LIBRARIES}
            pthread
        )

        # 设置RPATH - 运行时自动找到相对路径的库
        set_target_properties(${example_name} PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
            BUILD_RPATH "${DETECTOR_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )

        message(STATUS "✅ 创建示例程序: ${example_name}")
    else()
        message(WARNING "⚠️  跳过不存在的源文件: ${example_src}")
    endif()
endforeach()

# 显示配置信息
message(STATUS "📋 编译配置:")
message(STATUS "  OpenCV版本: ${OPENCV_VERSION}")
message(STATUS "  OpenCV库: ${OPENCV_LIBRARIES}")
message(STATUS "  Detector库路径: ${DETECTOR_LIB_DIR}")
message(STATUS "  包含目录: ${DETECTOR_INCLUDE_DIR}")