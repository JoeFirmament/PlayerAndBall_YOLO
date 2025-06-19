cmake_minimum_required(VERSION 3.10)

project(yolov8_post_basketball)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置头文件搜索路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fp16
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb_image
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64
    /usr/include/eigen3
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/librga/include
)

# 设置库文件搜索路径
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件列表
set(SRC_FILES
    src/main_camera_optimized.cc
    src/postprocess.cc
    src/yolov8-pose.cc
    src/BYTETracker.cpp
    src/STrack.cpp
    src/kalmanFilter.cpp
    src/lapjv.cpp
    src/utils.cpp
    utils/image_utils.c
    utils/file_utils.c
    utils/image_drawing.c
)

# 生成可执行文件
add_executable(yolov8_pose_basketball ${SRC_FILES})

# 链接库
# 注意：如有其它依赖库可在此处添加
# librknnrt.so、librga.so、OpenCV、pthread、dl

target_link_libraries(yolov8_pose_basketball
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 设置运行时库搜索路径
set_target_properties(yolov8_pose_basketball PROPERTIES
    INSTALL_RPATH "$ORIGIN/libs"
)

# 安装目标
install(TARGETS yolov8_pose_basketball DESTINATION .)
install(FILES models/yolov8_pose.rknn models/basketball_player_rk3588.rknn DESTINATION ./models) 