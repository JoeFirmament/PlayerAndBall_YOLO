cmake_minimum_required(VERSION 3.10)

project(yolov8_pose_basketball)

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
    src/letterbox_utils.cc
    src/basketball_postprocess.cpp
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

# 添加测试程序
add_executable(test_basketball_rknn src/test_basketball_rknn.cpp)

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

# 为测试程序链接库
target_link_libraries(test_basketball_rknn
    rknnrt
    ${OpenCV_LIBS}
)

# 设置运行时库搜索路径
set_target_properties(yolov8_pose_basketball PROPERTIES
    INSTALL_RPATH "$ORIGIN/libs"
)

# RGA性能测试程序
add_executable(rga_resize_test src/rga_resize_test.cpp)
target_link_libraries(rga_resize_test
    rga
    ${OpenCV_LIBS}
    pthread
)

# 篮球框和篮球检测器
add_executable(rim_basketball_detector 
    src/rim_basketball_detector.cc
    src/rim_basketball_postprocess.cpp
)
target_link_libraries(rim_basketball_detector
    rknnrt
    rga
    ${OpenCV_LIBS}
    pthread
    dl
)

# 安装目标
install(TARGETS yolov8_pose_basketball DESTINATION .)
install(FILES models/yolov8_pose.rknn models/basketball_player_rk3588.rknn DESTINATION ./models) 