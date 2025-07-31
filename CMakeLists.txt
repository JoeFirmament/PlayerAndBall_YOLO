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
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/include
)

# 设置库文件搜索路径
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rknpu2/Linux/aarch64
)

# 查找OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 原有的双线程版本源文件
set(DUAL_THREAD_SRC
    src/main_camera_optimized.cc
    src/pose_postprocess.cc
    src/pose_yolov8.cc
    src/pose_letterbox_utils.cc
    # src/basketball_postprocess.cpp  # 已删除
    src/BYTETracker.cpp
    src/STrack.cpp
    src/kalmanFilter.cpp
    src/lapjv.cpp
    src/utils.cpp
    utils/image_utils.c
    utils/file_utils.c
    utils/image_drawing.c
)

# 纯姿态检测版本源文件
set(POSE_ONLY_SRC
    src/pose_main.cc
    src/pose_postprocess.cc
    src/pose_yolov8.cc
    src/pose_letterbox_utils.cc
    src/BYTETracker.cpp
    src/STrack.cpp
    src/kalmanFilter.cpp
    src/lapjv.cpp
    src/utils.cpp
    utils/image_utils.c
    utils/file_utils.c
    utils/image_drawing.c
)

# 篮筐篮球检测版本源文件 (复杂版 - 已归档，不再使用)
# set(RIM_BASKETBALL_SRC
#     src/rim_basketball_detector_updated.cc
#     archived/src/rim_basketball_postprocess.cpp
#     utils/image_utils.c
#     utils/file_utils.c
# )

# 篮筐篮球检测版本源文件 (简化版 - 基于ref/rknn_yolov8_ref)
set(RIM_BASKETBALL_SIMPLE_SRC
    src/rim_basketball_detector_updated.cc
    src/rim_basketball_postprocess_simple.cpp
    utils/image_utils.c
    utils/file_utils.c
)

# 双摄像头双线程检测系统源文件
set(DUAL_CAMERA_SRC
    src/dual_camera_detector.cc
    src/pose_postprocess.cc
    src/pose_yolov8.cc
    src/pose_letterbox_utils.cc
    src/rim_basketball_postprocess_simple.cpp
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
add_executable(yolov8_pose_basketball ${DUAL_THREAD_SRC})            # 原有双线程版本
add_executable(yolov8_pose_only ${POSE_ONLY_SRC})                   # 纯姿态检测版本
add_executable(rim_basketball_detector_v2 ${RIM_BASKETBALL_SIMPLE_SRC})     # 篮筐篮球检测版本 (使用优化的simple版)
add_executable(rim_basketball_detector_simple ${RIM_BASKETBALL_SIMPLE_SRC}) # 篮筐篮球检测版本 (简化版)
add_executable(dual_camera_detector ${DUAL_CAMERA_SRC})             # 双摄像头双线程检测系统

# 添加测试程序 (已删除)
# add_executable(test_basketball_rknn src/test_basketball_rknn.cpp)

# 链接库
# 注意：如有其它依赖库可在此处添加
# librknnrt.so、librga.so、OpenCV、pthread、dl

# 链接库 - 原有双线程版本
target_link_libraries(yolov8_pose_basketball
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 链接库 - 纯姿态检测版本
target_link_libraries(yolov8_pose_only
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 链接库 - 篮筐篮球检测版本 (复杂版)
target_link_libraries(rim_basketball_detector_v2
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 链接库 - 篮筐篮球检测版本 (简化版)
target_link_libraries(rim_basketball_detector_simple
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 链接库 - 双摄像头双线程检测系统
target_link_libraries(dual_camera_detector
    rknnrt
    rga
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a
    ${OpenCV_LIBS}
    pthread
    dl
)

# 为测试程序链接库 (已删除)
# target_link_libraries(test_basketball_rknn
#     rknnrt
#     ${OpenCV_LIBS}
# )

# 设置运行时库搜索路径
set_target_properties(yolov8_pose_basketball PROPERTIES
    INSTALL_RPATH "$ORIGIN/libs"
)

# RGA性能测试程序 (已删除)
# add_executable(rga_resize_test src/rga_resize_test.cpp)
# target_link_libraries(rga_resize_test
#     rga
#     ${OpenCV_LIBS}
#     pthread
# )

# 篮球框和篮球检测器 (旧版本，已归档)
# add_executable(rim_basketball_detector 
#     src/rim_basketball_detector.cc
#     archived/src/rim_basketball_postprocess.cpp
# )
# target_link_libraries(rim_basketball_detector
#     rknnrt
#     rga
#     ${OpenCV_LIBS}
#     pthread
#     dl
# )

# 安装目标
install(TARGETS yolov8_pose_basketball DESTINATION .)
install(FILES models/yolov8_pose.rknn models/basketball_player_rk3588.rknn DESTINATION ./models) 