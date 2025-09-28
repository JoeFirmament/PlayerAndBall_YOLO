# 示例程序

# 基础姿态分析示例
add_executable(pose_analysis_example pose_analysis_example.cpp)
target_link_libraries(pose_analysis_example 
    pose_analysis_static
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
)

# 简单姿态分析示例
add_executable(simple_pose_analysis_example simple_pose_analysis_example.cpp)
target_link_libraries(simple_pose_analysis_example 
    pose_analysis_static
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
)

# 图像批量姿态分析示例
add_executable(pose_analysis_with_images pose_analysis_with_images.cpp)
target_link_libraries(pose_analysis_with_images 
    pose_analysis_static
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
)