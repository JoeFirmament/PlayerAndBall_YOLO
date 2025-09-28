# 测试程序

# 身高检测器测试
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_height_detector.cpp)
    add_executable(test_height_detector test_height_detector.cpp)
    target_link_libraries(test_height_detector 
        pose_analysis_static
        ${DETECTOR_LIB}
        ${OpenCV_LIBS}
    )
endif()

# 姿态分析器集成测试
add_executable(test_pose_analyzer ../examples/simple_pose_analysis_example.cpp)
target_link_libraries(test_pose_analyzer 
    pose_analysis_static
    ${DETECTOR_LIB}
    ${OpenCV_LIBS}
)