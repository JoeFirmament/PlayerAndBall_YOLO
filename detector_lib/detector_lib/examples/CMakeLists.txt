# 示例程序CMakeLists.txt

# 示例程序源文件
set(EXAMPLE_SOURCES
    test_detector_lib.cpp
    pose_image.cpp
    pose_image_with_homography.cpp
    pose_image_with_polar.cpp
    rim_basketball_image.cpp
)

# 为每个示例创建可执行文件
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_src})
        add_executable(${example_name} ${example_src})
        
        # 链接库
        target_link_libraries(${example_name}
            detector_lib_static  # 使用静态库
            ${OpenCV_LIBS}
            ${RKNN_RT_LIB}
            pthread
        )
        
        # 包含目录
        target_include_directories(${example_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )
        
        # 设置RPATH - 从示例程序的bin/目录查找../lib/中的库
        set_target_properties(${example_name} PROPERTIES
            INSTALL_RPATH "$ORIGIN/../lib"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH FALSE
        )
        
        message(STATUS "Created example: ${example_name}")
    endif()
endforeach()

# 安装示例程序 (可选)
option(INSTALL_EXAMPLES "Install example programs" OFF)
if(INSTALL_EXAMPLES)
    install(TARGETS ${EXAMPLE_TARGETS}
        RUNTIME DESTINATION bin/examples
    )
endif()