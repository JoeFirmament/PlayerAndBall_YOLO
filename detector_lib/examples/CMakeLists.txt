# 示例程序CMakeLists.txt

# 示例程序源文件
set(EXAMPLE_SOURCES
    test_detector_lib.cpp
    pose_image.cpp
    pose_image_with_homography.cpp
    pose_image_with_polar.cpp
    rim_basketball_image.cpp
    pose_camera_bytetrack_homography.cpp
)

# 为每个示例创建可执行文件
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${example_src})
        # 目标名与文件名一致
        set(target_name "${example_name}")

        add_executable(${target_name} ${example_src})

        # 链接库
        target_link_libraries(${target_name}
            detector_lib_static  # 使用静态库
            ${OpenCV_LIBS}
            ${RKNN_RT_LIB}
            pthread
        )

        # 包含目录
        target_include_directories(${target_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../include
        )

        # 设置RPATH - 强制使用项目内库，避免系统库版本冲突
        set_target_properties(${target_name} PROPERTIES
            # 构建时和运行时都使用相同的RPATH
            BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
            INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../detector_lib/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH_USE_LINK_PATH FALSE
            # 跳过BUILD RPATH，直接使用INSTALL RPATH
            SKIP_BUILD_RPATH FALSE
        )

        message(STATUS "Created example: ${target_name}")
    endif()
endforeach()

# 安装示例程序 (可选)
option(INSTALL_EXAMPLES "Install example programs" OFF)
if(INSTALL_EXAMPLES)
    install(TARGETS ${EXAMPLE_TARGETS}
        RUNTIME DESTINATION bin/examples
    )
endif()