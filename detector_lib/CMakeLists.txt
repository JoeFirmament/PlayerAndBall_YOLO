cmake_minimum_required(VERSION 3.10)
project(DetectorLib VERSION 1.0.3 LANGUAGES C CXX)

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

# =============================================================================
# 依赖库查找
# =============================================================================

# 查找OpenCV
find_package(OpenCV 4.0 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 设置项目根目录
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# 智能查找RKNN头文件路径
set(RKNN_INCLUDE_DIR)
set(RKNN_LIB_DIR)

# 优先使用本地包含的头文件，然后查找外部路径
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/rknn_api.h)
    set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "✓ Using bundled RKNN headers: ${RKNN_INCLUDE_DIR}")
else()
    # 尝试外部路径
    set(POSSIBLE_RKNN_PATHS
        ${PROJECT_ROOT_DIR}/3rdparty/rknpu2/include
        /usr/include/rknn
        /usr/local/include/rknn
        ${PROJECT_ROOT_DIR}/../3rdparty/rknpu2/include
        ${PROJECT_ROOT_DIR}/rknpu2/include
    )
    
    foreach(path ${POSSIBLE_RKNN_PATHS})
        if(EXISTS ${path}/rknn_api.h)
            set(RKNN_INCLUDE_DIR ${path})
            message(STATUS "Found RKNN headers at: ${RKNN_INCLUDE_DIR}")
            break()
        endif()
    endforeach()
    
    # 如果没找到RKNN头文件，给出详细错误信息
    if(NOT RKNN_INCLUDE_DIR)
        message(STATUS "RKNN header search paths tried:")
        foreach(path ${POSSIBLE_RKNN_PATHS})
            message(STATUS "  - ${path}")
        endforeach()
        message(FATAL_ERROR "RKNN header (rknn_api.h) not found! Please ensure RKNN SDK is properly installed.")
    endif()
endif()

# 查找RKNN库文件路径
set(POSSIBLE_RKNN_LIB_PATHS
    # 优先查找本地libs目录
    ${CMAKE_CURRENT_SOURCE_DIR}/libs
    
    # 然后查找外部路径
    ${PROJECT_ROOT_DIR}/libs
    ${PROJECT_ROOT_DIR}/3rdparty/rknpu2/Linux/aarch64
    
    # 根据头文件路径推断
    ${RKNN_INCLUDE_DIR}/../lib
    ${RKNN_INCLUDE_DIR}/../libs
    ${RKNN_INCLUDE_DIR}/../Linux/aarch64
)

foreach(path ${POSSIBLE_RKNN_LIB_PATHS})
    if(EXISTS ${path})
        set(RKNN_LIB_DIR ${path})
        message(STATUS "RKNN lib directory: ${RKNN_LIB_DIR}")
        break()
    endif()
endforeach()

# RGA头文件路径 (可选，不强制要求)
set(RGA_INCLUDE_DIR)
set(POSSIBLE_RGA_PATHS
    ${PROJECT_ROOT_DIR}/3rdparty/mpp/Linux/aarch64/include
    ${PROJECT_ROOT_DIR}/3rdparty/rga/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga/include
    /usr/include/rga
)

foreach(path ${POSSIBLE_RGA_PATHS})
    if(EXISTS ${path})
        set(RGA_INCLUDE_DIR ${path})
        message(STATUS "Found RGA headers at: ${RGA_INCLUDE_DIR}")
        break()
    endif()
endforeach()

if(NOT RGA_INCLUDE_DIR)
    message(STATUS "RGA headers not found (optional)")
endif()

# 简化项目头文件路径 (优先使用本地)
set(PROJECT_INCLUDE_DIRS)

# 本地头文件路径
set(LOCAL_INCLUDE_PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

foreach(path ${LOCAL_INCLUDE_PATHS})
    if(EXISTS ${path})
        list(APPEND PROJECT_INCLUDE_DIRS ${path})
        message(STATUS "✓ Using local headers: ${path}")
    endif()
endforeach()

# 可选：查找外部项目头文件（向后兼容）
if(EXISTS ${PROJECT_ROOT_DIR}/utils)
    list(APPEND PROJECT_INCLUDE_DIRS ${PROJECT_ROOT_DIR}/utils)
    message(STATUS "Found external utils headers: ${PROJECT_ROOT_DIR}/utils")
endif()

if(EXISTS ${PROJECT_ROOT_DIR}/src)  
    list(APPEND PROJECT_INCLUDE_DIRS ${PROJECT_ROOT_DIR}/src)
    message(STATUS "Found external src headers: ${PROJECT_ROOT_DIR}/src")
endif()

# 检查关键头文件是否可用
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/common.h)
    message(STATUS "✓ Using bundled common.h")
elseif(EXISTS ${PROJECT_ROOT_DIR}/utils/common.h)
    message(STATUS "Found external common.h")
else()
    message(STATUS "common.h may be included via other headers")
endif()

# =============================================================================
# 包含目录
# =============================================================================

include_directories(
    # 当前库的头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    
    # OpenCV
    ${OpenCV_INCLUDE_DIRS}
    
    # RKNN
    ${RKNN_INCLUDE_DIR}
    
    # 项目内部头文件
    ${PROJECT_INCLUDE_DIRS}
    
    # 系统头文件
    /usr/include/eigen3
)

# 添加RGA头文件路径 (如果存在)
if(RGA_INCLUDE_DIR)
    include_directories(${RGA_INCLUDE_DIR})
endif()

# =============================================================================
# 源文件收集
# =============================================================================

# 库源文件
set(LIB_SOURCES
    src/detector_lib.cpp
    src/PoseDetectorLib.cpp
    src/RimBasketballDetectorLib.cpp
    src/internal/detector_common.cpp
    src/detector_path_utils.cpp
)

# 依赖的项目源文件 (使用本地复制的文件)
set(PROJECT_SOURCES
    # 本地复制的必要源文件
    ${CMAKE_CURRENT_SOURCE_DIR}/src/detector_file_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rim_basketball_postprocess_simple.cpp
    
    # ByteTrack多目标跟踪支持（暂时禁用，等待完整实现）
    # ${PROJECT_ROOT_DIR}/src/BYTETracker.cpp
    # ${PROJECT_ROOT_DIR}/src/STrack.cpp
    # ${PROJECT_ROOT_DIR}/src/kalmanFilter.cpp
    # ${PROJECT_ROOT_DIR}/src/lapjv.cpp
    
    # 注意: 暂时不包含RGA相关文件
    # ${PROJECT_ROOT_DIR}/utils/image_utils.c
    # ${PROJECT_ROOT_DIR}/src/pose_yolov8.cc
    # ${PROJECT_ROOT_DIR}/src/pose_postprocess.cc
    # ${PROJECT_ROOT_DIR}/src/pose_letterbox_utils.cc
)

# 过滤存在的源文件
set(FILTERED_PROJECT_SOURCES)
foreach(src_file ${PROJECT_SOURCES})
    if(EXISTS ${src_file})
        list(APPEND FILTERED_PROJECT_SOURCES ${src_file})
        message(STATUS "Including source: ${src_file}")
    else()
        message(WARNING "Source file not found: ${src_file}")
    endif()
endforeach()

# =============================================================================
# 库目标
# =============================================================================

# 静态库
add_library(detector_lib_static STATIC 
    ${LIB_SOURCES}
    ${FILTERED_PROJECT_SOURCES}
)

# 动态库
add_library(detector_lib_shared SHARED 
    ${LIB_SOURCES}
    ${FILTERED_PROJECT_SOURCES}
)

# 设置库名称
set_target_properties(detector_lib_static PROPERTIES OUTPUT_NAME detector_lib)
set_target_properties(detector_lib_shared PROPERTIES OUTPUT_NAME detector_lib)

# 设置库版本（将在RPATH设置中一并配置）

# =============================================================================
# 链接库
# =============================================================================

# 查找RKNN库 - 优先使用项目自带的库
find_library(RKNN_RT_LIB 
    NAMES rknnrt
    PATHS 
        ${CMAKE_CURRENT_SOURCE_DIR}/lib
        ${CMAKE_CURRENT_SOURCE_DIR}/libs
        ${PROJECT_ROOT_DIR}/libs
    NO_DEFAULT_PATH
)

if(NOT RKNN_RT_LIB)
    # 备选方案：使用3rdparty目录下的库
    find_library(RKNN_RT_LIB 
        NAMES rknnrt
        PATHS ${RKNN_LIB_DIR}
        NO_DEFAULT_PATH
    )
endif()

if(NOT RKNN_RT_LIB)
    message(FATAL_ERROR "RKNN runtime library not found! Please ensure librknnrt.so exists in one of the search paths.")
endif()

# 链接库
target_link_libraries(detector_lib_static
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    pthread
)

target_link_libraries(detector_lib_shared
    ${OpenCV_LIBS}
    ${RKNN_RT_LIB}
    pthread
)

# =============================================================================
# 编译选项
# =============================================================================

# 设置编译定义
target_compile_definitions(detector_lib_static PRIVATE
    DETECTOR_LIB_VERSION="${PROJECT_VERSION}"
)

target_compile_definitions(detector_lib_shared PRIVATE
    DETECTOR_LIB_VERSION="${PROJECT_VERSION}"
    DETECTOR_LIB_SHARED
)

# =============================================================================
# RPATH设置 - 支持相对路径机制
# =============================================================================

# 设置动态库的RPATH，支持相对路径查找
set_target_properties(detector_lib_shared PROPERTIES
    # 运行时库搜索路径：从可执行文件位置查找相对路径
    INSTALL_RPATH "$ORIGIN/../lib"
    
    # 构建时也使用安装时的RPATH设置
    BUILD_WITH_INSTALL_RPATH TRUE
    
    # 不使用链接时的路径作为RPATH
    INSTALL_RPATH_USE_LINK_PATH FALSE
    
    # 版本信息
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

# =============================================================================
# 示例程序 (可选)
# =============================================================================

option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# =============================================================================
# 安装规则
# =============================================================================

# 安装库文件
install(TARGETS detector_lib_static detector_lib_shared
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# 安装头文件
install(FILES
    include/detector_lib.h
    include/detector_types.h
    include/PoseDetectorLib.h
    include/RimBasketballDetectorLib.h
    include/detector_common_types.h
    include/detector_file_utils.h
    include/detector_path_utils.h
    include/detector_rim_basketball_postprocess.h
    include/rknn_api.h
    DESTINATION include/detector_lib
)

# 安装pkg-config文件
configure_file(detector_lib.pc.in detector_lib.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/detector_lib.pc
    DESTINATION lib/pkgconfig
)

# =============================================================================
# 总结信息
# =============================================================================

message(STATUS "=== DetectorLib Configuration Summary ===")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN library: ${RKNN_RT_LIB}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "==========================================")

# 显示找到的源文件数量
list(LENGTH LIB_SOURCES lib_src_count)
list(LENGTH FILTERED_PROJECT_SOURCES proj_src_count)
message(STATUS "Library sources: ${lib_src_count}")
message(STATUS "Project sources: ${proj_src_count}")