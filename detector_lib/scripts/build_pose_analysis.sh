#!/bin/bash

# ========================================
# 姿态分析系统编译脚本
# ========================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build_pose_analysis"
BUILD_TYPE="Release"
ENABLE_TESTS=true
ENABLE_EXAMPLES=true
VERBOSE=false
CLEAN_BUILD=false
JOBS=$(nproc)

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
姿态分析系统编译脚本

使用方法:
    $0 [选项]

选项:
    -h, --help          显示帮助信息
    -d, --debug         Debug编译模式（默认Release）
    -c, --clean         清理后重新编译
    -t, --no-tests      不编译测试
    -e, --no-examples   不编译示例
    -j, --jobs N        并行编译任务数（默认$(nproc)）
    -v, --verbose       显示详细编译信息
    --asan              启用Address Sanitizer
    --tsan              启用Thread Sanitizer

示例:
    $0                  # 默认Release编译
    $0 -d -t            # Debug编译，包含测试
    $0 -c -j 4          # 清理后用4个线程编译
    $0 --asan           # 启用内存检查

EOF
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--debug)
                BUILD_TYPE="Debug"
                shift
                ;;
            -c|--clean)
                CLEAN_BUILD=true
                shift
                ;;
            -t|--no-tests)
                ENABLE_TESTS=false
                shift
                ;;
            -e|--no-examples)
                ENABLE_EXAMPLES=false
                shift
                ;;
            -j|--jobs)
                JOBS="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --asan)
                SANITIZER="address"
                shift
                ;;
            --tsan)
                SANITIZER="thread"
                shift
                ;;
            *)
                print_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查依赖
check_dependencies() {
    print_info "检查编译依赖..."
    
    local deps_missing=false
    
    # 检查必要的工具
    for tool in cmake g++ pkg-config; do
        if ! command -v $tool &> /dev/null; then
            print_error "缺少工具: $tool"
            deps_missing=true
        fi
    done
    
    # 检查OpenCV
    if ! pkg-config --exists opencv4 2>/dev/null && ! pkg-config --exists opencv 2>/dev/null; then
        print_warning "未检测到OpenCV，可能需要手动指定路径"
    fi
    
    # 检查jsoncpp
    if ! pkg-config --exists jsoncpp 2>/dev/null; then
        print_warning "未检测到jsoncpp，将尝试使用系统路径"
    fi
    
    if [ "$deps_missing" = true ]; then
        print_error "缺少必要的依赖，请先安装"
        exit 1
    fi
    
    print_success "依赖检查通过"
}

# 创建CMake配置文件
create_cmake_config() {
    print_info "创建CMake配置文件..."
    
    cat > "${PROJECT_ROOT}/CMakeLists_pose_analysis.txt" << 'EOF'
cmake_minimum_required(VERSION 3.16)
project(pose_analysis_system)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${JSONCPP_INCLUDE_DIRS}
)

# 源文件
set(POSE_ANALYSIS_SOURCES
    src/height_detector.cpp
    src/ball_request_detector.cpp
    src/id_priority_manager.cpp
    src/pose_analyzer.cpp
)

# 创建静态库
add_library(pose_analysis_static STATIC ${POSE_ANALYSIS_SOURCES})
target_link_libraries(pose_analysis_static
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)

# 创建动态库
add_library(pose_analysis SHARED ${POSE_ANALYSIS_SOURCES})
target_link_libraries(pose_analysis
    ${OpenCV_LIBS}
    ${JSONCPP_LIBRARIES}
)

# 安装规则
install(TARGETS pose_analysis pose_analysis_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include/pose_analysis
    FILES_MATCHING PATTERN "*.h"
)

# 选项
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
EOF

    # 创建示例CMakeLists.txt
    if [ "$ENABLE_EXAMPLES" = true ]; then
        mkdir -p "${PROJECT_ROOT}/examples"
        cat > "${PROJECT_ROOT}/examples/CMakeLists.txt" << 'EOF'
# 示例程序
add_executable(pose_analysis_example pose_analysis_example.cpp)
target_link_libraries(pose_analysis_example pose_analysis_static pthread)

add_executable(yolov8_pose_with_analysis yolov8_pose_with_analysis.cpp)
target_link_libraries(yolov8_pose_with_analysis pose_analysis_static pthread)
EOF
    fi
}

# 清理构建目录
clean_build() {
    if [ -d "$BUILD_DIR" ]; then
        print_info "清理构建目录: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi
}

# 执行编译
do_build() {
    print_info "开始编译..."
    print_info "构建类型: $BUILD_TYPE"
    print_info "并行任务: $JOBS"
    
    # 创建构建目录
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    # CMake配置
    local cmake_args=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DBUILD_TESTS="$ENABLE_TESTS"
        -DBUILD_EXAMPLES="$ENABLE_EXAMPLES"
    )
    
    # 添加sanitizer选项
    if [ -n "$SANITIZER" ]; then
        cmake_args+=(-DENABLE_SANITIZER="$SANITIZER")
        print_info "启用Sanitizer: $SANITIZER"
    fi
    
    # 使用临时CMake文件
    if [ -f "${PROJECT_ROOT}/CMakeLists_pose_analysis.txt" ]; then
        cp "${PROJECT_ROOT}/CMakeLists_pose_analysis.txt" "${PROJECT_ROOT}/CMakeLists.txt.bak" 2>/dev/null || true
        mv "${PROJECT_ROOT}/CMakeLists_pose_analysis.txt" "${PROJECT_ROOT}/CMakeLists.txt"
    fi
    
    print_info "运行CMake配置..."
    if [ "$VERBOSE" = true ]; then
        cmake "${cmake_args[@]}" "$PROJECT_ROOT"
    else
        cmake "${cmake_args[@]}" "$PROJECT_ROOT" > /dev/null 2>&1
    fi
    
    print_info "开始编译..."
    if [ "$VERBOSE" = true ]; then
        make -j"$JOBS"
    else
        make -j"$JOBS" 2>&1 | while IFS= read -r line; do
            # 只显示重要信息
            if [[ $line == *"error:"* ]] || [[ $line == *"warning:"* ]]; then
                echo "$line"
            elif [[ $line == *"[100%]"* ]] || [[ $line == *"Built target"* ]]; then
                echo "$line"
            fi
        done
    fi
    
    # 恢复原CMakeLists.txt
    if [ -f "${PROJECT_ROOT}/CMakeLists.txt.bak" ]; then
        mv "${PROJECT_ROOT}/CMakeLists.txt.bak" "${PROJECT_ROOT}/CMakeLists.txt"
    fi
}

# 显示编译结果
show_results() {
    print_success "编译完成！"
    echo ""
    print_info "编译输出目录: $BUILD_DIR"
    echo ""
    
    # 列出生成的文件
    if [ -f "$BUILD_DIR/libpose_analysis.so" ]; then
        print_success "✓ 动态库: libpose_analysis.so"
    fi
    
    if [ -f "$BUILD_DIR/libpose_analysis_static.a" ]; then
        print_success "✓ 静态库: libpose_analysis_static.a"
    fi
    
    if [ "$ENABLE_TESTS" = true ] && [ -f "$BUILD_DIR/tests/pose_analysis_tests" ]; then
        print_success "✓ 测试程序: tests/pose_analysis_tests"
    fi
    
    if [ "$ENABLE_EXAMPLES" = true ]; then
        if [ -f "$BUILD_DIR/examples/pose_analysis_example" ]; then
            print_success "✓ 示例程序: examples/pose_analysis_example"
        fi
    fi
    
    echo ""
    print_info "下一步操作："
    echo "  运行测试:   $SCRIPT_DIR/run_tests.sh"
    echo "  运行示例:   $BUILD_DIR/examples/pose_analysis_example"
    echo "  安装库:     sudo make -C $BUILD_DIR install"
}

# 主函数
main() {
    echo "========================================"
    echo "        姿态分析系统编译脚本"
    echo "========================================"
    echo ""
    
    # 解析参数
    parse_args "$@"
    
    # 检查依赖
    check_dependencies
    
    # 清理构建（如果需要）
    if [ "$CLEAN_BUILD" = true ]; then
        clean_build
    fi
    
    # 创建CMake配置
    create_cmake_config
    
    # 执行编译
    do_build
    
    # 显示结果
    show_results
}

# 运行主函数
main "$@"