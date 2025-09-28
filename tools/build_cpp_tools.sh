#!/bin/bash

# C++相机标定工具编译脚本
# 用于快速构建和测试C++工具

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "Checking dependencies..."

    # 检查CMake
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake 3.10 or higher."
        exit 1
    fi

    # 检查编译器
    if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
        log_error "No C++ compiler found. Please install g++ or clang++."
        exit 1
    fi

    # 检查pkg-config
    if ! command -v pkg-config &> /dev/null; then
        log_warning "pkg-config not found. OpenCV detection may fail."
    fi

    # 检查OpenCV
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv4)
        log_success "OpenCV 4.x found: $OPENCV_VERSION"
    elif pkg-config --exists opencv 2>/dev/null; then
        OPENCV_VERSION=$(pkg-config --modversion opencv)
        log_warning "OpenCV found but not version 4.x: $OPENCV_VERSION"
        log_warning "Recommended to use OpenCV 4.x for best compatibility."
    else
        log_error "OpenCV not found. Please install OpenCV 4.x."
        log_info "Installation commands:"
        log_info "  Ubuntu/Debian: sudo apt install libopencv-dev"
        log_info "  CentOS/RHEL: sudo yum install opencv-devel"
        log_info "  macOS: brew install opencv"
        exit 1
    fi

    # 检查Eigen (可选)
    if pkg-config --exists eigen3 2>/dev/null; then
        EIGEN_VERSION=$(pkg-config --modversion eigen3)
        log_success "Eigen3 found: $EIGEN_VERSION"
    else
        log_warning "Eigen3 not found. Advanced matrix operations will be limited."
        log_info "Optional installation: sudo apt install libeigen3-dev"
    fi

    log_success "Dependency check completed."
}

# 创建构建目录
setup_build_directory() {
    log_info "Setting up build directory..."

    if [ -d "build" ]; then
        log_warning "Build directory already exists. Cleaning..."
        rm -rf build/*
    else
        mkdir -p build
    fi

    cd build
    log_success "Build directory ready."
}

# 配置项目
configure_project() {
    log_info "Configuring project with CMake..."

    local cmake_args="-DCMAKE_BUILD_TYPE=Release"

    # 添加调试版本选项
    if [ "$1" = "debug" ]; then
        cmake_args="-DCMAKE_BUILD_TYPE=Debug"
        log_info "Building in Debug mode"
    fi

    # 运行CMake
    if cmake .. $cmake_args; then
        log_success "CMake configuration completed."
    else
        log_error "CMake configuration failed."
        exit 1
    fi
}

# 编译项目
build_project() {
    log_info "Building project..."

    local make_args="-j$(nproc 2>/dev/null || echo 4)"

    if make $make_args; then
        log_success "Build completed successfully."
    else
        log_error "Build failed."
        exit 1
    fi
}

# 运行测试
run_tests() {
    log_info "Running tests..."

    if [ -f "./calibration_example" ]; then
        # 检查是否有测试文件
        if [ -f "../camera_calibration.npz" ] && [ -f "../ground_calibration.npz" ]; then
            log_info "Found calibration files. Running example..."
            ./calibration_example ../camera_calibration.npz ../ground_calibration.npz
        else
            log_warning "Calibration files not found. Skipping full test."
            log_info "To test with real data, ensure camera_calibration.npz and ground_calibration.npz exist."
        fi
    else
        log_error "Executable not found. Build may have failed."
        exit 1
    fi
}

# 显示帮助信息
show_help() {
    echo "C++ Camera Calibration Tools Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --debug, -d         Build in debug mode"
    echo "  --test, -t          Run tests after building"
    echo "  --clean             Clean build directory"
    echo "  --install           Install to system (requires sudo)"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build in release mode"
    echo "  $0 --debug          # Build in debug mode"
    echo "  $0 --test           # Build and run tests"
    echo "  $0 --clean          # Clean build directory"
    echo "  $0 --install        # Build and install to system"
}

# 清理构建目录
clean_build() {
    log_info "Cleaning build directory..."
    if [ -d "build" ]; then
        rm -rf build
        log_success "Build directory cleaned."
    else
        log_info "Build directory does not exist."
    fi
}

# 安装到系统
install_system() {
    log_info "Installing to system..."

    if [ -f "build/calibration_example" ]; then
        cd build
        if sudo make install; then
            log_success "Installation completed."
        else
            log_error "Installation failed."
            exit 1
        fi
    else
        log_error "Build calibration_example first."
        exit 1
    fi
}

# 主函数
main() {
    local debug_mode=false
    local run_test=false
    local do_install=false
    local do_clean=false

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                show_help
                exit 0
                ;;
            --debug|-d)
                debug_mode=true
                shift
                ;;
            --test|-t)
                run_test=true
                shift
                ;;
            --clean)
                do_clean=true
                shift
                ;;
            --install)
                do_install=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # 执行清理
    if [ "$do_clean" = true ]; then
        clean_build
        exit 0
    fi

    # 检查依赖
    check_dependencies

    # 设置构建目录
    setup_build_directory

    # 配置项目
    if [ "$debug_mode" = true ]; then
        configure_project "debug"
    else
        configure_project
    fi

    # 编译项目
    build_project

    # 运行测试
    if [ "$run_test" = true ]; then
        run_tests
    fi

    # 安装到系统
    if [ "$do_install" = true ]; then
        install_system
    fi

    log_success "All operations completed successfully!"
    log_info "Executable location: $(pwd)/calibration_example"
    log_info "To use the tool: ./calibration_example <camera.npz> <ground.npz> [test_images...]"
}

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    log_error "CMakeLists.txt not found. Please run this script from the tools directory."
    exit 1
fi

# 运行主函数
main "$@"
