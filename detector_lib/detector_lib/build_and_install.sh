#!/bin/bash

# DetectorLib 构建和安装脚本
# 用途: 一键编译和安装检测器库

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 输出函数
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

# 显示帮助信息
show_help() {
    echo "DetectorLib 构建脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -c, --clean         清理构建目录"
    echo "  -d, --debug         Debug模式构建"
    echo "  -r, --release       Release模式构建 (默认)"
    echo "  -i, --install       安装到系统 (需要root权限)"
    echo "  -t, --test          编译后运行测试"
    echo "  --no-examples       不编译示例程序"
    echo ""
    echo "示例:"
    echo "  $0                  # 默认Release构建"
    echo "  $0 -d -t           # Debug构建并测试"
    echo "  $0 -r -i           # Release构建并安装"
}

# 默认参数
BUILD_TYPE="Release"
CLEAN_BUILD=false
INSTALL_LIB=false
RUN_TEST=false
BUILD_EXAMPLES=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -r|--release)
            BUILD_TYPE="Release"
            shift
            ;;
        -i|--install)
            INSTALL_LIB=true
            shift
            ;;
        -t|--test)
            RUN_TEST=true
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES=false
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 显示构建配置
log_info "=== DetectorLib 构建配置 ==="
log_info "构建类型: $BUILD_TYPE"
log_info "清理构建: $CLEAN_BUILD"
log_info "安装库: $INSTALL_LIB"
log_info "运行测试: $RUN_TEST"
log_info "构建示例: $BUILD_EXAMPLES"
log_info "=========================="

# 检查当前目录
if [ ! -f "CMakeLists.txt" ]; then
    log_error "未找到CMakeLists.txt，请在detector_lib目录下运行此脚本"
    exit 1
fi

# 检查依赖
log_info "检查构建依赖..."

# 检查CMake
if ! command -v cmake &> /dev/null; then
    log_error "CMake未安装，请安装: sudo apt install cmake"
    exit 1
fi

# 检查OpenCV
if ! pkg-config --exists opencv4; then
    log_warning "OpenCV4未找到，尝试检查opencv..."
    if ! pkg-config --exists opencv; then
        log_error "OpenCV未安装，请安装: sudo apt install libopencv-dev"
        exit 1
    fi
fi

# 检查编译器
if ! command -v g++ &> /dev/null; then
    log_error "g++未安装，请安装: sudo apt install build-essential"
    exit 1
fi

log_success "依赖检查完成"

# 清理构建目录
if [ "$CLEAN_BUILD" = true ]; then
    log_info "清理构建目录..."
    rm -rf build
    log_success "构建目录已清理"
fi

# 创建构建目录
if [ ! -d "build" ]; then
    log_info "创建构建目录..."
    mkdir build
fi

cd build

# 配置CMake
log_info "配置CMake..."
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    -DBUILD_EXAMPLES=$BUILD_EXAMPLES
)

if [ "$INSTALL_LIB" = true ]; then
    CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX=/usr/local)
fi

cmake "${CMAKE_ARGS[@]}" .. || {
    log_error "CMake配置失败"
    exit 1
}

log_success "CMake配置完成"

# 编译
log_info "开始编译..."
CPU_CORES=$(nproc)
log_info "使用 $CPU_CORES 个CPU核心并行编译"

make -j$CPU_CORES || {
    log_error "编译失败"
    exit 1
}

log_success "编译完成"

# 显示构建结果
log_info "构建结果:"
if [ -f "libdetector_lib.a" ]; then
    SIZE=$(du -h libdetector_lib.a | cut -f1)
    log_success "  静态库: libdetector_lib.a ($SIZE)"
fi

if [ -f "libdetector_lib.so" ]; then
    SIZE=$(du -h libdetector_lib.so | cut -f1)
    log_success "  动态库: libdetector_lib.so ($SIZE)"
fi

if [ -d "examples" ]; then
    EXAMPLE_COUNT=$(find examples -type f -executable | wc -l)
    log_success "  示例程序: $EXAMPLE_COUNT 个"
fi

# 运行测试
if [ "$RUN_TEST" = true ]; then
    log_info "运行测试程序..."
    
    if [ -f "examples/test_detector_lib" ]; then
        log_info "运行基础功能测试..."
        ./examples/test_detector_lib || {
            log_warning "测试程序运行失败 (可能缺少模型文件)"
        }
    else
        log_warning "测试程序未找到"
    fi
fi

# 安装库
if [ "$INSTALL_LIB" = true ]; then
    log_info "安装库文件..."
    
    if [ "$EUID" -ne 0 ]; then
        log_warning "需要root权限安装，尝试使用sudo..."
        sudo make install || {
            log_error "安装失败"
            exit 1
        }
    else
        make install || {
            log_error "安装失败"
            exit 1
        }
    fi
    
    log_success "库安装完成"
    
    # 更新库缓存
    log_info "更新系统库缓存..."
    sudo ldconfig
    
    # 显示安装路径
    log_info "安装路径:"
    log_info "  头文件: /usr/local/include/detector_lib/"
    log_info "  库文件: /usr/local/lib/"
    log_info "  pkg-config: /usr/local/lib/pkgconfig/"
fi

# 返回原目录
cd ..

# 显示使用说明
log_info "=== 使用说明 ==="

if [ "$INSTALL_LIB" = true ]; then
    log_info "库已安装到系统，您可以在项目中使用:"
    echo "    #include <detector_lib/detector_lib.h>"
    echo "    编译: g++ -o app app.cpp \`pkg-config --cflags --libs detector_lib\`"
else
    log_info "库未安装，您可以在当前项目中使用:"
    echo "    头文件: include/detector_lib.h"
    echo "    静态库: build/libdetector_lib.a"
    echo "    动态库: build/libdetector_lib.so"
fi

log_info ""
log_info "运行示例程序:"
log_info "✅ 推荐的图片测试程序（经过验证）:"
if [ -f "build/examples/pose_image" ]; then
    echo "    cd build/examples && ./pose_image                    # 基础姿态检测"
fi
if [ -f "build/examples/pose_image_with_homography" ]; then
    echo "    cd build/examples && ./pose_image_with_homography    # 姿态+Homography坐标映射（✅ 完整功能）"
fi
if [ -f "build/examples/rim_basketball_image" ]; then
    echo "    cd build/examples && ./rim_basketball_image          # 篮筐篮球检测"
fi

log_info ""
log_info "🔧 其他测试程序："  
if [ -f "build/examples/test_detector_lib" ]; then
    echo "    cd build/examples && ./test_detector_lib            # 综合功能测试"
fi

log_info ""
log_info "注意事项:"
log_info "1. 确保模型文件路径正确"
log_info "2. 确保NPU设备权限: sudo chmod 666 /dev/dri/renderD*"
log_info "3. 检查RKNN运行时库是否可用"

log_success "构建脚本执行完成!"