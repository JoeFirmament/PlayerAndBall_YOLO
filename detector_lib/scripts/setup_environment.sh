#!/bin/bash

# ========================================
# 环境设置和依赖安装脚本
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 系统检测
OS_TYPE=""
PKG_MANAGER=""

# 打印函数
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

# 检测操作系统
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS_TYPE="debian"
            PKG_MANAGER="apt"
        elif [ -f /etc/redhat-release ]; then
            OS_TYPE="redhat"
            PKG_MANAGER="yum"
        else
            OS_TYPE="linux"
            PKG_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        PKG_MANAGER="brew"
    else
        OS_TYPE="unknown"
        PKG_MANAGER="unknown"
    fi
    
    print_info "检测到系统: $OS_TYPE"
    print_info "包管理器: $PKG_MANAGER"
}

# 检查是否为RK3588
check_rk3588() {
    if [ -f /proc/device-tree/model ]; then
        local model=$(cat /proc/device-tree/model)
        if [[ $model == *"RK3588"* ]] || [[ $model == *"rk3588"* ]]; then
            print_success "检测到RK3588平台: $model"
            return 0
        fi
    fi
    
    print_warning "未检测到RK3588平台，某些功能可能受限"
    return 1
}

# 安装基础依赖
install_basic_deps() {
    print_info "安装基础编译依赖..."
    
    case $PKG_MANAGER in
        apt)
            sudo apt update
            sudo apt install -y \
                build-essential \
                cmake \
                git \
                pkg-config \
                wget \
                curl
            ;;
        yum)
            sudo yum install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                pkgconfig \
                wget \
                curl
            ;;
        brew)
            brew install \
                cmake \
                pkg-config \
                wget
            ;;
        *)
            print_error "不支持的包管理器: $PKG_MANAGER"
            return 1
            ;;
    esac
    
    print_success "基础依赖安装完成"
}

# 安装OpenCV
install_opencv() {
    print_info "检查OpenCV..."
    
    if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
        local version=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
        print_success "OpenCV已安装: $version"
        return 0
    fi
    
    print_info "安装OpenCV..."
    
    case $PKG_MANAGER in
        apt)
            sudo apt install -y \
                libopencv-dev \
                python3-opencv
            ;;
        yum)
            sudo yum install -y \
                opencv \
                opencv-devel
            ;;
        brew)
            brew install opencv
            ;;
    esac
    
    print_success "OpenCV安装完成"
}

# 安装JsonCpp
install_jsoncpp() {
    print_info "检查JsonCpp..."
    
    if pkg-config --exists jsoncpp 2>/dev/null; then
        local version=$(pkg-config --modversion jsoncpp 2>/dev/null)
        print_success "JsonCpp已安装: $version"
        return 0
    fi
    
    print_info "安装JsonCpp..."
    
    case $PKG_MANAGER in
        apt)
            sudo apt install -y libjsoncpp-dev
            ;;
        yum)
            sudo yum install -y jsoncpp-devel
            ;;
        brew)
            brew install jsoncpp
            ;;
    esac
    
    print_success "JsonCpp安装完成"
}

# 安装Eigen3
install_eigen() {
    print_info "检查Eigen3..."
    
    if pkg-config --exists eigen3 2>/dev/null; then
        local version=$(pkg-config --modversion eigen3 2>/dev/null)
        print_success "Eigen3已安装: $version"
        return 0
    fi
    
    print_info "安装Eigen3..."
    
    case $PKG_MANAGER in
        apt)
            sudo apt install -y libeigen3-dev
            ;;
        yum)
            sudo yum install -y eigen3-devel
            ;;
        brew)
            brew install eigen
            ;;
    esac
    
    print_success "Eigen3安装完成"
}

# 安装测试框架
install_gtest() {
    print_info "检查Google Test..."
    
    if pkg-config --exists gtest 2>/dev/null; then
        local version=$(pkg-config --modversion gtest 2>/dev/null)
        print_success "Google Test已安装: $version"
        return 0
    fi
    
    read -p "是否安装Google Test测试框架? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "跳过Google Test安装"
        return 0
    fi
    
    print_info "安装Google Test..."
    
    case $PKG_MANAGER in
        apt)
            sudo apt install -y libgtest-dev
            # Ubuntu需要编译gtest
            cd /usr/src/gtest
            sudo cmake CMakeLists.txt
            sudo make
            sudo cp lib/*.a /usr/lib
            ;;
        yum)
            sudo yum install -y gtest gtest-devel
            ;;
        brew)
            brew install googletest
            ;;
    esac
    
    print_success "Google Test安装完成"
}

# 安装开发工具
install_dev_tools() {
    print_info "安装开发工具..."
    
    local tools_to_install=""
    
    # 检查valgrind
    if ! command -v valgrind &> /dev/null && [[ "$OS_TYPE" != "macos" ]]; then
        print_info "安装valgrind内存检查工具..."
        tools_to_install="$tools_to_install valgrind"
    fi
    
    # 检查lcov
    if ! command -v lcov &> /dev/null; then
        print_info "安装lcov代码覆盖率工具..."
        tools_to_install="$tools_to_install lcov"
    fi
    
    # 检查clang-format
    if ! command -v clang-format &> /dev/null; then
        print_info "安装clang-format代码格式化工具..."
        tools_to_install="$tools_to_install clang-format"
    fi
    
    if [ -n "$tools_to_install" ]; then
        case $PKG_MANAGER in
            apt)
                sudo apt install -y $tools_to_install
                ;;
            yum)
                sudo yum install -y $tools_to_install
                ;;
            brew)
                brew install $tools_to_install
                ;;
        esac
    fi
    
    print_success "开发工具安装完成"
}

# 设置RK3588 NPU权限
setup_npu_permissions() {
    if ! check_rk3588; then
        return 0
    fi
    
    print_info "设置NPU设备权限..."
    
    # NPU设备权限
    if [ -e /dev/dri/renderD128 ] || [ -e /dev/dri/renderD129 ]; then
        sudo chmod 666 /dev/dri/renderD*
        print_success "NPU设备权限设置完成"
    else
        print_warning "未找到NPU设备"
    fi
    
    # 添加用户到video组
    if groups $USER | grep -q video; then
        print_success "用户已在video组"
    else
        sudo usermod -a -G video $USER
        print_warning "已将用户添加到video组，请重新登录生效"
    fi
}

# 检查RKNN环境
check_rknn() {
    print_info "检查RKNN环境..."
    
    # 检查RKNN库
    if [ -f /usr/lib/librknnrt.so ] || [ -f /lib/librknnrt.so ]; then
        local rknn_size=$(ls -la /usr/lib/librknnrt.so 2>/dev/null || ls -la /lib/librknnrt.so 2>/dev/null | awk '{print $5}')
        if [ "$rknn_size" -gt 5000000 ]; then
            print_success "RKNN Runtime库已安装 (新版本)"
        else
            print_warning "RKNN Runtime库版本可能过旧"
        fi
    else
        print_warning "未检测到RKNN Runtime库"
    fi
    
    # 检查项目内置的RKNN库
    local project_root="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
    if [ -f "$project_root/lib/librknnrt.so" ]; then
        print_success "项目内置RKNN库可用"
    fi
}

# 创建环境配置文件
create_env_config() {
    print_info "创建环境配置文件..."
    
    local project_root="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
    local env_file="$project_root/.env"
    
    cat > "$env_file" << EOF
# 姿态分析系统环境配置
# 生成时间: $(date '+%Y-%m-%d %H:%M:%S')

# 系统信息
OS_TYPE=$OS_TYPE
PKG_MANAGER=$PKG_MANAGER
IS_RK3588=$(check_rk3588 && echo "true" || echo "false")

# 路径配置
PROJECT_ROOT=$project_root
BUILD_DIR=$project_root/build_pose_analysis

# 编译选项
CMAKE_BUILD_TYPE=Release
ENABLE_TESTS=true
ENABLE_EXAMPLES=true

# 库路径
export LD_LIBRARY_PATH=$project_root/lib:\$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$project_root/lib/pkgconfig:\$PKG_CONFIG_PATH

# RKNN配置
export RKNN_RUNTIME_LIB=$project_root/lib/librknnrt.so

# 开发工具
HAS_VALGRIND=$(command -v valgrind &> /dev/null && echo "true" || echo "false")
HAS_LCOV=$(command -v lcov &> /dev/null && echo "true" || echo "false")
HAS_GTEST=$(pkg-config --exists gtest 2>/dev/null && echo "true" || echo "false")
EOF
    
    print_success "环境配置文件创建: $env_file"
    print_info "使用方法: source $env_file"
}

# 验证环境
verify_environment() {
    print_info "验证环境配置..."
    
    local all_good=true
    
    # 检查编译器
    if command -v g++ &> /dev/null; then
        print_success "✓ G++编译器"
    else
        print_error "✗ G++编译器"
        all_good=false
    fi
    
    # 检查CMake
    if command -v cmake &> /dev/null; then
        print_success "✓ CMake构建工具"
    else
        print_error "✗ CMake构建工具"
        all_good=false
    fi
    
    # 检查OpenCV
    if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
        print_success "✓ OpenCV库"
    else
        print_error "✗ OpenCV库"
        all_good=false
    fi
    
    # 检查JsonCpp
    if pkg-config --exists jsoncpp 2>/dev/null; then
        print_success "✓ JsonCpp库"
    else
        print_warning "⚠ JsonCpp库 (可选)"
    fi
    
    # 检查测试框架
    if pkg-config --exists gtest 2>/dev/null; then
        print_success "✓ Google Test"
    else
        print_warning "⚠ Google Test (可选)"
    fi
    
    if [ "$all_good" = true ]; then
        print_success "环境验证通过！"
        return 0
    else
        print_error "环境配置不完整"
        return 1
    fi
}

# 显示帮助
show_help() {
    cat << EOF
环境设置脚本

使用方法:
    $0 [选项]

选项:
    all         安装所有依赖（默认）
    basic       只安装基础依赖
    opencv      只安装OpenCV
    jsoncpp     只安装JsonCpp
    test        只安装测试框架
    npu         设置NPU权限（RK3588）
    verify      验证环境配置
    help        显示帮助

示例:
    $0          # 安装所有依赖
    $0 basic    # 只安装基础工具
    $0 verify   # 验证环境

EOF
}

# 主函数
main() {
    echo "========================================"
    echo "     姿态分析系统 - 环境设置脚本"
    echo "========================================"
    echo ""
    
    detect_os
    
    case "${1:-all}" in
        all)
            install_basic_deps
            install_opencv
            install_jsoncpp
            install_eigen
            install_gtest
            install_dev_tools
            setup_npu_permissions
            check_rknn
            create_env_config
            verify_environment
            ;;
        basic)
            install_basic_deps
            ;;
        opencv)
            install_opencv
            ;;
        jsoncpp)
            install_jsoncpp
            ;;
        test)
            install_gtest
            ;;
        npu)
            setup_npu_permissions
            check_rknn
            ;;
        verify)
            verify_environment
            ;;
        help)
            show_help
            ;;
        *)
            print_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    print_info "环境设置完成！"
    print_info "下一步: 运行 ./scripts/build_pose_analysis.sh 编译项目"
}

# 运行主函数
main "$@"