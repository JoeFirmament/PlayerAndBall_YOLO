#!/bin/bash

# ========================================
# 快速测试脚本 - 用于开发时的快速验证
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 打印函数
print_header() {
    echo ""
    echo -e "${MAGENTA}========================================${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}========================================${NC}"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# 检查文件是否存在
check_file() {
    if [ -f "$1" ]; then
        print_success "找到: $(basename $1)"
        return 0
    else
        print_error "缺失: $(basename $1)"
        return 1
    fi
}

# 1. 检查源文件完整性
check_source_files() {
    print_header "检查源文件完整性"
    
    local all_good=true
    
    # 检查头文件
    print_info "检查头文件..."
    local headers=(
        "include/pose_analysis_types.h"
        "include/temporal_buffer.h"
        "include/filter_interface.h"
        "include/height_detector.h"
        "include/ball_request_detector.h"
        "include/id_priority_manager.h"
        "include/pose_analyzer.h"
        "include/debug_visualizer.h"
    )
    
    for header in "${headers[@]}"; do
        check_file "$PROJECT_ROOT/$header" || all_good=false
    done
    
    # 检查源文件
    print_info "检查源文件..."
    local sources=(
        "src/height_detector.cpp"
        "src/ball_request_detector.cpp"
        "src/id_priority_manager.cpp"
        "src/pose_analyzer.cpp"
    )
    
    for source in "${sources[@]}"; do
        check_file "$PROJECT_ROOT/$source" || all_good=false
    done
    
    # 检查测试文件
    print_info "检查测试文件..."
    local tests=(
        "tests/test_temporal_buffer.cpp"
        "tests/test_filter_interface.cpp"
        "tests/test_height_detector.cpp"
    )
    
    for test in "${tests[@]}"; do
        check_file "$PROJECT_ROOT/$test" || all_good=false
    done
    
    if [ "$all_good" = true ]; then
        print_success "所有源文件完整"
    else
        print_error "部分源文件缺失"
        return 1
    fi
}

# 2. 编译单个测试文件
compile_single_test() {
    print_header "编译单个测试文件"
    
    local test_file="$1"
    local output_name="${2:-test_program}"
    
    print_info "编译 $test_file..."
    
    # 创建临时构建目录
    local temp_build="/tmp/pose_analysis_test_$$"
    mkdir -p "$temp_build"
    
    # 编译命令
    local compile_cmd="g++ -std=c++17 -O2 -g \
        -I$PROJECT_ROOT/include \
        $(pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv) \
        $(pkg-config --cflags jsoncpp 2>/dev/null || echo '') \
        $test_file \
        $PROJECT_ROOT/src/*.cpp \
        $(pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv) \
        $(pkg-config --libs jsoncpp 2>/dev/null || echo '-ljsoncpp') \
        -pthread \
        -o $temp_build/$output_name"
    
    if eval $compile_cmd 2>/dev/null; then
        print_success "编译成功: $temp_build/$output_name"
        return 0
    else
        print_error "编译失败"
        rm -rf "$temp_build"
        return 1
    fi
}

# 3. 运行语法检查
syntax_check() {
    print_header "语法检查"
    
    print_info "检查C++语法..."
    
    local errors_found=false
    
    # 检查所有cpp文件
    for file in $(find "$PROJECT_ROOT/src" -name "*.cpp"); do
        echo -n "  $(basename $file)... "
        if g++ -std=c++17 -fsyntax-only -I"$PROJECT_ROOT/include" "$file" 2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}ERROR${NC}"
            errors_found=true
        fi
    done
    
    if [ "$errors_found" = false ]; then
        print_success "语法检查通过"
    else
        print_error "发现语法错误"
        return 1
    fi
}

# 4. 检查依赖
check_dependencies() {
    print_header "检查编译依赖"
    
    local all_good=true
    
    # 检查编译器
    if command -v g++ &> /dev/null; then
        local gcc_version=$(g++ --version | head -n1)
        print_success "G++: $gcc_version"
    else
        print_error "G++: 未安装"
        all_good=false
    fi
    
    # 检查CMake
    if command -v cmake &> /dev/null; then
        local cmake_version=$(cmake --version | head -n1)
        print_success "CMake: $cmake_version"
    else
        print_error "CMake: 未安装"
        all_good=false
    fi
    
    # 检查OpenCV
    if pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null; then
        local opencv_version=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null)
        print_success "OpenCV: $opencv_version"
    else
        print_warning "OpenCV: 未通过pkg-config检测"
    fi
    
    # 检查JsonCpp
    if pkg-config --exists jsoncpp 2>/dev/null; then
        local jsoncpp_version=$(pkg-config --modversion jsoncpp 2>/dev/null)
        print_success "JsonCpp: $jsoncpp_version"
    else
        print_warning "JsonCpp: 未通过pkg-config检测"
    fi
    
    # 检查GTest（可选）
    if pkg-config --exists gtest 2>/dev/null; then
        local gtest_version=$(pkg-config --modversion gtest 2>/dev/null)
        print_success "GTest: $gtest_version"
    else
        print_warning "GTest: 未安装（测试可选）"
    fi
    
    if [ "$all_good" = true ]; then
        print_success "所有必要依赖已满足"
    else
        print_error "缺少必要依赖"
        return 1
    fi
}

# 5. 创建简单的测试程序
create_simple_test() {
    print_header "创建简单测试程序"
    
    local test_file="/tmp/simple_pose_test.cpp"
    
    cat > "$test_file" << 'EOF'
#include <iostream>
#include <vector>
#include "pose_analysis_types.h"
#include "temporal_buffer.h"
#include "filter_interface.h"

using namespace pose_analysis;

int main() {
    std::cout << "=== 姿态分析库简单测试 ===" << std::endl;
    
    // 测试1: 时序缓冲区
    std::cout << "\n1. 测试时序缓冲区..." << std::endl;
    TemporalBuffer<float> buffer(10);
    buffer.push(1.0f);
    buffer.push(2.0f);
    buffer.push(3.0f);
    std::cout << "   缓冲区大小: " << buffer.size() << std::endl;
    std::cout << "   最新值: " << buffer.get_latest() << std::endl;
    
    // 测试2: 滤波器
    std::cout << "\n2. 测试中值滤波器..." << std::endl;
    MedianFilter filter(5);
    std::vector<float> values = {1.0f, 5.0f, 2.0f, 8.0f, 3.0f};
    float result = 0.0f;
    for (float v : values) {
        result = filter.process(v);
    }
    std::cout << "   滤波结果: " << result << std::endl;
    
    // 测试3: 数据结构
    std::cout << "\n3. 测试数据结构..." << std::endl;
    PoseResult pose;
    pose.person_id = 1;
    pose.detection_confidence = 0.85f;
    std::cout << "   PoseResult创建成功" << std::endl;
    
    std::cout << "\n✓ 所有基础测试通过！" << std::endl;
    return 0;
}
EOF
    
    print_info "编译测试程序..."
    if compile_single_test "$test_file" "simple_test"; then
        print_info "运行测试程序..."
        if /tmp/pose_analysis_test_$$/simple_test; then
            print_success "简单测试通过"
            rm -rf /tmp/pose_analysis_test_$$
            rm -f "$test_file"
            return 0
        else
            print_error "测试运行失败"
            rm -rf /tmp/pose_analysis_test_$$
            rm -f "$test_file"
            return 1
        fi
    else
        print_error "测试编译失败"
        rm -f "$test_file"
        return 1
    fi
}

# 6. 检查代码规范
check_code_style() {
    print_header "代码规范检查"
    
    print_info "检查代码格式..."
    
    # 统计代码行数
    local total_lines=$(find "$PROJECT_ROOT/src" "$PROJECT_ROOT/include" -name "*.cpp" -o -name "*.h" | xargs wc -l | tail -n1 | awk '{print $1}')
    print_info "总代码行数: $total_lines"
    
    # 检查TODO和FIXME
    local todos=$(grep -r "TODO\|FIXME" "$PROJECT_ROOT/src" "$PROJECT_ROOT/include" 2>/dev/null | wc -l)
    if [ "$todos" -gt 0 ]; then
        print_warning "发现 $todos 个 TODO/FIXME 标记"
    else
        print_success "没有TODO/FIXME标记"
    fi
    
    # 检查包含保护
    local missing_guards=0
    for header in $(find "$PROJECT_ROOT/include" -name "*.h"); do
        if ! grep -q "#pragma once\|#ifndef.*#define" "$header"; then
            print_warning "缺少包含保护: $(basename $header)"
            ((missing_guards++))
        fi
    done
    
    if [ "$missing_guards" -eq 0 ]; then
        print_success "所有头文件都有包含保护"
    fi
}

# 7. 生成快速报告
generate_report() {
    print_header "生成测试报告"
    
    local report_file="$PROJECT_ROOT/quick_test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "姿态分析系统快速测试报告"
        echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================="
        echo ""
        echo "文件统计:"
        echo "  头文件: $(find "$PROJECT_ROOT/include" -name "*.h" | wc -l)"
        echo "  源文件: $(find "$PROJECT_ROOT/src" -name "*.cpp" | wc -l)"
        echo "  测试文件: $(find "$PROJECT_ROOT/tests" -name "*.cpp" | wc -l)"
        echo ""
        echo "代码统计:"
        echo "  总行数: $(find "$PROJECT_ROOT" -name "*.cpp" -o -name "*.h" | xargs wc -l | tail -n1 | awk '{print $1}')"
        echo ""
        echo "模块列表:"
        echo "  - 时序缓冲区 (temporal_buffer.h)"
        echo "  - 滤波器接口 (filter_interface.h)"
        echo "  - 身高检测 (height_detector.h/cpp)"
        echo "  - 要球检测 (ball_request_detector.h/cpp)"
        echo "  - ID管理 (id_priority_manager.h/cpp)"
        echo "  - 集成分析器 (pose_analyzer.h/cpp)"
        echo ""
        echo "测试状态: 待测试"
    } > "$report_file"
    
    print_success "报告生成: $report_file"
    cat "$report_file"
}

# 主函数
main() {
    echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║      姿态分析系统 - 快速测试脚本      ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${NC}"
    
    local total_tests=0
    local passed_tests=0
    
    # 运行各项检查
    if check_dependencies; then ((passed_tests++)); fi
    ((total_tests++))
    
    if check_source_files; then ((passed_tests++)); fi
    ((total_tests++))
    
    if syntax_check; then ((passed_tests++)); fi
    ((total_tests++))
    
    if create_simple_test; then ((passed_tests++)); fi
    ((total_tests++))
    
    check_code_style
    
    # 生成报告
    generate_report
    
    # 显示总结
    print_header "测试总结"
    echo ""
    if [ "$passed_tests" -eq "$total_tests" ]; then
        echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║         ✓ 所有测试通过！              ║${NC}"
        echo -e "${GREEN}║         通过: $passed_tests/$total_tests                        ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo -e "${YELLOW}╔════════════════════════════════════════╗${NC}"
        echo -e "${YELLOW}║         ⚠ 部分测试未通过              ║${NC}"
        echo -e "${YELLOW}║         通过: $passed_tests/$total_tests                        ║${NC}"
        echo -e "${YELLOW}╚════════════════════════════════════════╝${NC}"
        exit 1
    fi
}

# 如果带参数运行
case "$1" in
    files)
        check_source_files
        ;;
    deps)
        check_dependencies
        ;;
    syntax)
        syntax_check
        ;;
    test)
        create_simple_test
        ;;
    style)
        check_code_style
        ;;
    report)
        generate_report
        ;;
    *)
        main
        ;;
esac