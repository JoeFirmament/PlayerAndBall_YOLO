#!/bin/bash

# ========================================
# 姿态分析系统测试运行脚本
# ========================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build_pose_analysis"
TEST_BINARY="${BUILD_DIR}/tests/pose_analysis_tests"
VERBOSE=false
FILTER=""
REPEAT=1
OUTPUT_XML=false
OUTPUT_DIR="${PROJECT_ROOT}/test_results"
VALGRIND=false
COVERAGE=false

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

print_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

# 显示帮助
show_help() {
    cat << EOF
姿态分析系统测试脚本

使用方法:
    $0 [选项]

选项:
    -h, --help          显示帮助信息
    -v, --verbose       显示详细测试输出
    -f, --filter PATTERN 只运行匹配的测试
    -r, --repeat N      重复运行测试N次
    -x, --xml           生成XML测试报告
    -m, --memcheck      使用valgrind检查内存泄漏
    -c, --coverage      生成代码覆盖率报告
    -b, --build         先编译再测试
    -l, --list          列出所有测试用例

示例:
    $0                          # 运行所有测试
    $0 -f "HeightDetector*"     # 只运行身高检测测试
    $0 -v -r 3                  # 详细模式运行3次
    $0 -m                       # 内存泄漏检查
    $0 --coverage               # 生成覆盖率报告

测试分类:
    TemporalBuffer*     时序缓冲区测试
    FilterInterface*    滤波器接口测试
    HeightDetector*     身高检测模块测试
    BallRequest*        要球动作检测测试
    IDPriority*         ID优先级管理测试
    PoseAnalyzer*       集成分析器测试

EOF
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -f|--filter)
                FILTER="$2"
                shift 2
                ;;
            -r|--repeat)
                REPEAT="$2"
                shift 2
                ;;
            -x|--xml)
                OUTPUT_XML=true
                shift
                ;;
            -m|--memcheck)
                VALGRIND=true
                shift
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            -b|--build)
                BUILD_FIRST=true
                shift
                ;;
            -l|--list)
                LIST_ONLY=true
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

# 检查测试程序
check_test_binary() {
    if [ ! -f "$TEST_BINARY" ]; then
        print_error "测试程序不存在: $TEST_BINARY"
        print_info "请先运行编译脚本: $SCRIPT_DIR/build_pose_analysis.sh"
        
        # 询问是否自动编译
        read -p "是否现在编译? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            "$SCRIPT_DIR/build_pose_analysis.sh"
        else
            exit 1
        fi
    fi
}

# 列出所有测试
list_tests() {
    print_info "列出所有测试用例..."
    "$TEST_BINARY" --gtest_list_tests | while IFS= read -r line; do
        if [[ $line == *"."* ]]; then
            echo -e "${CYAN}$line${NC}"
        else
            echo "  $line"
        fi
    done
}

# 运行测试
run_tests() {
    local test_args=()
    
    # 添加过滤器
    if [ -n "$FILTER" ]; then
        test_args+=("--gtest_filter=$FILTER")
        print_info "测试过滤器: $FILTER"
    fi
    
    # 添加重复次数
    if [ "$REPEAT" -gt 1 ]; then
        test_args+=("--gtest_repeat=$REPEAT")
        print_info "重复运行: $REPEAT 次"
    fi
    
    # 添加XML输出
    if [ "$OUTPUT_XML" = true ]; then
        mkdir -p "$OUTPUT_DIR"
        local xml_file="${OUTPUT_DIR}/test_results_$(date +%Y%m%d_%H%M%S).xml"
        test_args+=("--gtest_output=xml:$xml_file")
        print_info "XML报告: $xml_file"
    fi
    
    # 颜色输出
    test_args+=("--gtest_color=yes")
    
    # 详细模式
    if [ "$VERBOSE" = false ]; then
        test_args+=("--gtest_brief=1")
    fi
    
    print_test "开始运行测试..."
    echo ""
    
    # 运行测试
    if [ "$VALGRIND" = true ]; then
        run_with_valgrind "${test_args[@]}"
    else
        "$TEST_BINARY" "${test_args[@]}"
    fi
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "所有测试通过！"
    else
        print_error "测试失败，退出码: $exit_code"
        return $exit_code
    fi
}

# 使用valgrind运行
run_with_valgrind() {
    print_info "使用valgrind进行内存检查..."
    
    local valgrind_args=(
        --tool=memcheck
        --leak-check=full
        --show-leak-kinds=all
        --track-origins=yes
        --verbose
        --log-file="${OUTPUT_DIR}/valgrind_$(date +%Y%m%d_%H%M%S).log"
    )
    
    if ! command -v valgrind &> /dev/null; then
        print_error "valgrind未安装，请先安装: sudo apt install valgrind"
        exit 1
    fi
    
    mkdir -p "$OUTPUT_DIR"
    valgrind "${valgrind_args[@]}" "$TEST_BINARY" "$@"
    
    print_info "Valgrind日志: ${valgrind_args[-1]#*=}"
}

# 生成覆盖率报告
generate_coverage() {
    print_info "生成代码覆盖率报告..."
    
    if ! command -v lcov &> /dev/null; then
        print_error "lcov未安装，请先安装: sudo apt install lcov"
        exit 1
    fi
    
    cd "$BUILD_DIR"
    
    # 清理之前的覆盖率数据
    lcov --zerocounters --directory .
    
    # 运行测试
    "$TEST_BINARY"
    
    # 收集覆盖率数据
    lcov --capture --directory . --output-file coverage.info
    
    # 过滤系统文件
    lcov --remove coverage.info '/usr/*' '*/tests/*' --output-file coverage_filtered.info
    
    # 生成HTML报告
    genhtml coverage_filtered.info --output-directory "${OUTPUT_DIR}/coverage_html"
    
    print_success "覆盖率报告生成在: ${OUTPUT_DIR}/coverage_html/index.html"
}

# 运行特定类别的测试
run_category() {
    local category=$1
    print_info "运行测试类别: $category"
    
    case $category in
        buffer)
            FILTER="TemporalBufferTest.*"
            ;;
        filter)
            FILTER="FilterInterfaceTest.*"
            ;;
        height)
            FILTER="HeightDetectorTest.*"
            ;;
        request)
            FILTER="BallRequestDetectorTest.*"
            ;;
        priority)
            FILTER="IDPriorityManagerTest.*"
            ;;
        analyzer)
            FILTER="PoseAnalyzerTest.*"
            ;;
        *)
            print_error "未知的测试类别: $category"
            print_info "可用类别: buffer, filter, height, request, priority, analyzer"
            exit 1
            ;;
    esac
    
    run_tests
}

# 运行性能测试
run_performance_tests() {
    print_info "运行性能测试..."
    
    # 设置性能测试环境变量
    export BENCHMARK_REPETITIONS=10
    export BENCHMARK_MIN_TIME=1
    
    # 运行性能相关的测试
    FILTER="*Performance*"
    run_tests
    
    # 生成性能报告
    if [ -f "${BUILD_DIR}/performance_report.txt" ]; then
        print_info "性能报告:"
        cat "${BUILD_DIR}/performance_report.txt"
    fi
}

# 显示测试统计
show_statistics() {
    print_info "测试统计信息"
    echo "----------------------------------------"
    
    # 统计测试数量
    local total_tests=$("$TEST_BINARY" --gtest_list_tests | grep -c "  " || true)
    echo "总测试数: $total_tests"
    
    # 按类别统计
    echo ""
    echo "按模块分类:"
    echo "  时序缓冲区: $("$TEST_BINARY" --gtest_list_tests | grep -c "TemporalBuffer" || true)"
    echo "  滤波器接口: $("$TEST_BINARY" --gtest_list_tests | grep -c "FilterInterface" || true)"
    echo "  身高检测: $("$TEST_BINARY" --gtest_list_tests | grep -c "HeightDetector" || true)"
    echo "  要球检测: $("$TEST_BINARY" --gtest_list_tests | grep -c "BallRequest" || true)"
    echo "  ID管理: $("$TEST_BINARY" --gtest_list_tests | grep -c "IDPriority" || true)"
    echo "  分析器: $("$TEST_BINARY" --gtest_list_tests | grep -c "PoseAnalyzer" || true)"
    echo "----------------------------------------"
}

# 清理测试结果
clean_results() {
    print_info "清理测试结果..."
    rm -rf "$OUTPUT_DIR"
    print_success "清理完成"
}

# 主函数
main() {
    echo "========================================"
    echo "        姿态分析系统测试脚本"
    echo "========================================"
    echo ""
    
    # 解析参数
    parse_args "$@"
    
    # 检查测试程序
    check_test_binary
    
    # 如果只是列出测试
    if [ "$LIST_ONLY" = true ]; then
        list_tests
        exit 0
    fi
    
    # 如果需要先编译
    if [ "$BUILD_FIRST" = true ]; then
        print_info "先编译项目..."
        "$SCRIPT_DIR/build_pose_analysis.sh"
    fi
    
    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"
    
    # 显示统计信息
    show_statistics
    echo ""
    
    # 运行测试
    if [ "$COVERAGE" = true ]; then
        generate_coverage
    else
        run_tests
    fi
    
    echo ""
    print_info "测试完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 如果有XML输出，显示位置
    if [ "$OUTPUT_XML" = true ]; then
        print_info "测试结果保存在: $OUTPUT_DIR"
    fi
}

# 特殊命令处理
if [ "$1" = "clean" ]; then
    clean_results
    exit 0
elif [ "$1" = "perf" ]; then
    shift
    run_performance_tests
    exit 0
elif [ "$1" = "category" ] && [ -n "$2" ]; then
    run_category "$2"
    exit 0
fi

# 运行主函数
main "$@"