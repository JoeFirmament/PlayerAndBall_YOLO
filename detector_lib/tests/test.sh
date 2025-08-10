#!/bin/bash

echo "====================================================="
echo "    YOLOv8检测器库 - 功能验证脚本"
echo "====================================================="
echo "版本: v1.0"
echo "测试: 姿态检测 + 篮筐篮球检测"
echo "====================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# 检查必要文件
check_prerequisites() {
    log_info "🔍 检查测试环境..."
    
    # 检查是否已编译
    if [ ! -d "build" ] || [ ! -f "build/examples/pose_image" ]; then
        log_error "项目未编译，请先运行: ./build_and_install.sh"
        exit 1
    fi
    
    # 检查模型文件
    if [ ! -f "models/Q_yolov8_pose.rknn" ]; then
        log_error "缺少姿态检测模型: models/Q_yolov8_pose.rknn"
        return 1
    fi
    
    if [ ! -f "models/Q_Rim_Basketball_724_JZ.rknn" ]; then
        log_error "缺少篮筐篮球检测模型: models/Q_Rim_Basketball_724_JZ.rknn"
        return 1
    fi
    
    # 检查测试图片
    local test_images=()
    if [ -f "imgs/pose.jpg" ]; then
        test_images+=("imgs/pose.jpg")
    fi
    if [ -f "imgs/rim.jpg" ]; then
        test_images+=("imgs/rim.jpg")
    fi
    if [ -f "test_person.jpg" ]; then
        test_images+=("test_person.jpg")
    fi
    
    if [ ${#test_images[@]} -eq 0 ]; then
        log_warning "未找到测试图片，将尝试生成测试数据"
        create_test_data
    else
        log_success "发现测试图片: ${test_images[*]}"
    fi
    
    log_success "环境检查完成"
    return 0
}

# 创建测试数据（如果需要）
create_test_data() {
    log_info "📝 创建测试数据..."
    
    # 创建简单的测试图片（纯色图片用于基础测试）
    cd build/examples
    
    # 使用OpenCV创建测试图片
    python3 -c "
import cv2
import numpy as np
import sys

try:
    # 创建一个简单的测试图片
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 添加一些简单的几何图形
    cv2.rectangle(img, (100, 100), (200, 300), (0, 255, 0), -1)
    cv2.circle(img, (400, 200), 50, (255, 0, 0), -1)
    cv2.putText(img, 'TEST IMAGE', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imwrite('test_image.jpg', img)
    print('测试图片创建成功: test_image.jpg')
except ImportError:
    print('Python3和OpenCV不可用，将跳过测试图片创建')
    sys.exit(1)
" 2>/dev/null || {
        log_warning "无法创建测试图片，将使用现有图片"
    }
    
    cd ../..
}

# 测试NPU权限
test_npu_permissions() {
    log_info "🔧 测试NPU设备权限..."
    
    if ls /dev/dri/renderD* &> /dev/null; then
        local npu_devices=($(ls /dev/dri/renderD*))
        log_success "发现NPU设备: ${npu_devices[*]}"
        
        # 检查权限
        local permission_ok=true
        for device in "${npu_devices[@]}"; do
            if [ ! -w "$device" ]; then
                log_warning "NPU设备无写权限: $device"
                permission_ok=false
            fi
        done
        
        if [ "$permission_ok" = true ]; then
            log_success "NPU设备权限正常"
        else
            log_warning "NPU权限不足，可能影响性能。建议运行:"
            echo "    sudo chmod 666 /dev/dri/renderD*"
        fi
    else
        log_warning "未发现NPU设备，可能影响推理性能"
    fi
}

# 测试姿态检测
test_pose_detection() {
    log_info "🧍 测试姿态检测功能..."
    
    cd build/examples
    
    # 查找测试图片
    local test_img=""
    if [ -f "../../imgs/pose.jpg" ]; then
        test_img="../../imgs/pose.jpg"
    elif [ -f "../../test_person.jpg" ]; then
        test_img="../../test_person.jpg"
    elif [ -f "test_image.jpg" ]; then
        test_img="test_image.jpg"
    else
        log_error "没有可用的测试图片"
        cd ../..
        return 1
    fi
    
    log_info "使用测试图片: $test_img"
    
    # 运行姿态检测测试
    timeout 60s ./pose_image "$test_img" > pose_test.log 2>&1
    local result=$?
    
    if [ $result -eq 0 ]; then
        # 分析结果
        local inference_time=$(grep "推理时间:" pose_test.log | grep -o '[0-9]\+ms' | head -1)
        local detection_count=$(grep "检测到目标数量:" pose_test.log | grep -o '[0-9]\+' | head -1)
        
        log_success "姿态检测测试通过"
        if [ -n "$inference_time" ]; then
            log_info "  推理时间: $inference_time"
        fi
        if [ -n "$detection_count" ]; then
            log_info "  检测数量: $detection_count"
        fi
        
        # 检查输出文件
        if [ -f "pose_detection_result.jpg" ]; then
            log_success "  结果图片: pose_detection_result.jpg"
        fi
    elif [ $result -eq 124 ]; then
        log_warning "姿态检测测试超时 (可能是首次运行预热)"
    else
        log_error "姿态检测测试失败"
        log_info "错误日志:"
        tail -10 pose_test.log | sed 's/^/    /'
    fi
    
    cd ../..
    return $result
}

# 测试篮筐篮球检测
test_basketball_detection() {
    log_info "🏀 测试篮筐篮球检测功能..."
    
    cd build/examples
    
    # 查找测试图片
    local test_img=""
    if [ -f "../../imgs/rim.jpg" ]; then
        test_img="../../imgs/rim.jpg"
    elif [ -f "test_image.jpg" ]; then
        test_img="test_image.jpg"
    else
        log_error "没有可用的测试图片"
        cd ../..
        return 1
    fi
    
    log_info "使用测试图片: $test_img"
    
    # 运行篮筐篮球检测测试
    timeout 60s ./rim_basketball_image "$test_img" > basketball_test.log 2>&1
    local result=$?
    
    if [ $result -eq 0 ]; then
        # 分析结果
        local inference_time=$(grep "推理时间:" basketball_test.log | grep -o '[0-9]\+ms' | head -1)
        local detection_count=$(grep "检测到目标数量:" basketball_test.log | grep -o '[0-9]\+' | head -1)
        
        log_success "篮筐篮球检测测试通过"
        if [ -n "$inference_time" ]; then
            log_info "  推理时间: $inference_time"
        fi
        if [ -n "$detection_count" ]; then
            log_info "  检测数量: $detection_count"
        fi
        
        # 检查输出文件
        if [ -f "rim_basketball_detection_result.jpg" ]; then
            log_success "  结果图片: rim_basketball_detection_result.jpg"
        fi
    elif [ $result -eq 124 ]; then
        log_warning "篮筐篮球检测测试超时 (可能是首次运行预热)"
    else
        log_error "篮筐篮球检测测试失败"
        log_info "错误日志:"
        tail -10 basketball_test.log | sed 's/^/    /'
    fi
    
    cd ../..
    return $result
}

# 性能基准测试
run_performance_test() {
    log_info "⚡ 运行性能基准测试..."
    
    cd build/examples
    
    # 准备测试图片
    local test_img=""
    if [ -f "../../imgs/pose.jpg" ]; then
        test_img="../../imgs/pose.jpg"
    elif [ -f "test_image.jpg" ]; then
        test_img="test_image.jpg"
    else
        log_warning "跳过性能测试 - 没有测试图片"
        cd ../..
        return 0
    fi
    
    log_info "运行多次推理测试..."
    
    # 运行多次姿态检测测试
    local total_time=0
    local success_count=0
    
    for i in {1..3}; do
        log_info "第 $i 次姿态检测..."
        if timeout 30s ./pose_image "$test_img" > /dev/null 2>&1; then
            success_count=$((success_count + 1))
        fi
    done
    
    if [ $success_count -eq 3 ]; then
        log_success "性能测试通过 (3/3 次成功)"
    else
        log_warning "性能测试部分通过 ($success_count/3 次成功)"
    fi
    
    cd ../..
}

# 生成测试报告
generate_report() {
    log_info "📊 生成测试报告..."
    
    local report_file="test_report.txt"
    
    cat > "$report_file" << EOF
YOLOv8检测器库 - 测试报告
==========================

测试时间: $(date)
平台信息: $(uname -a)

库文件信息:
EOF
    
    if [ -f "build/libdetector_lib.so" ]; then
        local lib_size=$(du -h build/libdetector_lib.so | cut -f1)
        echo "  动态库: build/libdetector_lib.so ($lib_size)" >> "$report_file"
    fi
    
    if [ -f "build/libdetector_lib.a" ]; then
        local lib_size=$(du -h build/libdetector_lib.a | cut -f1)
        echo "  静态库: build/libdetector_lib.a ($lib_size)" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "模型文件信息:" >> "$report_file"
    
    if [ -f "models/Q_yolov8_pose.rknn" ]; then
        local model_size=$(du -h models/Q_yolov8_pose.rknn | cut -f1)
        echo "  姿态检测模型: models/Q_yolov8_pose.rknn ($model_size)" >> "$report_file"
    fi
    
    if [ -f "models/Q_Rim_Basketball_724_JZ.rknn" ]; then
        local model_size=$(du -h models/Q_Rim_Basketball_724_JZ.rknn | cut -f1)
        echo "  篮筐篮球模型: models/Q_Rim_Basketball_724_JZ.rknn ($model_size)" >> "$report_file"
    fi
    
    echo "" >> "$report_file"
    echo "测试结果将在下方显示..." >> "$report_file"
    
    log_success "测试报告已生成: $report_file"
}

# 清理测试文件
cleanup_test_files() {
    log_info "🧹 清理测试临时文件..."
    
    cd build/examples 2>/dev/null || return 0
    
    # 清理日志文件
    rm -f pose_test.log basketball_test.log
    
    # 清理测试生成的图片（保留结果图片）
    rm -f test_image.jpg
    
    cd ../..
    
    log_info "临时文件清理完成"
}

# 主函数
main() {
    echo ""
    
    # 检查环境
    check_prerequisites || exit 1
    
    # 测试NPU权限
    test_npu_permissions
    
    # 生成报告
    generate_report
    
    echo ""
    log_info "🚀 开始功能测试..."
    
    local pose_result=1
    local basketball_result=1
    
    # 测试姿态检测
    if test_pose_detection; then
        pose_result=0
    fi
    
    echo ""
    
    # 测试篮筐篮球检测
    if test_basketball_detection; then
        basketball_result=0
    fi
    
    echo ""
    
    # 性能测试
    run_performance_test
    
    echo ""
    log_info "📋 测试总结:"
    
    if [ $pose_result -eq 0 ]; then
        log_success "✅ 姿态检测功能正常"
    else
        log_error "❌ 姿态检测功能异常"
    fi
    
    if [ $basketball_result -eq 0 ]; then
        log_success "✅ 篮筐篮球检测功能正常"
    else
        log_error "❌ 篮筐篮球检测功能异常"
    fi
    
    # 整体结果
    if [ $pose_result -eq 0 ] && [ $basketball_result -eq 0 ]; then
        log_success "🎉 所有测试通过！库功能正常"
        echo ""
        log_info "📖 使用指南:"
        log_info "1. 查看示例代码: examples/"
        log_info "2. 阅读API文档: docs/"
        log_info "3. 集成到项目: #include \"detector_lib.h\""
        
        # 清理测试文件
        cleanup_test_files
        
        exit 0
    else
        log_warning "⚠️  部分测试失败，请检查:"
        log_info "1. NPU设备权限是否正确"
        log_info "2. 模型文件是否完整"
        log_info "3. 系统是否为RK3588平台"
        
        exit 1
    fi
}

# 信号处理
trap 'echo -e "\n❌ 测试被中断"; cleanup_test_files; exit 1' INT TERM

# 运行主程序
main "$@"