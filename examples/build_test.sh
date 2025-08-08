#!/bin/bash

echo "=== 编译封装类测试程序 ==="

# 检查依赖
echo "检查编译环境..."
if ! pkg-config --exists opencv4; then
    echo "错误: 未找到OpenCV4，请安装 libopencv-dev"
    exit 1
fi

if [ ! -f "/usr/include/rknn_api.h" ] && [ ! -f "../include/rknn_api.h" ]; then
    echo "警告: 未找到RKNN头文件，可能需要设置包含路径"
fi

# 设置编译参数
CC=g++
CFLAGS="-std=c++11 -O2 -Wall"
INCLUDES="-I../include -I../src -I../utils -I../3rdparty/rknpu2/include"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"
LIBS="-L../3rdparty/rknpu2/Linux/aarch64 -lrknn_api -pthread"

echo "编译参数:"
echo "  编译器: $CC"
echo "  标志: $CFLAGS"
echo "  包含: $INCLUDES"
echo "  OpenCV: $OPENCV_FLAGS"
echo "  库: $LIBS"

# 编译PoseDetector测试程序 (仅编译检查，实际运行需要完整环境)
echo ""
echo "1. 编译 PoseDetector 测试程序..."
echo "命令: $CC $CFLAGS -o test_pose_detector test_pose_detector.cc ../src/PoseDetector.cc $INCLUDES $OPENCV_FLAGS $LIBS"

# 注意：这里只是展示编译命令，实际编译可能需要更多依赖文件
echo "   注意: 需要确保以下文件存在:"
echo "   - ../src/pose_yolov8.cc"
echo "   - ../src/pose_postprocess.cc" 
echo "   - ../src/pose_letterbox_utils.cc"
echo "   - ../src/BYTETracker.cpp"
echo "   - ../utils/image_utils.c"

# 编译RimBasketballDetector测试程序
echo ""
echo "2. 编译 RimBasketballDetector 测试程序..."
echo "命令: $CC $CFLAGS -o test_rim_basketball_detector test_rim_basketball_detector.cc ../src/RimBasketballDetector.cc $INCLUDES $OPENCV_FLAGS $LIBS"

echo "   注意: 需要确保以下文件存在:"
echo "   - ../src/rim_basketball_postprocess_simple.cpp"

# 简单的语法检查 (不链接)
echo ""
echo "3. 语法检查 (仅编译，不链接)..."

echo "检查 PoseDetector.h 语法..."
echo '#include "PoseDetector.h"' > temp_test.cc
echo 'int main() { return 0; }' >> temp_test.cc
if $CC $CFLAGS -c temp_test.cc $INCLUDES 2>/dev/null; then
    echo "   ✓ PoseDetector.h 语法正确"
else
    echo "   ✗ PoseDetector.h 语法错误"
fi
rm -f temp_test.cc temp_test.o

echo "检查 RimBasketballDetector.h 语法..."
echo '#include "RimBasketballDetector.h"' > temp_test.cc
echo 'int main() { return 0; }' >> temp_test.cc
if $CC $CFLAGS -c temp_test.cc $INCLUDES 2>/dev/null; then
    echo "   ✓ RimBasketballDetector.h 语法正确"
else
    echo "   ✗ RimBasketballDetector.h 语法错误"
fi
rm -f temp_test.cc temp_test.o

echo ""
echo "=== 编译检查完成 ==="
echo ""
echo "使用说明:"
echo "1. 确保模型文件存在:"
echo "   - models/Q_yolov8_pose.rknn"
echo "   - models/Q_Rim_Basketball_724_JZ.rknn"
echo ""
echo "2. 确保NPU权限:"
echo "   sudo chmod 666 /dev/dri/renderD*"
echo ""
echo "3. 运行测试程序:"
echo "   ./test_pose_detector 0                    # 姿态检测，使用摄像头0"
echo "   ./test_rim_basketball_detector 2         # 篮筐检测，使用摄像头2"
echo ""
echo "4. 按键控制:"
echo "   ESC: 退出程序"
echo "   T: 切换跟踪 (仅姿态检测)"
echo "   S: 截图保存"
echo "   C: 调整置信度 (仅篮筐检测)"