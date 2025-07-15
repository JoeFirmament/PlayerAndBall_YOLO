#!/bin/bash

echo "=== RGA硬件加速性能测试脚本 ==="

# 编译测试程序
echo "编译RGA性能测试程序..."
cd build

g++ -std=c++11 \
    -I../include \
    -I../3rdparty/librga/include \
    -I../3rdparty/opencv/opencv-linux-aarch64/include \
    -L../3rdparty/librga/Linux/aarch64 \
    -L../3rdparty/opencv/opencv-linux-aarch64/lib \
    -o rga_resize_test \
    ../src/rga_resize_test.cpp \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -lrga -lpthread

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo "✅ 编译成功"

# 设置环境变量
export LD_LIBRARY_PATH=../libs:../3rdparty/opencv/opencv-linux-aarch64/lib:$LD_LIBRARY_PATH

# 运行测试
echo ""
echo "开始性能测试..."
echo "测试场景: 1920x1080 -> 640x640 (模拟实际使用)"
echo ""

./rga_resize_test

echo ""
echo "=== 性能分析建议 ==="
echo "1. 如果RGA比OpenCV快2x以上，建议使用RGA"
echo "2. 如果差异不大，保持OpenCV（兼容性更好）"
echo "3. 检查结果图像质量是否满足要求"
echo ""
echo "结果图像保存在 build/ 目录下"

cd ..