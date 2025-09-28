#!/bin/bash

# 基于detector_lib的直接身高测量测试程序编译和运行脚本
# 使用方法: ./run_direct_height_test.sh

echo "🔥 编译基于detector_lib的身高测量测试程序..."

# 编译参数
CXX="g++"
CXXFLAGS="-std=c++17 -O2"

# 包含路径
INCLUDES="-I../../detector_lib/include"
INCLUDES="$INCLUDES -I/usr/include/opencv4"

# 库路径和链接库
DETECTOR_LIB_PATH="../../detector_lib"
LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs"

# 查找detector_lib库
if [ -f "$DETECTOR_LIB_PATH/build/libdetector_lib_static.a" ]; then
    DETECTOR_LIB="$DETECTOR_LIB_PATH/build/libdetector_lib_static.a"
    echo "✓ 使用静态库: $DETECTOR_LIB"
    
    # 静态库需要额外的RKNN链接
    DETECTOR_LIB_DIR="$DETECTOR_LIB_PATH"
    RKNN_LIB="-L$DETECTOR_LIB_DIR/lib -lrknnrt -ljsoncpp"
    
elif [ -f "$DETECTOR_LIB_PATH/build/libdetector_lib.so" ]; then
    DETECTOR_LIB="-L$DETECTOR_LIB_PATH/build -ldetector_lib"
    RKNN_LIB="-ljsoncpp"
    echo "✓ 使用动态库: $DETECTOR_LIB"
else
    echo "❌ 无法找到detector_lib库，请先编译detector_lib"
    echo "  提示: cd ../../detector_lib && ./scripts/build_pose_analysis.sh"
    exit 1
fi

# 源文件和输出
SOURCE="direct_height_measurement.cpp"
OUTPUT="direct_height_measurement"

# 构建编译命令
COMPILE_CMD="$CXX $CXXFLAGS $INCLUDES $SOURCE $DETECTOR_LIB $LIBS $RKNN_LIB -o $OUTPUT"

echo "执行编译命令:"
echo "$COMPILE_CMD"
echo ""

# 编译
eval $COMPILE_CMD

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 运行基于detector_lib的身高测量测试..."
    echo "================================"
    
    # 设置运行时库路径（确保找到RKNN库）
    export LD_LIBRARY_PATH="$DETECTOR_LIB_PATH/lib:$DETECTOR_LIB_PATH/build:$LD_LIBRARY_PATH"
    
    # 运行程序
    ./$OUTPUT
    
    echo ""
    echo "================================"
    echo "✅ 测试完成！"
else
    echo "❌ 编译失败！"
    echo ""
    echo "可能的解决方案:"
    echo "1. 确保detector_lib已编译: cd ../../detector_lib && ./scripts/build_pose_analysis.sh"
    echo "2. 检查RKNN库路径: ls ../../detector_lib/lib/librknnrt.so"
    echo "3. 检查OpenCV是否正确安装"
    exit 1
fi