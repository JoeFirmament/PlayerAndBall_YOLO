#!/bin/bash

# 基于Pose检测的身高测量测试程序编译和运行脚本
# 使用方法: ./run_pose_height_test.sh

echo "🔥 编译基于Pose检测的身高测量测试程序..."

# 编译参数
CXX="g++"
CXXFLAGS="-std=c++17 -O2"

# 包含路径
INCLUDES="-I../include"
INCLUDES="$INCLUDES -I../../detector_lib/include"
INCLUDES="$INCLUDES -I/usr/include/opencv4"
INCLUDES="$INCLUDES -I/usr/include/jsoncpp"

# 链接库
LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs"
LIBS="$LIBS -ljsoncpp"

# 检测器库路径
DETECTOR_LIB_PATH="../../detector_lib"

# 查找detector_lib库
if [ -f "$DETECTOR_LIB_PATH/build/libdetector_lib_static.a" ]; then
    DETECTOR_LIB="$DETECTOR_LIB_PATH/build/libdetector_lib_static.a"
    echo "✓ 使用静态库: $DETECTOR_LIB"
elif [ -f "$DETECTOR_LIB_PATH/build/libdetector_lib.so" ]; then
    DETECTOR_LIB="-L$DETECTOR_LIB_PATH/build -ldetector_lib"
    echo "✓ 使用动态库: $DETECTOR_LIB"
else
    echo "❌ 无法找到detector_lib库，请先编译detector_lib"
    echo "  提示: cd ../../detector_lib && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 查找pose_analysis库
POSE_ANALYSIS_LIB_PATH="../build"
if [ -f "$POSE_ANALYSIS_LIB_PATH/libpose_analysis_static.a" ]; then
    POSE_ANALYSIS_LIB="$POSE_ANALYSIS_LIB_PATH/libpose_analysis_static.a"
    echo "✓ 使用姿态分析静态库: $POSE_ANALYSIS_LIB"
elif [ -f "$POSE_ANALYSIS_LIB_PATH/libpose_analysis.so" ]; then
    POSE_ANALYSIS_LIB="-L$POSE_ANALYSIS_LIB_PATH -lpose_analysis"
    echo "✓ 使用姿态分析动态库: $POSE_ANALYSIS_LIB"
else
    echo "⚠️  未找到pose_analysis预编译库，将尝试直接编译源文件"
    POSE_ANALYSIS_SOURCES="../src/pose_analyzer.cpp ../src/height_detector.cpp"
    echo "✓ 使用姿态分析源文件: $POSE_ANALYSIS_SOURCES"
fi

# 源文件和输出
SOURCE="height_measurement_test_with_pose.cpp"
OUTPUT="height_measurement_test_with_pose"

# 构建编译命令
COMPILE_CMD="$CXX $CXXFLAGS $INCLUDES $SOURCE"

if [ -n "$POSE_ANALYSIS_LIB" ]; then
    # 使用预编译库
    COMPILE_CMD="$COMPILE_CMD $POSE_ANALYSIS_LIB $DETECTOR_LIB $LIBS"
else
    # 直接编译源文件
    COMPILE_CMD="$COMPILE_CMD $POSE_ANALYSIS_SOURCES $DETECTOR_LIB $LIBS"
fi

COMPILE_CMD="$COMPILE_CMD -o $OUTPUT"

echo "执行编译命令:"
echo "$COMPILE_CMD"
echo ""

# 编译
eval $COMPILE_CMD

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 运行基于Pose检测的身高测量测试..."
    echo "================================"
    
    # 设置运行时库路径
    export LD_LIBRARY_PATH="$DETECTOR_LIB_PATH/build:$POSE_ANALYSIS_LIB_PATH:$LD_LIBRARY_PATH"
    
    # 运行程序
    ./$OUTPUT
    
    echo ""
    echo "================================"
    echo "✅ 测试完成！"
else
    echo "❌ 编译失败！"
    echo ""
    echo "可能的解决方案:"
    echo "1. 确保detector_lib已编译: cd ../../detector_lib && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    echo "2. 确保pose_analysis已编译: cd .. && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    echo "3. 检查OpenCV和jsoncpp是否正确安装"
    exit 1
fi