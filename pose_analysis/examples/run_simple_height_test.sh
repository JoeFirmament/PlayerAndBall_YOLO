#!/bin/bash

# 简化版身高测量测试程序编译和运行脚本
# 使用方法: ./run_simple_height_test.sh

echo "🔥 编译简化版身高测量测试程序..."

# 编译参数
CXX="g++"
CXXFLAGS="-std=c++17 -O2"
INCLUDES="-I/usr/include/opencv4 -I/usr/include/jsoncpp"
LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -ljsoncpp"

# 源文件和输出
SOURCE="simple_height_test_with_roi.cpp"
OUTPUT="simple_height_test_with_roi"

echo "编译命令: $CXX $CXXFLAGS $INCLUDES $SOURCE $LIBS -o $OUTPUT"

# 编译
$CXX $CXXFLAGS $INCLUDES $SOURCE $LIBS -o $OUTPUT

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 运行简化版身高测量测试..."
    echo "================================"
    
    # 运行程序
    ./$OUTPUT
    
    echo ""
    echo "================================"
    echo "✅ 测试完成！"
else
    echo "❌ 编译失败！请检查OpenCV和jsoncpp是否正确安装"
    exit 1
fi