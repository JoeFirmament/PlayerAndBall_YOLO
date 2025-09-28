#!/bin/bash

# 身高测量测试程序编译和运行脚本
# 使用方法: ./run_height_test.sh

echo "🔥 编译身高测量测试程序..."

# 编译参数
CXX="g++"
CXXFLAGS="-std=c++17 -O2"
INCLUDES="-I../include -I../../detector_lib/include -I/usr/include/opencv4"
LIBS="-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -ljsoncpp"
JSON_INCLUDE="-I/usr/include/jsoncpp"

# 源文件和输出
SOURCE="height_measurement_test.cpp"
OUTPUT="height_measurement_test"

# 编译
$CXX $CXXFLAGS $INCLUDES $JSON_INCLUDE $SOURCE $LIBS -o $OUTPUT

if [ $? -eq 0 ]; then
    echo "✅ 编译成功！"
    echo ""
    echo "🚀 运行身高测量测试..."
    echo "================================"
    
    # 运行程序
    ./$OUTPUT
    
    echo ""
    echo "================================"
    echo "✅ 测试完成！"
else
    echo "❌ 编译失败！"
    exit 1
fi