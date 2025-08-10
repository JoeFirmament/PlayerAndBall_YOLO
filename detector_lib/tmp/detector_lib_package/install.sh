#!/bin/bash

# Detector Lib 一键安装脚本

set -e

echo "🚀 开始安装 Detector Lib..."

# 检查权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用sudo运行此脚本"
    exit 1
fi

# 检查依赖
echo "📦 检查系统依赖..."
if ! dpkg -l | grep -q libopencv-dev; then
    echo "⚠️  警告: 未找到 libopencv-dev，请先安装："
    echo "sudo apt install libopencv-dev libeigen3-dev"
    read -p "继续安装吗？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 安装库文件
echo "📚 安装库文件到 /usr/local/lib..."
cp lib/* /usr/local/lib/
chmod 755 /usr/local/lib/libdetector_lib.so.1.0.3
chmod 755 /usr/local/lib/libdetector_lib.so.1
chmod 755 /usr/local/lib/libdetector_lib.so
chmod 644 /usr/local/lib/libdetector_lib.a
# 设置RKNN Runtime库权限
if [ -f "/usr/local/lib/librknnrt.so" ]; then
    chmod 755 /usr/local/lib/librknnrt.so
    echo "✅ RKNN Runtime库权限设置完成"
fi

# 安装头文件
echo "📄 安装头文件到 /usr/local/include..."
cp include/* /usr/local/include/
chmod 644 /usr/local/include/*.h

# 安装模型文件
echo "🤖 安装模型文件到 /usr/local/share/detector_lib..."
mkdir -p /usr/local/share/detector_lib/models
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    cp models/* /usr/local/share/detector_lib/models/
    chmod 644 /usr/local/share/detector_lib/models/*
    echo "✅ 模型文件安装完成"
else
    echo "⚠️  警告: 没有找到模型文件，请手动复制.rknn文件到 /usr/local/share/detector_lib/models/"
fi

# 安装pkg-config文件
echo "⚙️  安装pkg-config配置..."
mkdir -p /usr/local/lib/pkgconfig
cp scripts/detector_lib.pc /usr/local/lib/pkgconfig/
chmod 644 /usr/local/lib/pkgconfig/detector_lib.pc

# 更新动态库缓存
echo "🔄 更新动态库缓存..."
ldconfig

# 设置NPU设备权限
echo "🎮 设置NPU设备权限..."
if [ -e /dev/dri/renderD128 ]; then
    chmod 666 /dev/dri/renderD*
    echo "✅ NPU设备权限设置完成"
else
    echo "⚠️  警告: 未找到NPU设备，请确认在RK3588平台上运行"
fi

echo ""
echo "🎉 Detector Lib 安装完成！"
echo ""
echo "📝 测试安装："
echo "  cd examples"
echo "  g++ sample_code.cpp \$(pkg-config --cflags --libs detector_lib) -o test_detector"
echo "  ./test_detector your_image.jpg"
echo ""
echo "📚 更多信息请查看用户指南: USER_GUIDE.md"
