#!/bin/bash

# Detector Lib 卸载脚本

set -e

echo "🗑️  开始卸载 Detector Lib..."

# 检查权限
if [ "$EUID" -ne 0 ]; then
    echo "❌ 请使用sudo运行此脚本"
    exit 1
fi

# 删除库文件
echo "📚 删除库文件..."
rm -f /usr/local/lib/libdetector_lib.so*
rm -f /usr/local/lib/libdetector_lib.a

# 删除头文件
echo "📄 删除头文件..."
rm -f /usr/local/include/detector_lib.h

# 删除模型文件
echo "🤖 删除模型文件..."
rm -rf /usr/local/share/detector_lib

# 删除pkg-config文件
echo "⚙️  删除pkg-config配置..."
rm -f /usr/local/lib/pkgconfig/detector_lib.pc

# 更新动态库缓存
echo "🔄 更新动态库缓存..."
ldconfig

echo "✅ Detector Lib 卸载完成！"
