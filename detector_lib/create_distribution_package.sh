#!/bin/bash

# Detector Lib 分发包创建脚本
# 用于为用户创建完整的库使用包

set -e

# 配置变量
PACKAGE_NAME="detector_lib_package"
PACKAGE_VERSION="1.0.0"
BUILD_DIR="build"
DIST_DIR="dist"

echo "🚀 创建 Detector Lib 分发包 v${PACKAGE_VERSION}"

# 检查build目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ 错误: build目录不存在，请先编译项目"
    echo "运行: mkdir build && cd build && cmake .. && make"
    exit 1
fi

# 检查必要文件
required_files=(
    "$BUILD_DIR/libdetector_lib.so.1.0.0"
    "$BUILD_DIR/libdetector_lib.so"
    "$BUILD_DIR/libdetector_lib.a"
    "include/detector_lib.h"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 错误: 找不到必要文件 $file"
        exit 1
    fi
done

# 创建分发目录结构
echo "📁 创建分发包目录结构..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/$PACKAGE_NAME"/{lib,include,models,examples,scripts}

# 复制库文件
echo "📚 复制库文件..."
cp "$BUILD_DIR/libdetector_lib.so.1.0.0" "$DIST_DIR/$PACKAGE_NAME/lib/"
cp "$BUILD_DIR/libdetector_lib.so.1" "$DIST_DIR/$PACKAGE_NAME/lib/"
cp "$BUILD_DIR/libdetector_lib.so" "$DIST_DIR/$PACKAGE_NAME/lib/"
cp "$BUILD_DIR/libdetector_lib.a" "$DIST_DIR/$PACKAGE_NAME/lib/"

# 复制头文件
echo "📄 复制头文件..."
cp include/detector_lib.h "$DIST_DIR/$PACKAGE_NAME/include/"
cp include/detector_types.h "$DIST_DIR/$PACKAGE_NAME/include/"
cp include/PoseDetectorLib.h "$DIST_DIR/$PACKAGE_NAME/include/"
cp include/RimBasketballDetectorLib.h "$DIST_DIR/$PACKAGE_NAME/include/"
# 复制其他必要的头文件
cp include/common.h "$DIST_DIR/$PACKAGE_NAME/include/" 2>/dev/null || true
cp include/file_utils.h "$DIST_DIR/$PACKAGE_NAME/include/" 2>/dev/null || true

# 复制模型文件（如果存在）
echo "🤖 复制模型文件..."
if [ -d "models" ]; then
    cp models/*.rknn "$DIST_DIR/$PACKAGE_NAME/models/" 2>/dev/null || echo "⚠️  警告: 没有找到.rknn模型文件"
fi

# 复制示例程序
echo "💡 复制示例程序..."
if [ -d "$BUILD_DIR/examples" ]; then
    cp "$BUILD_DIR/examples/pose_image" "$DIST_DIR/$PACKAGE_NAME/examples/" 2>/dev/null || true
    cp "$BUILD_DIR/examples/rim_basketball_image" "$DIST_DIR/$PACKAGE_NAME/examples/" 2>/dev/null || true
    cp "$BUILD_DIR/examples/pose_image_with_homography" "$DIST_DIR/$PACKAGE_NAME/examples/" 2>/dev/null || true
fi

# 创建示例代码文件
echo "📝 创建示例代码..."
cat > "$DIST_DIR/$PACKAGE_NAME/examples/sample_code.cpp" << 'EOF'
#include "detector_lib.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <图片路径>" << std::endl;
        return -1;
    }
    
    try {
        // 1. 创建检测器
        detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
        
        // 2. 加载图片
        cv::Mat image = cv::imread(argv[1]);
        if (image.empty()) {
            std::cout << "错误: 无法加载图片 " << argv[1] << std::endl;
            return -1;
        }
        
        // 3. 执行检测
        std::cout << "正在检测..." << std::endl;
        auto results = detector.detect(image);
        
        // 4. 显示结果
        std::cout << "检测到 " << results.size() << " 个人员:" << std::endl;
        for (const auto& pose : results) {
            std::cout << "  - 人员ID: " << pose.person_id 
                      << ", 置信度: " << pose.confidence
                      << ", 位置: (" << pose.bbox.x << "," << pose.bbox.y 
                      << "," << pose.bbox.width << "," << pose.bbox.height << ")"
                      << std::endl;
        }
        
        std::cout << "检测完成！推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
EOF

# 创建pkg-config文件
echo "⚙️  创建pkg-config文件..."
cat > "$DIST_DIR/$PACKAGE_NAME/scripts/detector_lib.pc" << EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: DetectorLib
Description: YOLOv8 Pose and Basketball Detection Library for RK3588
Version: ${PACKAGE_VERSION}
Requires: opencv4
Libs: -L\${libdir} -ldetector_lib -lrknn_api -lpthread
Cflags: -I\${includedir}
EOF

# 创建安装脚本
echo "🔧 创建安装脚本..."
cat > "$DIST_DIR/$PACKAGE_NAME/install.sh" << 'EOF'
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
chmod 755 /usr/local/lib/libdetector_lib.so.1.0.0
chmod 644 /usr/local/lib/libdetector_lib.a

# 安装头文件
echo "📄 安装头文件到 /usr/local/include..."
cp include/* /usr/local/include/
chmod 644 /usr/local/include/detector_lib.h

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
EOF

chmod +x "$DIST_DIR/$PACKAGE_NAME/install.sh"

# 创建卸载脚本
echo "🗑️  创建卸载脚本..."
cat > "$DIST_DIR/$PACKAGE_NAME/uninstall.sh" << 'EOF'
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
EOF

chmod +x "$DIST_DIR/$PACKAGE_NAME/uninstall.sh"

# 复制文档
echo "📖 复制文档..."
cp README.md "$DIST_DIR/$PACKAGE_NAME/" 2>/dev/null || echo "⚠️  警告: 未找到README.md"
cp USER_GUIDE.md "$DIST_DIR/$PACKAGE_NAME/" 2>/dev/null || echo "⚠️  警告: 未找到USER_GUIDE.md"

# 创建简单的README
cat > "$DIST_DIR/$PACKAGE_NAME/README_QUICK.md" << EOF
# Detector Lib ${PACKAGE_VERSION} 快速开始

## 🚀 快速安装
\`\`\`bash
sudo ./install.sh
\`\`\`

## 📝 测试安装
\`\`\`bash
cd examples
g++ sample_code.cpp \$(pkg-config --cflags --libs detector_lib) -o test_detector
./test_detector your_image.jpg
\`\`\`

## 📚 详细文档
- **完整指南**: USER_GUIDE.md
- **API文档**: README.md

## 🗑️ 卸载
\`\`\`bash
sudo ./uninstall.sh
\`\`\`

## 📞 技术支持
如有问题请联系技术支持或查看文档中的故障排除部分。
EOF

# 创建版本信息文件
cat > "$DIST_DIR/$PACKAGE_NAME/VERSION" << EOF
Detector Lib v${PACKAGE_VERSION}
构建时间: $(date)
构建平台: $(uname -a)
构建目录: $(pwd)
EOF

# 计算文件校验和
echo "🔐 生成文件校验和..."
cd "$DIST_DIR/$PACKAGE_NAME"
find . -type f -exec sha256sum {} \; > SHA256SUMS
cd - > /dev/null

# 创建tar.gz包
echo "📦 创建压缩包..."
cd "$DIST_DIR"
tar -czf "${PACKAGE_NAME}_v${PACKAGE_VERSION}_$(date +%Y%m%d).tar.gz" "$PACKAGE_NAME"
cd - > /dev/null

# 显示结果
echo ""
echo "✅ 分发包创建完成！"
echo ""
echo "📦 分发包位置:"
echo "  目录: $DIST_DIR/$PACKAGE_NAME/"
echo "  压缩包: $DIST_DIR/${PACKAGE_NAME}_v${PACKAGE_VERSION}_$(date +%Y%m%d).tar.gz"
echo ""
echo "📋 包含内容:"
echo "  📚 库文件: lib/libdetector_lib.so*, lib/libdetector_lib.a"
echo "  📄 头文件: include/detector_lib.h"
echo "  🤖 模型文件: models/*.rknn (如果存在)"
echo "  💡 示例程序: examples/"
echo "  🔧 安装脚本: install.sh, uninstall.sh"
echo "  📖 文档: README.md, USER_GUIDE.md"
echo "  ⚙️  配置: scripts/detector_lib.pc"
echo ""
echo "📨 给用户的使用说明:"
echo "  1. 解压: tar -xzf ${PACKAGE_NAME}_v${PACKAGE_VERSION}_$(date +%Y%m%d).tar.gz"
echo "  2. 安装: cd $PACKAGE_NAME && sudo ./install.sh"
echo "  3. 测试: cd examples && g++ sample_code.cpp \$(pkg-config --cflags --libs detector_lib) -o test"
echo ""
EOF

chmod +x "$DIST_DIR/$PACKAGE_NAME/create_distribution_package.sh"