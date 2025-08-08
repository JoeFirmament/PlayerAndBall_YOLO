#!/bin/bash

# YOLOv8检测器库打包脚本
# 用途: 创建用户交付的完整压缩包

echo "====================================================="
echo "    YOLOv8检测器库 - 打包脚本"
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

# 检查当前目录
if [ ! -f "CMakeLists.txt" ] || [ ! -d "include" ]; then
    log_error "请在detector_lib目录下运行此脚本"
    exit 1
fi

# 设置打包信息
PACKAGE_NAME="yolov8_detector_lib_rk3588"
VERSION="v1.0.3"
DATE=$(date +%Y%m%d)
ARCHIVE_NAME="${PACKAGE_NAME}_${VERSION}_${DATE}.tar.gz"

# 创建临时打包目录
TEMP_DIR="/tmp/${PACKAGE_NAME}_${DATE}"
PACKAGE_DIR="${TEMP_DIR}/detector_lib"

log_info "准备打包目录: ${PACKAGE_DIR}"

# 清理旧的临时目录
rm -rf "${TEMP_DIR}"
mkdir -p "${PACKAGE_DIR}"

# 复制必要文件
log_info "复制核心文件..."

# 1. 脚本和核心文档
cp -v build_and_install.sh "${PACKAGE_DIR}/"
cp -v test.sh "${PACKAGE_DIR}/"
cp -v README.md "${PACKAGE_DIR}/"
cp -v CMakeLists.txt "${PACKAGE_DIR}/"
cp -v detector_lib.pc.in "${PACKAGE_DIR}/"

# 用户指南(如果存在)
if [ -f "USER_GUIDE.md" ]; then
    cp -v USER_GUIDE.md "${PACKAGE_DIR}/"
    log_success "已包含用户指南"
fi

# 文档目录 (如果存在)
if [ -d "docs" ]; then
    log_info "复制文档目录..."
    cp -r docs "${PACKAGE_DIR}/"
fi

# 2. 头文件
log_info "复制头文件..."
mkdir -p "${PACKAGE_DIR}/include"
cp -v include/*.h "${PACKAGE_DIR}/include/"

# 3. 源代码
log_info "复制源代码..."
mkdir -p "${PACKAGE_DIR}/src/internal"
cp -v src/*.cpp "${PACKAGE_DIR}/src/" 2>/dev/null || echo "  - 无.cpp文件"
cp -v src/*.c "${PACKAGE_DIR}/src/" 2>/dev/null || echo "  - 无.c文件"
cp -v src/internal/*.h "${PACKAGE_DIR}/src/internal/" 2>/dev/null || echo "  - 内部无头文件"
cp -v src/internal/*.cpp "${PACKAGE_DIR}/src/internal/" 2>/dev/null || echo "  - 内部无.cpp文件"

# 4. 示例程序源码和可执行文件
log_info "复制示例程序..."
mkdir -p "${PACKAGE_DIR}/examples"
mkdir -p "${PACKAGE_DIR}/bin"

# 复制示例程序源码
cp -v examples/*.cpp "${PACKAGE_DIR}/examples/"
cp -v examples/CMakeLists.txt "${PACKAGE_DIR}/examples/"

# 复制编译后的示例程序到bin目录
if [ -d "build/examples" ]; then
    for exe in build/examples/*; do
        if [ -x "$exe" ] && [ -f "$exe" ]; then
            cp -v "$exe" "${PACKAGE_DIR}/bin/"
        fi
    done
    log_success "已包含可执行示例程序"
else
    log_warning "未找到编译的示例程序，请先运行 make"
fi

# 5. 模型文件
log_info "复制模型文件..."
mkdir -p "${PACKAGE_DIR}/models"
if [ -f "models/Q_yolov8_pose.rknn" ]; then
    cp -v models/Q_yolov8_pose.rknn "${PACKAGE_DIR}/models/"
else
    log_warning "姿态检测模型未找到"
fi

if [ -f "models/Q_Rim_Basketball_724_JZ.rknn" ]; then
    cp -v models/Q_Rim_Basketball_724_JZ.rknn "${PACKAGE_DIR}/models/"
else
    log_warning "篮筐篮球检测模型未找到"
fi

# 6. 测试图片
log_info "复制测试图片..."
mkdir -p "${PACKAGE_DIR}/imgs"
if [ -d "imgs" ]; then
    cp -v imgs/*.jpg "${PACKAGE_DIR}/imgs/" 2>/dev/null || echo "  - 无测试图片"
fi

# 7. 标定数据文件 (极坐标功能)
log_info "复制标定数据文件..."
mkdir -p "${PACKAGE_DIR}/data"
if [ -d "data" ]; then
    cp -v data/*.json "${PACKAGE_DIR}/data/" 2>/dev/null || echo "  - 无标定数据文件"
    if [ -f "data/2025_8_6_1280_720.json" ]; then
        log_success "已包含极坐标标定文件"
    fi
fi

# 8. 库文件（重要：包含我们编译的库和RKNN运行时库）
log_info "复制库文件..."
mkdir -p "${PACKAGE_DIR}/lib"

# 复制我们编译的库文件
if [ -f "build/libdetector_lib.so" ]; then
    cp -v build/libdetector_lib.so* "${PACKAGE_DIR}/lib/" 2>/dev/null || echo "  - 部分符号链接复制失败（正常）"
    log_success "已包含检测器动态库"
else
    log_error "未找到编译的动态库，请先运行 make"
    exit 1
fi

if [ -f "build/libdetector_lib.a" ]; then
    cp -v build/libdetector_lib.a "${PACKAGE_DIR}/lib/"
    log_success "已包含检测器静态库"
fi

# 查找并复制RKNN运行时库
RKNN_LIB_PATH="../libs/librknnrt.so"
if [ -f "${RKNN_LIB_PATH}" ]; then
    cp -v "${RKNN_LIB_PATH}" "${PACKAGE_DIR}/lib/"
    log_success "已包含RKNN运行时库"
else
    log_warning "未找到RKNN运行时库，请手动添加"
fi

# 9. 创建快速开始文档
log_info "创建快速开始文档..."
cat > "${PACKAGE_DIR}/QUICK_START.txt" << 'EOF'
YOLOv8检测器库 v1.0.3 - 快速开始
===============================

🚀 新功能: 极坐标系统! 同时输出笛卡尔坐标(x,y)和极坐标(r,θ)
🔧 新特性: 相对路径机制! 解压即用，无需复杂配置

1. 快速开始 (推荐):
   # 直接运行预编译的示例程序
   cd bin/
   ./pose_image_with_polar    # 极坐标演示
   ./pose_image               # 基础姿态检测  
   ./rim_basketball_image     # 篮筐检测
   
2. 开发模式:
   ./build_and_install.sh     # 重新编译安装
   ./test.sh                  # 功能测试

4. 查看结果:
   查看生成的 *_result.jpg 图片
   
   极坐标示例输出:
   笛卡尔坐标: (35.2, 3929.4)mm
   极坐标: 距离=3929.5mm, 角度=89.5°

5. 标定数据:
   data/2025_8_6_1280_720.json - 包含极坐标配置
   
更多详情请查看 README.md 和 docs/DetectorAPI_Usage.md
EOF

# 10. 创建.gitignore（可选）
cat > "${PACKAGE_DIR}/.gitignore" << 'EOF'
build/
*.o
*.a
*.so
*_result.jpg
*.log
.vscode/
.idea/
EOF

# 设置脚本权限
log_info "设置可执行权限..."
chmod +x "${PACKAGE_DIR}/build_and_install.sh"
chmod +x "${PACKAGE_DIR}/test.sh"

# 创建压缩包
log_info "创建压缩包..."
cd "${TEMP_DIR}"
tar -czf "${ARCHIVE_NAME}" detector_lib/

# 计算文件大小
SIZE=$(du -h "${ARCHIVE_NAME}" | cut -f1)

# 移动到当前目录
mv "${ARCHIVE_NAME}" "${OLDPWD}/"
cd "${OLDPWD}"

# 清理临时目录
rm -rf "${TEMP_DIR}"

# 显示打包结果
log_success "打包完成!"
echo ""
echo "📦 压缩包信息:"
echo "   文件名: ${ARCHIVE_NAME}"
echo "   大小: ${SIZE}"
echo "   位置: $(pwd)/${ARCHIVE_NAME}"
echo ""

# 验证压缩包内容
log_info "压缩包内容预览:"
tar -tzf "${ARCHIVE_NAME}" | head -20
echo "   ... (更多文件)"
echo ""

# 生成MD5校验和
log_info "生成校验和..."
md5sum "${ARCHIVE_NAME}" > "${ARCHIVE_NAME}.md5"
log_success "MD5: $(cat ${ARCHIVE_NAME}.md5)"

# 用户提示
echo ""
log_info "📋 用户使用步骤:"
echo "   1. 解压: tar -xzf ${ARCHIVE_NAME}"
echo "   2. 进入: cd detector_lib"
echo "   3. 编译: ./build_and_install.sh"
echo "   4. 测试: ./test.sh"
echo ""
log_info "📧 发送给用户时包含:"
echo "   - ${ARCHIVE_NAME} (主文件)"
echo "   - ${ARCHIVE_NAME}.md5 (校验文件)"
echo ""

log_success "打包脚本执行完成! 🎉"