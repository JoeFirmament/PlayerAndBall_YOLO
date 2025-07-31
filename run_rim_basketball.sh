#!/bin/bash

# 篮筐篮球检测系统启动脚本
# 用途: 单独运行篮筐和篮球检测功能

# 设置模型路径
RIM_MODEL="models/Q_Rim_Basketball_724_JZ.rknn"

# 摄像头设备配置 - 使用持久化路径
RIM_CAMERA_PATH="/dev/v4l/by-id/usb-DECXIN_CAMERA_DECXIN_CAMERA_01.00.00-video-index0"

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "========================================"
echo "      篮筐篮球检测系统 v2.0"
echo "========================================"
echo "检测模型: $RIM_MODEL"
if [ -e "$RIM_CAMERA_PATH" ]; then
    echo "摄像头设备: $RIM_CAMERA_PATH"
else
    echo "摄像头设备: 默认配置 (路径不存在)"
    RIM_CAMERA_PATH=""  # 空值表示使用默认
fi
echo "按键控制:"
echo "  [ESC] - 退出程序"
echo "  [S]   - 截图保存"
echo "========================================"

# 检查模型文件
if [ ! -f "$SCRIPT_DIR/$RIM_MODEL" ]; then
    echo "❌ 篮筐篮球检测模型不存在: $RIM_MODEL"
    exit 1
fi

# 检查可执行文件
if [ ! -f "$BUILD_DIR/rim_basketball_detector_v2" ]; then
    echo "❌ 可执行文件不存在，正在编译..."
    cd "$BUILD_DIR" || exit 1
    make rim_basketball_detector_v2
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败！"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi


# 设置NPU权限
echo "设置NPU设备权限..."
sudo chmod 666 /dev/dri/renderD* 2>/dev/null || true

# 运行程序
cd "$BUILD_DIR" || exit 1

echo "启动篮筐篮球检测系统..."
if [ -n "$RIM_CAMERA_PATH" ]; then
    exec ./rim_basketball_detector_v2 "../$RIM_MODEL" "$RIM_CAMERA_PATH"
else
    exec ./rim_basketball_detector_v2 "../$RIM_MODEL"
fi