#!/bin/bash

# 纯姿态检测系统启动脚本
# 用途: 单独运行姿态检测和跟踪功能

# 设置模型路径
POSE_MODEL="models/Q_yolov8_pose.rknn"
CALIB_FILE="data/2025_7_11pm.json"

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "========================================"
echo "        纯姿态检测系统 v1.0"
echo "========================================"
echo "姿态模型: $POSE_MODEL"
echo "标定文件: $CALIB_FILE"
echo "按键控制:"
echo "  [ESC] - 退出程序"
echo "  [T]   - 切换ByteTrack跟踪开关"
echo "========================================"

# 检查模型文件
if [ ! -f "$SCRIPT_DIR/$POSE_MODEL" ]; then
    echo "❌ 姿态检测模型不存在: $POSE_MODEL"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/$CALIB_FILE" ]; then
    echo "⚠️  标定文件不存在: $CALIB_FILE，将跳过坐标映射功能"
    CALIB_FILE=""
fi

# 检查可执行文件
if [ ! -f "$BUILD_DIR/yolov8_pose_only" ]; then
    echo "❌ 可执行文件不存在，正在编译..."
    cd "$BUILD_DIR" || exit 1
    make yolov8_pose_only
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

if [ -n "$CALIB_FILE" ]; then
    echo "启动姿态检测系统 (带标定)..."
    exec ./yolov8_pose_only "../$POSE_MODEL" "../$CALIB_FILE"
else
    echo "启动姿态检测系统 (无标定)..."
    exec ./yolov8_pose_only "../$POSE_MODEL"
fi