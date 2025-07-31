#!/bin/bash

# 双摄像头双线程检测系统启动脚本
# 用途: 同时运行姿态检测和篮筐篮球检测

# 设置模型路径
POSE_MODEL="models/Q_yolov8_pose.rknn"
RIM_MODEL="models/Q_Rim_Basketball_724_JZ.rknn"
CALIB_FILE="data/2025_7_11pm.json"

# 摄像头设备配置 - 直接使用持久化路径
POSE_CAMERA_PATH="/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"      # 姿态检测摄像头  
RIM_CAMERA_PATH="/dev/v4l/by-id/usb-DECXIN_CAMERA_DECXIN_CAMERA_01.00.00-video-index0"  # 篮筐篮球检测摄像头

# 检查摄像头设备
if [ ! -e "$POSE_CAMERA_PATH" ]; then
    echo "⚠️  姿态检测摄像头 $POSE_CAMERA_PATH 不存在"
    POSE_CAMERA_PATH=""  # 空值表示使用默认设备
else
    echo "✅ 姿态检测摄像头: $POSE_CAMERA_PATH"
fi

if [ ! -e "$RIM_CAMERA_PATH" ]; then
    echo "⚠️  篮筐检测摄像头 $RIM_CAMERA_PATH 不存在"
    RIM_CAMERA_PATH=""   # 空值表示使用默认设备
else
    echo "✅ 篮筐检测摄像头: $RIM_CAMERA_PATH"
fi

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "========================================="
echo "     双摄像头双线程检测系统 v1.0"
echo "========================================="
if [ -n "$POSE_CAMERA_PATH" ]; then
    echo "姿态检测: $POSE_MODEL (摄像头: $POSE_CAMERA_PATH)"
else
    echo "姿态检测: $POSE_MODEL (默认摄像头)"
fi

if [ -n "$RIM_CAMERA_PATH" ]; then
    echo "篮筐检测: $RIM_MODEL (摄像头: $RIM_CAMERA_PATH)"
else
    echo "篮筐检测: $RIM_MODEL (默认摄像头)"
fi
echo "标定文件: $CALIB_FILE"
echo "========================================="

# 检查模型文件
if [ ! -f "$SCRIPT_DIR/$POSE_MODEL" ]; then
    echo "❌ 姿态检测模型不存在: $POSE_MODEL"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/$RIM_MODEL" ]; then
    echo "❌ 篮筐篮球检测模型不存在: $RIM_MODEL"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/$CALIB_FILE" ]; then
    echo "⚠️  标定文件不存在: $CALIB_FILE，将跳过坐标映射功能"
    CALIB_FILE=""
fi

# 检查可执行文件
if [ ! -f "$BUILD_DIR/dual_camera_detector" ]; then
    echo "❌ 可执行文件不存在，正在编译..."
    cd "$BUILD_DIR" || exit 1
    make dual_camera_detector
    if [ $? -ne 0 ]; then
        echo "❌ 编译失败！"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# 摄像头设备配置已由camera_manager.sh处理
echo "✅ 摄像头配置成功"

# 设置NPU权限
echo "设置NPU设备权限..."
sudo chmod 666 /dev/dri/renderD* 2>/dev/null || true

# 运行程序
cd "$BUILD_DIR" || exit 1

if [ -n "$CALIB_FILE" ]; then
    echo "启动双摄像头检测系统 (带标定)..."
    if [ -n "$POSE_CAMERA_PATH" ] && [ -n "$RIM_CAMERA_PATH" ]; then
        exec ./dual_camera_detector "../$POSE_MODEL" "../$RIM_MODEL" "../$CALIB_FILE" "$POSE_CAMERA_PATH" "$RIM_CAMERA_PATH"
    else
        exec ./dual_camera_detector "../$POSE_MODEL" "../$RIM_MODEL" "../$CALIB_FILE"
    fi
else
    echo "启动双摄像头检测系统 (无标定)..."
    if [ -n "$POSE_CAMERA_PATH" ] && [ -n "$RIM_CAMERA_PATH" ]; then
        exec ./dual_camera_detector "../$POSE_MODEL" "../$RIM_MODEL" "" "$POSE_CAMERA_PATH" "$RIM_CAMERA_PATH"
    else
        exec ./dual_camera_detector "../$POSE_MODEL" "../$RIM_MODEL" ""
    fi
fi