#!/bin/bash

# 姿态检测 + 极坐标系统 启动脚本
# 使用1280x720分辨率和对应的标定文件

echo "========================================"
echo "    YOLOv8 姿态检测 + 极坐标系统"
echo "========================================"

# 检查模型文件
MODEL_FILE="models/Q_yolov8_pose.rknn"
if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_FILE"
    exit 1
fi

# 检查标定文件
CALIB_FILE="data/2025_8_6_1280_720.json"
if [ ! -f "$CALIB_FILE" ]; then
    echo "❌ 错误: 标定文件不存在: $CALIB_FILE"
    exit 1
fi

# 检查可执行文件
if [ ! -f "build/yolov8_pose_only" ]; then
    echo "❌ 错误: 程序未编译，请先运行 make"
    echo "编译命令: mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

# 摄像头设备配置
CAMERA_PATH="/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0"

echo "配置信息:"
echo "  模型文件: $MODEL_FILE"
echo "  标定文件: $CALIB_FILE" 
echo "  摄像头设备: $CAMERA_PATH"

# 检查摄像头设备
if [ ! -e "$CAMERA_PATH" ]; then
    echo "⚠️  警告: USB摄像头设备不存在: $CAMERA_PATH"
    echo "程序将使用默认摄像头"
    CAMERA_PATH=""
fi

echo ""
echo "功能说明:"
echo "  • 同时显示笛卡尔坐标 (x,y)mm 和极坐标 (距离mm,角度°)"
echo "  • 紫色圆点: ROI地面定位点"
echo "  • 黄色文字: 笛卡尔坐标"
echo "  • 青色文字: 极坐标"
echo ""
echo "按键控制:"
echo "  [ESC] - 退出程序"
echo "  [T]   - 切换ByteTrack跟踪开关"
echo "========================================"
echo ""

# 启动程序
if [ -n "$CAMERA_PATH" ]; then
    echo "启动命令: ./build/yolov8_pose_only $MODEL_FILE $CALIB_FILE $CAMERA_PATH"
    ./build/yolov8_pose_only "$MODEL_FILE" "$CALIB_FILE" "$CAMERA_PATH"
else
    echo "启动命令: ./build/yolov8_pose_only $MODEL_FILE $CALIB_FILE"
    ./build/yolov8_pose_only "$MODEL_FILE" "$CALIB_FILE"
fi

echo ""
echo "程序已退出"