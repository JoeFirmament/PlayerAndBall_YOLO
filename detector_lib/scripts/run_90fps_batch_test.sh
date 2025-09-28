#!/bin/bash

# 90fps批量推理测试脚本
# 用于快速验证高帧率批量推理功能

echo "=== RIM Basketball 90fps批量推理测试 ==="
echo "目标：验证90fps摄像头批量推理，提升NPU利用率"

# 检查摄像头设备
CAMERA_ID=2
if [ "$#" -ge 1 ]; then
    CAMERA_ID=$1
fi

echo "检查摄像头设备 /dev/video$CAMERA_ID ..."
if [ ! -e "/dev/video$CAMERA_ID" ]; then
    echo "❌ 错误: 摄像头设备 /dev/video$CAMERA_ID 不存在"
    echo "可用摄像头设备:"
    ls /dev/video* 2>/dev/null || echo "未找到摄像头设备"
    exit 1
fi

# 检查v4l2-utils工具
if command -v v4l2-ctl >/dev/null 2>&1; then
    echo "📷 摄像头能力检查:"
    echo "支持的格式:"
    v4l2-ctl --device=/dev/video$CAMERA_ID --list-formats-ext | grep -E "(MJPG|1280x960|90\.000)" || true
else
    echo "⚠️  未安装v4l2-utils，跳过摄像头能力检查"
    echo "安装命令: sudo apt install v4l-utils"
fi

# 编译程序
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/../build"

echo "📁 项目路径: $PROJECT_ROOT"
echo "🔧 编译程序..."

cd "$PROJECT_ROOT/.."
mkdir -p build
cd build
cmake .. && make rim_basketball_90fps_batch -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 编译失败！"
    exit 1
fi

# 检查模型文件
MODEL_PATH="../models/Q_Rim_Basketball_724_JZ.rknn"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 找不到模型文件 $MODEL_PATH"
    echo "请确保模型文件存在"
    exit 1
fi

# 检查NPU设备权限
echo "🔧 检查NPU设备权限..."
if [ ! -r "/dev/dri/renderD128" ] 2>/dev/null; then
    echo "⚠️  NPU设备权限可能不足，尝试修复..."
    sudo chmod 666 /dev/dri/renderD* 2>/dev/null || true
fi

echo "✅ 准备工作完成"
echo ""
echo "🚀 启动90fps批量推理测试"
echo "   - 摄像头ID: $CAMERA_ID"
echo "   - 目标配置: 1280x960@90fps MJPEG"
echo "   - 批处理大小: 4帧"
echo "   - NPU核心: 1"
echo ""
echo "📊 日志输出说明:"
echo "   - 实际采集帧率应接近90fps"
echo "   - 推理处理帧率应为采集帧率的90%+"
echo "   - NPU利用率提升应为3-4倍"
echo ""
echo "按Ctrl+C停止测试..."
echo "================================="

# 运行程序
cd examples
./rim_basketball_90fps_batch "$MODEL_PATH" $CAMERA_ID 1

echo ""
echo "📝 测试完成！"
echo "   - 检查生成的.log文件查看详细统计"
echo "   - 关注最终统计中的'推理效率提升'指标"
echo "   - 理想情况下应该看到3-4倍的效率提升"