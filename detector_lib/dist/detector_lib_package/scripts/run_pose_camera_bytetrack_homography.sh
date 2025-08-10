#!/usr/bin/env bash
set -euo pipefail

# 简介:
# - 运行 detector_lib 的摄像头示例: pose_camera_bytetrack_homography
# - 自动为模型与标定文件提供默认绝对路径, 可通过参数覆盖
# - 自动设置 LD_LIBRARY_PATH 以加载 librknnrt.so
#
# 用法:
#   ./run_pose_camera_bytetrack_homography.sh [camera_index] [model_path] [calibration_path]
#   示例:
#     ./run_pose_camera_bytetrack_homography.sh 0
#     ./run_pose_camera_bytetrack_homography.sh 0 /abs/path/Q_yolov8_pose.rknn /abs/path/2025_8_6_1280_720.json

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
BIN_PATH="$BUILD_DIR/examples/pose_camera_bytetrack_homography"

# 默认路径(绝对路径)
DEFAULT_MODEL="$REPO_ROOT/models/Q_yolov8_pose.rknn"
DEFAULT_CALIB="$REPO_ROOT/data/2025_8_6_1280_720.json"

# 参数解析
CAM_INDEX="${1:-0}"
MODEL_PATH="${2:-$DEFAULT_MODEL}"
CALIB_PATH="${3:-$DEFAULT_CALIB}"

# 运行前检查
if [[ ! -x "$BIN_PATH" ]]; then
  echo "[ERR] 未找到可执行文件: $BIN_PATH" >&2
  echo "请先编译: (cd $BUILD_DIR && cmake .. && make -j2)" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[WARN] 模型文件不存在: $MODEL_PATH" >&2
  echo "将尝试继续运行, 但程序可能无法加载模型。" >&2
fi

if [[ ! -f "$CALIB_PATH" ]]; then
  echo "[WARN] 标定文件不存在: $CALIB_PATH" >&2
  echo "将不显示地面与极坐标信息。你可以提供第三个参数指定标定文件。" >&2
fi

# 动态库路径 (确保可找到 librknnrt.so)
export LD_LIBRARY_PATH="$REPO_ROOT/libs:${LD_LIBRARY_PATH:-}"

# 平台信息输出(便于排查)
echo "[INFO] Platform: $(uname -a)"
echo "[INFO] Repo: $REPO_ROOT"
echo "[INFO] Binary: $BIN_PATH"
echo "[INFO] Camera Index: $CAM_INDEX"
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Calibration: $CALIB_PATH"
echo "[INFO] LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# 运行
exec "$BIN_PATH" "$MODEL_PATH" "$CAM_INDEX" "$CALIB_PATH"


