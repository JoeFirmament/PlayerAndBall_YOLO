# 平台信息日志

## 当前开发环境
- **操作系统**: Linux 6.1.43-rockchip-rk3588
- **CPU架构**: ARM aarch64 (RK3588)
- **Shell**: /bin/bash
- **工作空间**: /home/orangepi/Qworkspace

## 项目信息
- **项目**: yolov8_pose_basketball
- **模块**: detector_lib
- **构建系统**: CMake
- **目标平台**: ARM64 (aarch64)

## 最新分析 (2024)
- **库文件分析**: detector_lib 同时生成了静态库和动态库
- **依赖**: OpenCV4, RKNN API, pthread
- **架构**: ELF 64-bit LSB shared object for ARM aarch64

## 变更记录 (2025-08-08)
- **GUI兼容性**: 将 `Q_CameraTools/camera_calibration.py` 的UI文本改为纯ASCII（移除Emoji），降低在SSH -Y远程X11环境下对字体的依赖，避免 `X_OpenFont BadValue` 异常。
- **影响范围**: 标签/按钮/选项卡标题与控制台打印的可视文本保持英文ASCII；不影响图像处理与校准逻辑。