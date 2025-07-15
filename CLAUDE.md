# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

使用CMake构建系统，支持RK3588平台的双NPU核心优化：

```bash
# 编译项目
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 运行主程序（使用默认标定文件）
./yolov8_pose_basketball ../models/Q_yolov8_pose.rknn

# 运行主程序（指定标定文件）
./yolov8_pose_basketball ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json

# 运行测试程序
./test_basketball_rknn
```

## 核心架构

### 双线程架构
- **主线程**: 使用NPU1进行YOLOv8姿态检测，零拷贝优化
- **副线程**: 使用NPU2进行篮球检测，独立运行避免资源竞争
- **线程安全**: 使用queue + mutex + condition_variable进行线程间通信

### 零拷贝优化
- NPU内存直接访问，消除CPU↔NPU数据拷贝开销
- 性能相比基础版本提升100%
- 关键函数：`init_zero_copy_mem()`, `optimized_letterbox_to_npu()`

### 关键组件

1. **姿态检测模块** (`src/yolov8-pose.cc`, `src/postprocess.cc`)
   - YOLOv8 pose estimation
   - 17个关键点检测（COCO格式）
   - 零拷贝推理优化

2. **篮球检测模块** (`src/basketball_postprocess.cpp`)
   - 专门的2类检测（player, basketball）
   - 独立线程运行在NPU2
   - 置信度阈值：0.5

3. **多目标跟踪** (`src/BYTETracker.cpp`, `src/STrack.cpp`)
   - ByteTrack算法实现
   - 卡尔曼滤波器用于状态估计
   - 匈牙利算法用于数据关联

4. **坐标映射系统**
   - Homography变换：图像坐标→真实世界坐标
   - 标定数据存储在JSON文件中
   - 支持篮球场地实际位置测量

## 数据流程

1. **图像采集**: 摄像头 → 1920x1080 MJPEG格式
2. **预处理**: letterbox resize → 640x640，直接写入NPU内存
3. **推理**: 
   - 主线程：姿态检测（NPU1）
   - 副线程：篮球检测（NPU2）
4. **后处理**: 
   - 关键点提取和骨架绘制
   - 多目标跟踪
   - 坐标映射转换
5. **结果融合**: 合并两个线程的检测结果进行显示

## 重要文件

- `src/main_camera_optimized.cc`: 主程序，包含完整的双线程架构
- `data/2025_7_11pm.json`: Homography标定数据
- `models/`: RKNN模型文件目录
  - `Q_yolov8_pose.rknn`: 姿态检测模型
  - `Q_Player_Ball_8n_4090_Drun_500E.rknn`: 篮球检测模型

## 性能优化特性

- **零拷贝内存管理**: 避免CPU-NPU数据传输开销
- **双NPU并行**: 充分利用RK3588双核NPU
- **MJPEG格式**: 降低摄像头采集延迟
- **V4L2驱动**: 避免GStreamer开销，提升帧率

## 依赖库

- OpenCV 4.x (aarch64)
- RKNN Runtime 2.x
- RGA (硬件加速图像处理)
- Eigen3 (矩阵运算)
- libturbojpeg (JPEG解码)

## 调试和测试

- 信号处理：SIGINT优雅退出
- 性能统计：FPS计算和延迟分析
- 可视化调试：关键点、跟踪框、坐标映射可视化
- 多线程安全：互斥锁保护共享资源

## 按键控制

程序运行时支持以下按键控制：

- **ESC键**: 退出程序
- **T键**: 切换ByteTrack多目标跟踪功能开关
  - 开启时：显示绿色跟踪框和ID号码
  - 关闭时：显示蓝色检测框和置信度
- **B键**: 切换篮球检测功能开关
  - 开启时：显示红色篮球检测框
  - 关闭时：不显示篮球检测结果，节省NPU2资源

## 注意事项

- 确保摄像头支持1920x1080 MJPEG格式
- Homography标定文件必须存在且格式正确
- 模型文件路径必须正确，支持相对和绝对路径
- 运行时需要root权限访问NPU设备
- 按键控制为实时切换，可以在运行过程中随时调整功能