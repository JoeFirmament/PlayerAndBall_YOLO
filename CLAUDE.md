# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

使用CMake构建系统，支持RK3588平台的双摄像头独立检测架构：

## 快速启动脚本 (推荐)

```bash
# 主启动菜单 - 一键选择运行模式
./start.sh

# 直接启动脚本
./run_dual_camera.sh      # 双摄像头双线程检测 (推荐)
./run_pose_only.sh        # 纯姿态检测
./run_rim_basketball.sh   # 篮筐篮球检测
```

## 手动编译和运行

```bash
# 编译项目
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 新架构 - 双摄像头独立检测
./yolov8_pose_only ../models/Q_yolov8_pose.rknn                    # 纯姿态检测 (摄像头1)
./yolov8_pose_only ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json  # 指定标定文件

# 篮筐篮球检测程序 (两个版本可选)
./rim_basketball_detector_v2 ../models/Q_Rim_Basketball_724_JZ.rknn      # 复杂版 (基于modern_dual_comparator.py)
./rim_basketball_detector_simple ../models/Q_Rim_Basketball_724_JZ.rknn  # 简化版 (基于ref/rknn_yolov8_ref)
./rim_basketball_detector_simple ../models/Q_Rim_Basketball_724_JZ.rknn 0 # 指定摄像头设备

# 双摄像头双线程检测系统 (推荐)
./dual_camera_detector ../models/Q_yolov8_pose.rknn ../models/Q_Rim_Basketball_724_JZ.rknn
./dual_camera_detector ../models/Q_yolov8_pose.rknn ../models/Q_Rim_Basketball_724_JZ.rknn ../data/2025_7_11pm.json 0 2

# 原有版本 (兼容保留)
./yolov8_pose_basketball ../models/Q_yolov8_pose.rknn           # 原双线程版本
./test_basketball_rknn                                          # 篮球检测测试程序
./rga_resize_test                                               # RGA性能测试
```

## 核心架构

### 双摄像头双线程检测架构 (推荐)
- **程序**: `dual_camera_detector` - 集成检测系统
  - **线程1**: 姿态检测 (摄像头0, NPU1)
    - 模型: `Q_yolov8_pose.rknn`
    - 功能: YOLOv8姿态检测 + ByteTrack跟踪 + 坐标映射
  - **线程2**: 篮筐篮球检测 (摄像头2, NPU2)
    - 模型: `Q_Rim_Basketball_724_JZ.rknn`
    - 功能: 篮筐检测 + 篮球检测 + ROI分析
  - **显示**: 支持拼接显示或分别显示模式

### 双摄像头独立检测架构 (调试用)
- **程序1**: `yolov8_pose_only` - 纯姿态检测系统
  - 摄像头: 主摄像头 (`/dev/video0`)
  - 模型: `Q_yolov8_pose.rknn`
  - 功能: 姿态检测 + ByteTrack跟踪 + 坐标映射
  - NPU: 单线程，使用NPU资源进行姿态推理

- **程序2**: `rim_basketball_detector_v2` - 篮筐篮球检测系统
  - 摄像头: 副摄像头 (`/dev/video2` 或指定设备)
  - 模型: `Q_Rim_Basketball_724_JZ.rknn`
  - 功能: 篮筐检测 + 篮球检测 + ROI分析
  - NPU: 独立使用NPU资源进行目标检测

### 零拷贝优化
- NPU内存直接访问，消除CPU↔NPU数据拷贝开销
- 性能相比基础版本提升100%
- 关键函数：`init_zero_copy_mem()`, `letterbox_resize_to_npu()`

### 关键组件

1. **姿态检测模块** (`src/main_pose_only.cc`, `src/postprocess.cc`)
   - YOLOv8 pose estimation
   - 17个关键点检测（COCO格式）
   - 零拷贝推理优化，移除篮球检测功能

2. **篮筐篮球检测模块** (`src/rim_basketball_detector_updated.cc`)
   - 2类检测：rim(篮筐), basketball(篮球)
   - 基于`modern_dual_comparator.py`验证的后处理逻辑
   - ROI位置分析和距离计算

3. **多目标跟踪** (`src/BYTETracker.cpp`, `src/STrack.cpp`)
   - ByteTrack算法实现，仅用于姿态检测程序
   - 卡尔曼滤波器用于状态估计
   - 匈牙利算法用于数据关联

4. **坐标映射系统**
   - Homography变换：图像坐标→真实世界坐标
   - 标定数据存储在JSON文件中
   - 仅用于姿态检测程序的球员位置测量

### 架构优势
- **资源无冲突**: 两个程序独立运行，避免NPU资源竞争
- **功能专业化**: 每个程序专注于特定检测任务，提高准确性
- **易于调试**: 可独立调试和优化每个检测模块
- **灵活部署**: 可根据需要选择运行单个或多个程序

## 数据流程

### 姿态检测程序流程 (`yolov8_pose_only`)
1. **图像采集**: 主摄像头 → 1920x1080 MJPEG格式
2. **预处理**: letterbox resize → 640x640，直接写入NPU内存
3. **推理**: 使用NPU进行YOLOv8姿态检测
4. **后处理**: 
   - 关键点提取和骨架绘制
   - ByteTrack多目标跟踪
   - Homography坐标映射
5. **显示**: 实时显示姿态检测和跟踪结果

### 篮筐篮球检测程序流程 (`rim_basketball_detector_v2`)
1. **图像采集**: 副摄像头 → 1920x1080 MJPEG格式
2. **预处理**: letterbox resize → 640x640，直接写入NPU内存
3. **推理**: 使用NPU进行篮筐和篮球检测
4. **后处理**: 
   - 基于`modern_dual_comparator.py`的DFL解码
   - NMS处理和坐标转换
   - ROI分析和距离计算
5. **显示**: 实时显示检测框、ROI信息和统计数据

## 重要文件

### 新架构文件
- `src/dual_camera_detector.cc`: 双摄像头双线程集成检测系统 (推荐)
- `src/main_pose_only.cc`: 纯姿态检测程序，单线程零拷贝优化
- `src/rim_basketball_detector_updated.cc`: 篮筐篮球检测程序，独立运行
- `src/rim_basketball_postprocess_simple.cpp`: 篮筐篮球后处理模块 (基于ref/rknn_yolov8_ref)
- `include/rim_basketball_postprocess.h`: 篮筐篮球检测接口定义
- `src/letterbox_utils.cc`: 零拷贝letterbox预处理工具

### 模型文件
- `models/Q_yolov8_pose.rknn`: 姿态检测模型
- `models/Q_Rim_Basketball_724_JZ.rknn`: 篮筐篮球检测模型 (新)

### 配置和验证文件
- `data/2025_7_11pm.json`: Homography标定数据
- `ref/modern_dual_comparator.py`: 篮筐篮球模型后处理验证工具
- `todo_multiframe_fusion.md`: 多帧融合开发计划

### 原有架构文件 (兼容保留)
- `src/main_camera_optimized.cc`: 原双线程架构主程序
- `src/basketball_postprocess.cpp`: 原篮球检测后处理模块

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

### 编译时调试
```bash
# 清理重新编译
cd build && make clean && cd .. && rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)

# 检查NPU模型兼容性
file ../models/*.rknn
```

### 运行时调试
- 信号处理：SIGINT优雅退出
- 性能统计：FPS计算和延迟分析  
- 可视化调试：关键点、跟踪框、坐标映射可视化
- 多线程安全：互斥锁保护共享资源

### 权限设置
```bash
# NPU设备权限
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video $USER
```

## 按键控制

### 姿态检测程序 (`yolov8_pose_only`)
- **ESC键**: 退出程序
- **T键**: 切换ByteTrack多目标跟踪功能开关
  - 开启时：显示绿色跟踪框和ID号码
  - 关闭时：显示蓝色检测框和置信度

### 双摄像头双线程程序 (`dual_camera_detector`)
- **ESC键**: 退出程序
- **T键**: 切换ByteTrack多目标跟踪功能开关
- **C键**: 切换显示模式 (拼接显示/分别显示)
  - 拼接显示：两个摄像头结果水平拼接，单窗口显示
  - 分别显示：两个摄像头结果分别显示在独立窗口

### 篮筐篮球检测程序 (`rim_basketball_detector_v2`)
- **ESC键**: 退出程序
- **S键**: 截图保存当前检测结果
  - 保存格式：`screenshot_XXXX.jpg`
  - 包含检测框、置信度和ROI分析信息

## 性能调试与分析

### 性能测试工具
- `test_rga_performance.sh`: RGA硬件加速性能测试脚本  
- `./rga_resize_test`: RGA resize性能基准测试
- 内置FPS统计：主程序实时显示帧率和延迟

### 调试模式
程序内置多种调试输出：
- NPU内存分配状态
- 线程队列长度监控
- 坐标映射转换结果
- 检测框置信度分布

### USB摄像头设备管理

项目使用持久化USB设备路径，解决设备重启后设备号变化的问题：

#### 设备配置
- 姿态检测摄像头: `/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0`
- 篮筐检测摄像头: `/dev/v4l/by-id/usb-DECXIN_CAMERA_DECXIN_CAMERA_01.00.00-video-index0`

#### 自动回退机制
启动脚本会自动：
1. 检查USB摄像头的by-id路径是否存在
2. 如果设备不存在，自动使用C++程序的默认配置
3. 显示设备状态信息

### 常见问题排查
- **NPU权限**: 需要root权限或加入video用户组
- **模型版本**: 确保使用RK3588对应的.rknn模型
- **内存不足**: 检查NPU内存分配，调整队列大小
- **摄像头兼容性**: 验证V4L2设备支持MJPEG格式
- **设备路径**: 使用 `ls /dev/v4l/by-id/` 检查USB摄像头设备路径

## 待实现功能

基于`todo_multiframe_fusion.md`的开发计划：
- **多帧融合**: 滑动窗口平均、置信度投票、连续帧确认
- **轨迹平滑**: 结合ByteTrack进行目标一致性优化  
- **去抖动**: 误检过滤和轨迹稳定化算法

## 注意事项

- 确保摄像头支持1920x1080 MJPEG格式
- Homography标定文件必须存在且格式正确
- 模型文件路径必须正确，支持相对和绝对路径
- 运行时需要root权限访问NPU设备
- 按键控制为实时切换，可以在运行过程中随时调整功能