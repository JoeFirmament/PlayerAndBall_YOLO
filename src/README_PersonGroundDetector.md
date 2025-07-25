# PersonGroundDetector - 人物地面坐标检测器

## 概述

PersonGroundDetector 是一个完整的人物地面坐标检测模块，专为篮球场景设计。该模块集成了YOLOv8姿态估计、RK3588 NPU优化、ByteTrack多目标跟踪、坐标映射和平滑滤波等功能。

## 核心功能

### 🎯 检测与跟踪
- **YOLOv8 Pose模型**: 17个关键点的人体姿态估计
- **RK3588 NPU**: 零拷贝优化，高性能推理
- **ByteTrack**: 多目标跟踪，保持身份一致性
- **双模式支持**: 检测模式和跟踪模式可切换

### 📍 坐标映射
- **ROI方法**: 基于检测框下沿中点
- **脚踝方法**: 基于关键点脚踝位置+动态偏移
- **Homography变换**: 图像坐标→真实世界坐标
- **平滑滤波**: 5帧窗口平滑，减少抖动

### ⚡ 性能优化
- **零拷贝内存**: 避免CPU↔NPU数据传输
- **letterbox预处理**: 直接写入NPU内存
- **线程安全**: 多线程环境下稳定运行

## 快速开始

### 1. 依赖库
```bash
# 必需依赖
- OpenCV 4.x (aarch64)
- RKNN Runtime 2.x
- libjsoncpp
- pthread

# 可选依赖
- Eigen3 (矩阵运算)
```

### 2. 编译
```bash
# 使用CMake编译
mkdir build && cd build
cmake .. -f ../CMakeLists_person_detector.txt
make -j$(nproc)

# 或者手动编译
g++ -std=c++14 -O3 -o example_person_detector \
    example_person_detector.cpp \
    person_ground_detector.cpp \
    yolov8-pose.cc \
    postprocess.cc \
    letterbox_utils.cc \
    BYTETracker.cpp \
    STrack.cpp \
    kalmanFilter.cpp \
    lapjv.cpp \
    utils.cpp \
    -I../3rdparty/rknpu2/Linux/librknn_api/include \
    -I../3rdparty/opencv/opencv-linux-aarch64/include \
    -L../3rdparty/rknpu2/Linux/librknn_api/aarch64 \
    -L../3rdparty/opencv/opencv-linux-aarch64/lib \
    -lrknnrt -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_highgui \
    -ljsoncpp -lpthread
```

### 3. 运行示例
```bash
# 基本用法
./example_person_detector ../models/Q_yolov8_pose.rknn

# 完整参数
./example_person_detector ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json 0

# 按键控制
ESC - 退出程序
T   - 切换跟踪模式
空格 - 暂停/继续
```

## API 使用

### 基础使用
```cpp
#include "person_ground_detector.h"

int main() {
    // 1. 创建检测器实例
    PersonGroundDetector detector(
        "../models/Q_yolov8_pose.rknn",     // 模型路径
        "../data/2025_7_11pm.json",        // 标定文件
        true                               // 启用跟踪
    );
    
    // 2. 初始化
    if (detector.initialize() != 0) {
        printf("初始化失败\n");
        return -1;
    }
    
    // 3. 处理图像
    cv::Mat frame;
    std::vector<PersonGroundResult> results;
    
    int person_count = detector.detectPersons(frame, results);
    
    // 4. 使用结果
    for (const auto& result : results) {
        printf("ID:%d 地面坐标:(%.0f,%.0f)mm\n", 
               result.track_id, 
               result.ground_ankle.x, 
               result.ground_ankle.y);
    }
    
    // 5. 绘制结果
    detector.drawResults(frame, results);
    
    return 0;
}
```

### 高级功能
```cpp
// 动态切换跟踪模式
detector.setTrackingEnabled(false);  // 关闭跟踪
detector.setTrackingEnabled(true);   // 开启跟踪

// 获取性能统计
float inference_time, total_time;
detector.getPerformanceStats(inference_time, total_time);
printf("推理时间: %.1fms, 总时间: %.1fms\n", inference_time, total_time);

// 检查跟踪状态
if (detector.isTrackingEnabled()) {
    printf("跟踪模式已启用\n");
}
```

## 数据结构

### PersonGroundResult
```cpp
struct PersonGroundResult {
    int track_id;                        // 跟踪ID
    cv::Rect bbox;                       // 检测框
    cv::Point2f ground_roi;              // ROI方法地面坐标(mm)
    cv::Point2f ground_ankle;            // 脚踝+偏移方法地面坐标(mm)
    float vertical_offset;               // 垂直偏移量(px)
    float confidence;                    // 检测置信度
    std::vector<cv::Point2f> keypoints;  // 17个关键点
    bool is_tracked;                     // 是否来自跟踪
};
```

### 关键点索引（COCO格式）
```cpp
0: 鼻子        1: 左眼        2: 右眼        3: 左耳        4: 右耳
5: 左肩        6: 右肩        7: 左肘        8: 右肘        9: 左腕
10: 右腕       11: 左髋       12: 右髋       13: 左膝       14: 右膝
15: 左脚踝     16: 右脚踝
```

## 配置文件

### Homography标定文件格式
```json
{
    "timestamp": "2025-07-11T16:54:46.584028",
    "points": [
        {
            "pixel": [999.1, 1076.2],
            "world": [0.0, 1820.0],
            "id": 0
        }
    ],
    "matrix": [
        [-2.460, 0.012, 2462.065],
        [0.075, 0.764, -2805.956],
        [0.0001, -0.002, 1.0]
    ],
    "point_count": 7
}
```

## 算法原理

### 动态垂直偏移
```cpp
人物高度像素 = |鼻子y - 脚踝y|
标准化比例 = 人物高度像素 / 640
垂直偏移量 = 标准化比例 × 25像素
最终限制在 [5, 40] 像素范围内
```

### 平滑滤波
```cpp
使用5帧窗口的移动平均滤波：
smoothed_point = (p1 + p2 + p3 + p4 + p5) / 5
```

## 性能指标

### RK3588平台性能
- **推理时间**: ~15ms (单人)
- **总处理时间**: ~25ms (包括预处理、后处理)
- **内存占用**: ~100MB
- **帧率**: 30+ FPS (1920x1080输入)

### 精度指标
- **检测准确率**: >95% (正常光照)
- **跟踪稳定性**: >90% (无遮挡)
- **坐标映射误差**: <50mm (标定范围内)

## 常见问题

### Q: 为什么地面坐标显示(-1, -1)？
A: 检查Homography标定文件是否存在且格式正确。

### Q: 跟踪ID经常变化？
A: 调整ByteTracker参数，增大track_buffer值。

### Q: 检测性能不佳？
A: 确保运行时有root权限访问NPU设备。

### Q: 编译错误？
A: 检查依赖库路径是否正确，确保OpenCV和RKNN版本匹配。

## 扩展开发

### 添加新的坐标计算方法
```cpp
// 在calculateFootPosition中添加新方法
cv::Point2f new_method = calculateNewMethod(keypoints);
result.new_method = new_method;
```

### 自定义平滑滤波器
```cpp
// 修改SMOOTH_WINDOW_SIZE或实现新的滤波算法
cv::Point2f customSmoothFilter(const cv::Point2f& point, int track_id);
```

### 集成其他跟踪算法
```cpp
// 替换ByteTracker为其他跟踪器
class CustomTracker {
    // 实现自定义跟踪逻辑
};
```

## 许可证

本项目遵循MIT许可证。

## 支持

如有问题或建议，请联系开发团队。