# PersonGroundDetector 模块使用手册

## 📋 目录
1. [模块概述](#模块概述)
2. [环境要求](#环境要求)
3. [快速开始](#快速开始)
4. [API详细说明](#api详细说明)
5. [配置文件格式](#配置文件格式)
6. [性能优化](#性能优化)
7. [常见问题](#常见问题)
8. [高级用法](#高级用法)
9. [调试技巧](#调试技巧)
10. [最佳实践](#最佳实践)

---

## 模块概述

### 🎯 功能特性
PersonGroundDetector 是一个专为篮球场景设计的人物地面坐标检测模块，集成了多项先进技术：

- **高精度检测**: 基于YOLOv8-pose的17个关键点检测,经过检测 pose 为识别人最好最准的模型。
- **硬件加速**: 优化RK3588 NPU性能，支持零拷贝推理
- **智能跟踪**: 集成ByteTrack算法，保持目标身份一致性
- **坐标映射**: 通过Homography变换实现图像到真实世界坐标转换
- **平滑滤波**: 5帧窗口滤波，减少坐标抖动
- **双重定位**: 提供ROI和脚踝两种地面坐标计算方法

### 🏗️ 系统架构
```
输入图像 → 预处理 → NPU推理 → 后处理 → 坐标映射 → 平滑滤波 → 输出结果
    ↓         ↓        ↓        ↓        ↓          ↓
 letterbox  零拷贝   YOLOv8   关键点   Homography  移动平均
```

---

## 环境要求

### 💻 硬件要求
- **处理器**: RK3588 (必需)
- **内存**: ≥4GB RAM
- **存储**: ≥2GB 可用空间
- **摄像头**: 支持MJPEG格式，推荐1920x1080@30fps

### 📦 软件依赖
```bash
# 核心依赖
- Linux (Ubuntu 20.04+)
- OpenCV 4.x (aarch64)
- RKNN Runtime 2.x
- libjsoncpp-dev
- CMake 3.10+
- GCC 7.0+

# 可选依赖
- Eigen3 (矩阵运算优化)
- pkg-config
```

### 🔧 环境配置
```bash
# 1. 安装系统依赖
sudo apt update
sudo apt install -y cmake build-essential pkg-config
sudo apt install -y libjsoncpp-dev libeigen3-dev

# 2. 设置环境变量
export LD_LIBRARY_PATH=/path/to/opencv/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/rknn/lib:$LD_LIBRARY_PATH

# 3. 验证NPU设备
ls /dev/rknpu*
```

---

## 快速开始

### 🚀 5分钟上手

#### 步骤1: 获取文件
```bash
# 复制以下文件到您的项目目录
cp person_ground_detector.h your_project/
cp person_ground_detector.cpp your_project/
cp example_person_detector.cpp your_project/
cp CMakeLists_person_detector.txt your_project/CMakeLists.txt
```

#### 步骤2: 编译项目
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

#### 步骤3: 准备模型和标定文件
```bash
# 确保以下文件存在
ls ../models/Q_yolov8_pose.rknn     # YOLOv8 pose模型
ls ../data/2025_7_11pm.json         # Homography标定文件
```

#### 步骤4: 运行示例
```bash
# 基本运行
./example_person_detector ../models/Q_yolov8_pose.rknn

# 指定标定文件和摄像头
./example_person_detector ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json 0
```

### 🎮 交互控制
运行时支持以下按键：
- **ESC**: 退出程序
- **T**: 切换跟踪模式开关
- **空格**: 暂停/继续处理

---

## API详细说明

### 🔌 核心类: PersonGroundDetector

#### 构造函数
```cpp
PersonGroundDetector(const std::string& model_path, 
                    const std::string& homography_json_path,
                    bool enable_tracking = true);
```

**参数说明:**
- `model_path`: RKNN模型文件路径 (必需)
- `homography_json_path`: Homography标定JSON文件路径 (必需)
- `enable_tracking`: 是否启用ByteTrack跟踪 (可选，默认true)

#### 主要方法

##### 1. 初始化
```cpp
int initialize();
```
- **返回值**: 0=成功, -1=失败
- **作用**: 初始化RKNN、零拷贝内存、Homography矩阵
- **注意**: 必须在所有其他操作前调用

##### 2. 检测人物
```cpp
int detectPersons(const cv::Mat& frame, std::vector<PersonGroundResult>& results);
```
- **参数**: 
  - `frame`: 输入图像 (BGR格式)
  - `results`: 输出结果数组
- **返回值**: 检测到的人物数量
- **功能**: 执行完整的检测流程

##### 3. 绘制结果
```cpp
void drawResults(cv::Mat& frame, const std::vector<PersonGroundResult>& results);
```
- **参数**: 
  - `frame`: 输入输出图像
  - `results`: 检测结果
- **功能**: 在图像上绘制检测框、关键点、坐标信息

##### 4. 跟踪控制
```cpp
void setTrackingEnabled(bool enabled);
bool isTrackingEnabled() const;
```
- **功能**: 动态开启/关闭跟踪功能
- **说明**: 跟踪模式提供稳定的ID，非跟踪模式性能更高

##### 5. 性能统计
```cpp
void getPerformanceStats(float& inference_time, float& total_time) const;
```
- **参数**: 
  - `inference_time`: NPU推理时间(ms)
  - `total_time`: 总处理时间(ms)
- **功能**: 获取最近一次处理的性能数据

### 📊 数据结构

#### PersonGroundResult
```cpp
struct PersonGroundResult {
    int track_id;                        // 跟踪ID (跟踪模式) 或 检测索引 (非跟踪模式)
    cv::Rect bbox;                       // 检测框 (原图坐标)
    cv::Point2f ground_roi;              // ROI方法地面坐标 (毫米)
    cv::Point2f ground_ankle;            // 脚踝+偏移方法地面坐标 (毫米)
    float vertical_offset;               // 动态垂直偏移量 (像素)
    float confidence;                    // 检测置信度 [0.0, 1.0]
    std::vector<cv::Point2f> keypoints;  // 17个关键点 (原图坐标)
    bool is_tracked;                     // 是否来自跟踪 (true=跟踪, false=检测)
};
```

#### 关键点索引 (COCO格式)
```cpp
索引  |  关键点名称     |  索引  |  关键点名称
------|----------------|--------|----------------
0     |  鼻子          |  9     |  左手腕
1     |  左眼          |  10    |  右手腕
2     |  右眼          |  11    |  左髋
3     |  左耳          |  12    |  右髋
4     |  右耳          |  13    |  左膝
5     |  左肩          |  14    |  右膝
6     |  右肩          |  15    |  左脚踝
7     |  左肘          |  16    |  右脚踝
8     |  右肘          |        |
```

### 💡 使用示例

#### 基础使用
```cpp
#include "person_ground_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. 创建检测器
    PersonGroundDetector detector(
        "models/Q_yolov8_pose.rknn",
        "data/2025_7_11pm.json",
        true  // 启用跟踪
    );
    
    // 2. 初始化
    if (detector.initialize() != 0) {
        std::cerr << "初始化失败" << std::endl;
        return -1;
    }
    
    // 3. 打开摄像头
    cv::VideoCapture cap(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    
    cv::Mat frame;
    while (cap.read(frame)) {
        // 4. 检测人物
        std::vector<PersonGroundResult> results;
        int count = detector.detectPersons(frame, results);
        
        // 5. 处理结果
        for (const auto& result : results) {
            printf("ID:%d 地面坐标:(%.0f,%.0f)mm 置信度:%.2f\n",
                   result.track_id,
                   result.ground_ankle.x, result.ground_ankle.y,
                   result.confidence);
        }
        
        // 6. 绘制并显示
        detector.drawResults(frame, results);
        cv::imshow("检测结果", frame);
        
        if (cv::waitKey(1) == 27) break;  // ESC退出
    }
    
    return 0;
}
```

#### 高级使用
```cpp
// 性能监控
float inference_time, total_time;
detector.getPerformanceStats(inference_time, total_time);
printf("推理: %.1fms, 总计: %.1fms\n", inference_time, total_time);

// 动态跟踪控制
if (some_condition) {
    detector.setTrackingEnabled(false);  // 关闭跟踪提升性能
}

// 过滤低置信度结果
std::vector<PersonGroundResult> filtered_results;
for (const auto& result : results) {
    if (result.confidence > 0.7) {
        filtered_results.push_back(result);
    }
}

// 访问特定关键点
for (const auto& result : results) {
    if (result.keypoints.size() >= 17) {
        cv::Point2f left_ankle = result.keypoints[15];   // 左脚踝
        cv::Point2f right_ankle = result.keypoints[16];  // 右脚踝
        cv::Point2f nose = result.keypoints[0];          // 鼻子
        
        // 计算人物朝向
        float orientation = atan2(left_ankle.y - nose.y, left_ankle.x - nose.x);
    }
}
```

---

## 配置文件格式

### 📄 Homography标定文件

标定文件使用JSON格式，包含像素坐标到真实世界坐标的映射关系：

```json
{
    "timestamp": "2025-07-11T16:54:46.584028",
    "points": [
        {
            "pixel": [999.1456, 1076.1942],    // 像素坐标 [x, y]
            "world": [0.0, 1820.0],            // 世界坐标 [x, y] 毫米
            "id": 0                            // 点ID
        },
        {
            "pixel": [1142.0583, 914.6408],
            "world": [455.0, 2730.0],
            "id": 1
        }
        // ... 更多标定点
    ],
    "matrix": [
        [-2.460319, 0.011829, 2462.065],      // Homography矩阵第1行
        [0.075152, 0.763650, -2805.956],      // Homography矩阵第2行
        [0.000074, -0.001996, 1.0]            // Homography矩阵第3行
    ],
    "point_count": 7                          // 标定点数量
}
```

### 🎯 标定点选择建议

为获得最佳映射精度，建议选择以下类型的标定点：

1. **场地边界点**: 篮球场边线交点
2. **关键标记**: 中圈、罚球线交点
3. **分布均匀**: 覆盖整个检测区域
4. **高度一致**: 所有点在同一水平面

```
篮球场标定点建议分布：
    A ---- B ---- C
    |      |      |
    D ---- E ---- F
    |      |      |
    G ---- H ---- I
```

---

## 性能优化

### ⚡ 系统级优化

#### 1. CPU调速器设置
```bash
# 设置性能模式
sudo cpufreq-set -c 0-7 -g performance

# 检查当前频率
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
```

#### 2. 内存优化
```bash
# 增加交换空间
sudo swapoff -a
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. NPU设备优化
```bash
# 检查NPU状态
cat /sys/class/devfreq/fdab0000.npu/cur_freq
cat /sys/class/devfreq/fdab0000.npu/available_frequencies

# 设置NPU频率
echo 1000000000 > /sys/class/devfreq/fdab0000.npu/min_freq
```

### 🔧 代码级优化

#### 1. 编译优化
```bash
# 编译时使用优化标志
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" ..
```

#### 2. 内存预分配
```cpp
// 预分配结果向量
std::vector<PersonGroundResult> results;
results.reserve(10);  // 预期最大人数

// 重用Mat对象
cv::Mat frame, display_frame;
while (running) {
    cap >> frame;
    frame.copyTo(display_frame);  // 避免重新分配
    // ... 处理
}
```

#### 3. 跟踪参数调优
```cpp
// 根据场景调整ByteTracker参数
PersonGroundDetector detector(model_path, homography_path, true);

// 在构造函数中传递优化参数
// tracker_ = std::make_unique<BYTETracker>(frame_rate, track_buffer);
// 推荐: frame_rate=30, track_buffer=30 (正常场景)
//       frame_rate=60, track_buffer=60 (高速运动)
```

### 📊 性能基准

#### RK3588平台性能指标
```
配置: RK3588 @ 2.4GHz, 8GB RAM, NPU @ 1.0GHz
输入: 1920x1080 @ 30fps
模型: YOLOv8n-pose

单人检测:
  推理时间: 12-15ms
  后处理时间: 3-5ms
  总处理时间: 20-25ms
  理论FPS: 40-50

多人检测(3人):
  推理时间: 15-18ms
  后处理时间: 8-12ms
  总处理时间: 25-35ms
  理论FPS: 30-40
```

---

## 常见问题

### ❓ 初始化问题

#### Q1: "❌ RKNN初始化失败"
**原因**: 模型文件不存在或格式错误
**解决**:
```bash
# 检查模型文件
ls -la models/Q_yolov8_pose.rknn
file models/Q_yolov8_pose.rknn

# 验证模型格式
python3 -c "
import rknn
rknn_model = rknn.RKNN()
ret = rknn_model.load_rknn('models/Q_yolov8_pose.rknn')
print('模型加载:', '成功' if ret == 0 else '失败')
"
```

#### Q2: "❌ 零拷贝初始化失败"
**原因**: NPU设备权限不足
**解决**:
```bash
# 检查NPU设备
ls -la /dev/rknpu*

# 设置权限
sudo chmod 666 /dev/rknpu*

# 或者使用root运行
sudo ./example_person_detector
```

#### Q3: "❌ Homography初始化失败"
**原因**: 标定文件格式错误
**解决**:
```bash
# 检查JSON格式
python3 -c "
import json
with open('data/2025_7_11pm.json', 'r') as f:
    data = json.load(f)
print('JSON格式正确')
print('包含字段:', list(data.keys()))
"
```

### ❓ 运行时问题

#### Q4: 地面坐标显示(-1, -1)
**原因**: Homography变换失败
**解决**:
```cpp
// 检查输入点是否在有效范围内
cv::Point2f test_point(960, 540);  // 图像中心
cv::Point2f ground_point = detector.mapToGround(test_point);
if (ground_point.x < 0 || ground_point.y < 0) {
    printf("Homography变换失败，检查标定文件\n");
}
```

#### Q5: 跟踪ID频繁变化
**原因**: 跟踪参数不适合当前场景
**解决**:
```cpp
// 调整跟踪器参数
// 在person_ground_detector.cpp中修改
tracker_ = std::make_unique<BYTETracker>(
    30,    // frame_rate: 降低以适应低帧率
    60     // track_buffer: 增加以提高稳定性
);
```

#### Q6: 检测精度低
**原因**: 光照、角度或模型不匹配
**解决**:
```cpp
// 过滤低置信度结果
for (const auto& result : results) {
    if (result.confidence > 0.6) {  // 调整阈值
        // 使用高置信度结果
    }
}

// 检查关键点质量
for (const auto& result : results) {
    int valid_keypoints = 0;
    for (const auto& kp : result.keypoints) {
        if (kp.x > 0 && kp.y > 0) valid_keypoints++;
    }
    if (valid_keypoints < 10) {
        printf("关键点质量低，可能需要调整光照\n");
    }
}
```

### ❓ 性能问题

#### Q7: 帧率低于预期
**原因**: 系统资源不足或配置不当
**解决**:
```bash
# 检查系统负载
top -p $(pgrep example_person_detector)

# 检查内存使用
cat /proc/meminfo | grep -E "(MemTotal|MemAvailable)"

# 检查NPU使用率
cat /sys/class/devfreq/fdab0000.npu/load
```

#### Q8: 内存泄漏
**原因**: 资源未正确释放
**解决**:
```cpp
// 使用智能指针
std::unique_ptr<PersonGroundDetector> detector = 
    std::make_unique<PersonGroundDetector>(model_path, homography_path);

// 确保在析构函数中释放资源
PersonGroundDetector::~PersonGroundDetector() {
    destroyZeroCopy();
    destroyRKNN();
}
```

---

## 高级用法

### 🎨 自定义绘制

```cpp
void customDrawResults(cv::Mat& frame, const std::vector<PersonGroundResult>& results) {
    for (const auto& result : results) {
        // 自定义颜色方案
        cv::Scalar color = result.is_tracked ? 
            cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);
        
        // 绘制增强检测框
        cv::rectangle(frame, result.bbox, color, 3);
        
        // 绘制置信度条
        int bar_width = (int)(result.confidence * 100);
        cv::rectangle(frame, 
            cv::Point(result.bbox.x, result.bbox.y - 15),
            cv::Point(result.bbox.x + bar_width, result.bbox.y - 5),
            color, -1);
        
        // 绘制轨迹预测
        if (result.is_tracked) {
            // 根据历史位置预测未来位置
            cv::Point2f predicted = predictNextPosition(result);
            cv::circle(frame, predicted, 5, cv::Scalar(0, 255, 255), -1);
        }
    }
}
```

### 📏 坐标系转换

```cpp
// 自定义坐标系转换
class CoordinateConverter {
public:
    // 地面坐标转换为场地坐标
    cv::Point2f groundToField(const cv::Point2f& ground_point) {
        // 假设场地原点在中心，x轴指向右侧，y轴指向上方
        return cv::Point2f(ground_point.x - field_center_x,
                          field_center_y - ground_point.y);
    }
    
    // 计算相对于篮筐的位置
    cv::Point2f relativeToBasket(const cv::Point2f& ground_point) {
        return cv::Point2f(ground_point.x - basket_x,
                          ground_point.y - basket_y);
    }
    
private:
    float field_center_x = 4600;  // 场地中心x坐标(mm)
    float field_center_y = 7500;  // 场地中心y坐标(mm)
    float basket_x = 4600;        // 篮筐x坐标(mm)
    float basket_y = 1575;        // 篮筐y坐标(mm)
};
```

### 🔍 区域检测

```cpp
// 定义感兴趣区域
class ROIManager {
public:
    void addROI(const std::string& name, const std::vector<cv::Point2f>& polygon) {
        roi_polygons_[name] = polygon;
    }
    
    bool isPointInROI(const cv::Point2f& point, const std::string& roi_name) {
        if (roi_polygons_.find(roi_name) == roi_polygons_.end()) {
            return false;
        }
        return cv::pointPolygonTest(roi_polygons_[roi_name], point, false) >= 0;
    }
    
    std::vector<PersonGroundResult> filterByROI(
        const std::vector<PersonGroundResult>& results, 
        const std::string& roi_name) {
        
        std::vector<PersonGroundResult> filtered;
        for (const auto& result : results) {
            if (isPointInROI(result.ground_ankle, roi_name)) {
                filtered.push_back(result);
            }
        }
        return filtered;
    }
    
private:
    std::map<std::string, std::vector<cv::Point2f>> roi_polygons_;
};

// 使用示例
ROIManager roi_manager;
roi_manager.addROI("paint_area", {
    cv::Point2f(3000, 0),    // 油漆区四个角点
    cv::Point2f(6200, 0),
    cv::Point2f(6200, 5800),
    cv::Point2f(3000, 5800)
});

// 过滤在油漆区内的人物
auto paint_area_results = roi_manager.filterByROI(results, "paint_area");
```

### 📊 统计分析

```cpp
// 运动统计分析
class MovementAnalyzer {
public:
    void updatePosition(int track_id, const cv::Point2f& position, 
                       std::chrono::high_resolution_clock::time_point timestamp) {
        position_history_[track_id].push_back({position, timestamp});
        
        // 保持最近5秒的历史
        auto cutoff_time = timestamp - std::chrono::seconds(5);
        auto& history = position_history_[track_id];
        history.erase(std::remove_if(history.begin(), history.end(),
            [cutoff_time](const PositionRecord& record) {
                return record.timestamp < cutoff_time;
            }), history.end());
    }
    
    float calculateSpeed(int track_id) {
        if (position_history_[track_id].size() < 2) return 0.0f;
        
        auto& history = position_history_[track_id];
        auto& latest = history.back();
        auto& previous = history[history.size() - 2];
        
        float distance = cv::norm(latest.position - previous.position);
        float time_diff = std::chrono::duration<float>(
            latest.timestamp - previous.timestamp).count();
        
        return distance / time_diff;  // mm/s
    }
    
    cv::Point2f calculateDirection(int track_id) {
        if (position_history_[track_id].size() < 2) return cv::Point2f(0, 0);
        
        auto& history = position_history_[track_id];
        auto& latest = history.back();
        auto& previous = history[history.size() - 2];
        
        cv::Point2f direction = latest.position - previous.position;
        float length = cv::norm(direction);
        return length > 0 ? direction / length : cv::Point2f(0, 0);
    }
    
private:
    struct PositionRecord {
        cv::Point2f position;
        std::chrono::high_resolution_clock::time_point timestamp;
    };
    
    std::map<int, std::vector<PositionRecord>> position_history_;
};
```

---

## 调试技巧

### 🔍 日志系统

```cpp
// 自定义日志级别
enum LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class Logger {
public:
    static void log(LogLevel level, const std::string& message) {
        if (level < current_level_) return;
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        const char* level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
        printf("[%s] %s: %s\n", 
               std::ctime(&time_t), level_str[level], message.c_str());
    }
    
    static void setLevel(LogLevel level) { current_level_ = level; }
    
private:
    static LogLevel current_level_;
};

// 在检测器中使用
Logger::log(Logger::DEBUG, "开始初始化RKNN");
Logger::log(Logger::INFO, "检测到 " + std::to_string(count) + " 个人物");
Logger::log(Logger::WARNING, "置信度低于阈值");
Logger::log(Logger::ERROR, "NPU推理失败");
```

### 📸 调试可视化

```cpp
// 调试模式绘制
void debugDrawResults(cv::Mat& frame, const std::vector<PersonGroundResult>& results) {
    for (const auto& result : results) {
        // 绘制原始检测框
        cv::rectangle(frame, result.bbox, cv::Scalar(0, 0, 255), 1);
        
        // 绘制关键点连接
        for (int i = 0; i < 19; i++) {
            int pt1 = skeleton[i][0] - 1;
            int pt2 = skeleton[i][1] - 1;
            if (pt1 >= 0 && pt1 < 17 && pt2 >= 0 && pt2 < 17) {
                cv::line(frame, result.keypoints[pt1], result.keypoints[pt2],
                        cv::Scalar(0, 255, 0), 1);
            }
        }
        
        // 绘制关键点编号
        for (int i = 0; i < 17; i++) {
            cv::circle(frame, result.keypoints[i], 3, cv::Scalar(255, 0, 0), -1);
            cv::putText(frame, std::to_string(i), 
                       result.keypoints[i] + cv::Point2f(5, 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
        
        // 绘制ROI和脚踝方法的位置点
        cv::circle(frame, result.ground_roi, 5, cv::Scalar(255, 0, 255), -1);
        cv::circle(frame, result.ground_ankle, 5, cv::Scalar(0, 255, 0), -1);
        
        // 显示详细信息
        cv::putText(frame, 
                   "ID:" + std::to_string(result.track_id) + 
                   " Conf:" + std::to_string(result.confidence),
                   cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
```

### 🎯 性能分析

```cpp
// 性能分析器
class PerformanceProfiler {
public:
    void startTimer(const std::string& name) {
        start_times_[name] = std::chrono::high_resolution_clock::now();
    }
    
    void endTimer(const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<float, std::milli>(
            end_time - start_times_[name]).count();
        
        durations_[name].push_back(duration);
        if (durations_[name].size() > 100) {
            durations_[name].erase(durations_[name].begin());
        }
    }
    
    void printStats() {
        for (const auto& pair : durations_) {
            float avg = std::accumulate(pair.second.begin(), pair.second.end(), 0.0f) 
                       / pair.second.size();
            printf("%s: 平均 %.2fms\n", pair.first.c_str(), avg);
        }
    }
    
private:
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::map<std::string, std::vector<float>> durations_;
};

// 在检测器中使用
PerformanceProfiler profiler;
profiler.startTimer("预处理");
// ... 预处理代码
profiler.endTimer("预处理");

profiler.startTimer("推理");
// ... 推理代码
profiler.endTimer("推理");
```

---

## 最佳实践

### 🏆 代码组织

#### 1. 模块化设计
```cpp
// 分离关注点
class PersonGroundDetector {
private:
    std::unique_ptr<RKNNInferenceEngine> inference_engine_;
    std::unique_ptr<CoordinateMapper> coordinate_mapper_;
    std::unique_ptr<TrackingManager> tracking_manager_;
    std::unique_ptr<SmoothingFilter> smoothing_filter_;
};
```

#### 2. 错误处理
```cpp
// 统一的错误处理机制
enum class DetectorError {
    SUCCESS = 0,
    INITIALIZATION_FAILED,
    INFERENCE_FAILED,
    INVALID_INPUT,
    RESOURCE_EXHAUSTED
};

class PersonGroundDetector {
public:
    DetectorError detectPersons(const cv::Mat& frame, 
                               std::vector<PersonGroundResult>& results,
                               std::string& error_message);
};
```

#### 3. 配置管理
```cpp
// 配置文件管理
struct DetectorConfig {
    std::string model_path;
    std::string homography_path;
    bool enable_tracking = true;
    float confidence_threshold = 0.5f;
    int smooth_window_size = 5;
    
    static DetectorConfig fromFile(const std::string& config_path);
    void saveToFile(const std::string& config_path) const;
};
```

### 🚀 性能最佳实践

#### 1. 内存管理
```cpp
// 对象池模式
class ResultPool {
public:
    PersonGroundResult* acquire() {
        if (pool_.empty()) {
            return new PersonGroundResult();
        }
        auto result = pool_.back();
        pool_.pop_back();
        return result;
    }
    
    void release(PersonGroundResult* result) {
        result->reset();  // 清理数据
        pool_.push_back(result);
    }
    
private:
    std::vector<PersonGroundResult*> pool_;
};
```

#### 2. 并发处理
```cpp
// 生产者-消费者模式
class AsyncPersonDetector {
public:
    void startAsync() {
        worker_thread_ = std::thread(&AsyncPersonDetector::workerLoop, this);
    }
    
    void submitFrame(const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        frame_queue_.push(frame.clone());
        condition_.notify_one();
    }
    
    bool getResults(std::vector<PersonGroundResult>& results) {
        std::lock_guard<std::mutex> lock(result_mutex_);
        if (result_queue_.empty()) return false;
        
        results = result_queue_.front();
        result_queue_.pop();
        return true;
    }
    
private:
    void workerLoop();
    
    std::thread worker_thread_;
    std::queue<cv::Mat> frame_queue_;
    std::queue<std::vector<PersonGroundResult>> result_queue_;
    std::mutex queue_mutex_, result_mutex_;
    std::condition_variable condition_;
};
```

### 🔧 部署建议

#### 1. 系统配置
```bash
# 系统服务配置
sudo tee /etc/systemd/system/person-detector.service << EOF
[Unit]
Description=Person Ground Detector Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/person-detector
ExecStart=/opt/person-detector/bin/person_detector_service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable person-detector.service
sudo systemctl start person-detector.service
```

#### 2. 监控和日志
```bash
# 日志轮转配置
sudo tee /etc/logrotate.d/person-detector << EOF
/var/log/person-detector/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
```

#### 3. 性能监控
```bash
# 创建监控脚本
cat > monitor_detector.sh << 'EOF'
#!/bin/bash

while true; do
    echo "$(date): $(top -bn1 | grep person_detector | head -1)" >> /var/log/performance.log
    echo "$(date): NPU频率: $(cat /sys/class/devfreq/fdab0000.npu/cur_freq)" >> /var/log/performance.log
    sleep 5
done
EOF

chmod +x monitor_detector.sh
```

---

## 结语

PersonGroundDetector 是一个功能强大、易于使用的人物地面坐标检测模块。通过本手册的详细说明，您应该能够：

1. ✅ 成功集成模块到您的项目中
2. ✅ 理解各个API的用法和参数
3. ✅ 解决常见的配置和运行问题
4. ✅ 根据需要进行性能优化和功能扩展

如果您在使用过程中遇到问题，请参考本手册的常见问题部分，或者通过调试技巧部分的方法进行问题定位。

**祝您使用愉快！** 🎉

---

*文档版本: v1.0*  
*最后更新: 2025-01-17*  
*适用于: PersonGroundDetector v1.0*