# 检测器API使用文档

## 概述

本文档介绍了封装后的 `PoseDetector` 和 `RimBasketballDetector` 类的使用方法。这两个类将原有的复杂检测逻辑封装为简单易用的接口。

## 核心特性

### ✅ 极简接口
- 用户只需调用 `detect()` 函数
- 自动处理NPU内存管理、零拷贝优化
- 自动清理资源，无内存泄漏风险

### ✅ 延迟初始化
- 构造函数不会初始化重资源
- 首次调用 `detect()` 时自动初始化
- 初始化失败会返回空结果，不会崩溃

### ✅ 高性能优化
- 保留了原有的零拷贝NPU内存优化
- 预分配所有内存，`detect()` 函数无内存分配
- 支持连续多帧检测，性能稳定

## PoseDetector 使用说明

### 接口详细说明

#### 构造函数
```cpp
PoseDetector::PoseDetector(const std::string& model_path)
```
**参数**：
- `model_path` (string): RKNN模型文件的**绝对路径**或**相对路径**
  - 示例：`"models/Q_yolov8_pose.rknn"` 或 `"/home/user/models/Q_yolov8_pose.rknn"`
  - 必须是RK3588平台的.rknn格式文件
  - 文件必须存在且可读

**注意**：构造函数只保存路径，不进行初始化，不会抛出异常。

#### 核心检测接口
```cpp
std::vector<PoseResult> detect(const cv::Mat& frame)
```
**输入参数**：
- `frame` (cv::Mat): 输入图像帧
  - **格式要求**：BGR或RGB色彩空间
  - **推荐分辨率**：1920x1080 (与标定数据一致)
  - **支持分辨率**：任意分辨率 (内部会resize到640x640)
  - **数据类型**：CV_8UC3 (8位无符号3通道)

**返回值**：
- `std::vector<PoseResult>`: 检测结果数组
  - **数组大小含义**：
    - `size() == 0`: 无人员检测到，或初始化失败，或输入图像为空
    - `size() == 1`: 检测到1个人员
    - `size() == N`: 检测到N个人员 (N可达10+人)
  - **数组内容**：每个元素是一个完整的PoseResult对象
  - **元素排序**：按检测置信度从高到低排序
  - **ID分配**：如果启用跟踪，person_id为ByteTracker分配的持续ID；否则为-1

#### PoseResult数组详细说明

```cpp
std::vector<PoseResult> results = detector.detect(frame);

// 1. 检查数组大小
printf("检测到 %zu 个人员\n", results.size());

// 2. 遍历每个检测结果
for (size_t i = 0; i < results.size(); i++) {
    const auto& pose = results[i];
    
    printf("=== 人员 %zu ===\n", i);
    printf("跟踪ID: %d\n", pose.person_id);           // ByteTrack ID，-1表示未跟踪
    printf("置信度: %.3f\n", pose.confidence);         // [0.0-1.0]，越高越可信
    printf("边界框: x=%d, y=%d, w=%d, h=%d\n", 
           pose.bbox.x, pose.bbox.y, pose.bbox.width, pose.bbox.height);
    
    // 3. 关键点信息
    printf("关键点数量: %zu (固定17个COCO关键点)\n", pose.keypoints.size());
    for (size_t j = 0; j < pose.keypoints.size(); j++) {
        printf("  关键点%zu: (%.1f, %.1f) 置信度:%.3f\n", 
               j, pose.keypoints[j].x, pose.keypoints[j].y, pose.keypoint_scores[j]);
    }
    
    // 4. 地面坐标 (如果启用Homography映射)
    if (pose.has_ground_position) {
        printf("地面坐标: (%.1f, %.1f)\n", 
               pose.ground_position.x, pose.ground_position.y);
    } else {
        printf("地面坐标: 未计算 (无标定文件)\n");
    }
}
```

#### 返回值状态判断

```cpp
auto results = detector.detect(frame);

if (results.empty()) {
    // 情况1: 初始化失败
    if (!detector.is_initialized()) {
        printf("错误: 检测器初始化失败\n");
        // 处理初始化错误...
    }
    // 情况2: 正常情况，但当前帧无人员
    else {
        printf("当前帧无人员检测到\n");
    }
}
else {
    // 成功检测到人员
    printf("成功: 检测到 %zu 个人员\n", results.size());
    
    // 获取置信度最高的人员 (数组第一个元素)
    const auto& best_person = results[0];
    printf("置信度最高人员: ID=%d, 置信度=%.3f\n", 
           best_person.person_id, best_person.confidence);
}

### 基本用法

```cpp
#include "PoseDetector.h"

// 1. 创建检测器 (延迟初始化)
PoseDetector detector("models/Q_yolov8_pose.rknn");

// 2. 直接检测 (首次调用时自动初始化)
cv::Mat frame;  // 用户自己获取图像
std::vector<PoseResult> results = detector.detect(frame);

// 3. 析构时自动清理资源 (用户无需手动清理)
```

### 完整示例

```cpp
// 创建检测器
PoseDetector detector("models/Q_yolov8_pose.rknn");

// 可选配置
detector.load_calibration("data/calibration.json");  // 加载Homography标定
detector.enable_tracking(true);                      // 启用ByteTrack跟踪
detector.set_confidence_threshold(0.3f);             // 设置置信度阈值

// 摄像头循环
cv::VideoCapture cap(0);
cv::Mat frame;
while (cap.read(frame)) {
    // 核心接口：超简单!
    auto results = detector.detect(frame);
    
    // 处理结果
    for (const auto& pose : results) {
        printf("人员ID: %d, 置信度: %.2f\n", pose.person_id, pose.confidence);
        printf("边界框: (%d,%d,%d,%d)\n", pose.bbox.x, pose.bbox.y, pose.bbox.width, pose.bbox.height);
        printf("关键点数量: %zu\n", pose.keypoints.size());
        
        if (pose.has_ground_position) {
            printf("地面坐标: (%.1f, %.1f)\n", pose.ground_position.x, pose.ground_position.y);
        }
    }
}
```

### PoseResult 结构体说明

```cpp
struct PoseResult {
    int person_id;                      // ByteTrack分配的人员ID
    float confidence;                   // 检测置信度 [0-1]
    cv::Rect bbox;                      // 边界框 (x, y, width, height)
    std::vector<cv::Point2f> keypoints; // 17个COCO关键点坐标
    std::vector<float> keypoint_scores; // 关键点置信度 [0-1]
    cv::Point2f ground_position;        // Homography映射的地面坐标
    bool has_ground_position;           // 是否有有效地面坐标
};
```

#### COCO 17关键点索引说明
`keypoints` 和 `keypoint_scores` 数组固定包含17个元素，索引对应关系：

```cpp
// COCO关键点索引定义 (索引从0开始)
enum COCOKeypoints {
    NOSE = 0,           // 鼻子
    LEFT_EYE = 1,       // 左眼
    RIGHT_EYE = 2,      // 右眼
    LEFT_EAR = 3,       // 左耳
    RIGHT_EAR = 4,      // 右耳
    LEFT_SHOULDER = 5,  // 左肩
    RIGHT_SHOULDER = 6, // 右肩
    LEFT_ELBOW = 7,     // 左肘
    RIGHT_ELBOW = 8,    // 右肘
    LEFT_WRIST = 9,     // 左腕
    RIGHT_WRIST = 10,   // 右腕
    LEFT_HIP = 11,      // 左髋
    RIGHT_HIP = 12,     // 右髋
    LEFT_KNEE = 13,     // 左膝
    RIGHT_KNEE = 14,    // 右膝
    LEFT_ANKLE = 15,    // 左踝
    RIGHT_ANKLE = 16    // 右踝
};

// 使用示例
const auto& pose = results[0];
cv::Point2f nose = pose.keypoints[NOSE];
float nose_confidence = pose.keypoint_scores[NOSE];

printf("鼻子坐标: (%.1f, %.1f), 置信度: %.3f\n", 
       nose.x, nose.y, nose_confidence);

// 常用关键点获取
cv::Point2f left_ankle = pose.keypoints[LEFT_ANKLE];   // 左脚踝
cv::Point2f right_ankle = pose.keypoints[RIGHT_ANKLE]; // 右脚踝
cv::Point2f ground_center = (left_ankle + right_ankle) * 0.5f; // 地面接触点
```

## RimBasketballDetector 使用说明

### 接口详细说明

#### 构造函数
```cpp
RimBasketballDetector::RimBasketballDetector(const std::string& model_path)
```
**参数**：
- `model_path` (string): RKNN模型文件的**绝对路径**或**相对路径**
  - 示例：`"models/Q_Rim_Basketball_724_JZ.rknn"` 或 `"/home/user/models/Q_Rim_Basketball_724_JZ.rknn"`
  - 必须是RK3588平台的.rknn格式文件
  - 文件必须存在且可读

**注意**：构造函数只保存路径，不进行初始化，不会抛出异常。

#### 核心检测接口
```cpp
std::vector<RimBasketballResult> detect(const cv::Mat& frame)
```
**输入参数**：
- `frame` (cv::Mat): 输入图像帧
  - **格式要求**：BGR或RGB色彩空间
  - **推荐分辨率**：1920x1080 (获得最佳检测效果)
  - **支持分辨率**：任意分辨率 (内部会resize到640x640)
  - **数据类型**：CV_8UC3 (8位无符号3通道)

**返回值**：
- `std::vector<RimBasketballResult>`: 检测结果数组
  - **数组大小含义**：
    - `size() == 0`: 无目标检测到，或初始化失败，或输入图像为空
    - `size() == 1`: 检测到1个目标 (篮筐或篮球)
    - `size() == N`: 检测到N个目标，可能包含多个篮球和篮筐
  - **数组内容**：每个元素是一个完整的RimBasketballResult对象
  - **元素排序**：按检测置信度从高到低排序
  - **类别混合**：数组中可能同时包含篮筐(rim)和篮球(basketball)对象

#### RimBasketballResult数组详细说明

```cpp
std::vector<RimBasketballResult> results = detector.detect(frame);

// 1. 检查数组大小
printf("检测到 %zu 个目标\n", results.size());

// 2. 分类统计
int rim_count = 0, basketball_count = 0;
for (const auto& result : results) {
    if (result.class_id == 1) rim_count++;        // rim
    else if (result.class_id == 0) basketball_count++; // basketball
}
printf("其中: %d 个篮筐, %d 个篮球\n", rim_count, basketball_count);

// 3. 遍历每个检测结果
for (size_t i = 0; i < results.size(); i++) {
    const auto& obj = results[i];
    
    printf("=== 目标 %zu ===\n", i);
    printf("类别: %s (ID=%d)\n", obj.class_name.c_str(), obj.class_id);
    printf("置信度: %.3f\n", obj.confidence);      // [0.0-1.0]
    printf("边界框: x=%d, y=%d, w=%d, h=%d\n", 
           obj.bbox.x, obj.bbox.y, obj.bbox.width, obj.bbox.height);
    printf("中心点: (%.1f, %.1f)\n", obj.center.x, obj.center.y);
    
    // 4. 篮球专有信息
    if (obj.class_id == 0) {  // basketball
        printf("到篮筐距离: %.1f 像素\n", obj.distance_to_rim);
        printf("是否在篮筐ROI: %s\n", obj.is_in_rim_roi ? "是" : "否");
    }
}
```

#### 典型使用场景

```cpp
auto results = detector.detect(frame);

// 场景1: 寻找篮筐
std::vector<RimBasketballResult> rims;
for (const auto& obj : results) {
    if (obj.class_id == 1) {  // rim
        rims.push_back(obj);
    }
}
printf("发现 %zu 个篮筐\n", rims.size());

// 场景2: 分析篮球状态
for (const auto& obj : results) {
    if (obj.class_id == 0) {  // basketball
        printf("篮球置信度: %.3f\n", obj.confidence);
        if (obj.is_in_rim_roi) {
            printf("🎯 篮球靠近篮筐! 距离: %.1f\n", obj.distance_to_rim);
        }
    }
}

// 场景3: 空结果处理
if (results.empty()) {
    if (!detector.is_initialized()) {
        printf("错误: 检测器初始化失败\n");
    } else {
        printf("当前帧无篮筐和篮球\n");
    }
}

### 基本用法

```cpp
#include "RimBasketballDetector.h"

// 1. 创建检测器
RimBasketballDetector detector("models/Q_Rim_Basketball_724_JZ.rknn");

// 2. 直接检测
cv::Mat frame;
std::vector<RimBasketballResult> results = detector.detect(frame);
```

### 完整示例

```cpp
// 创建检测器
RimBasketballDetector detector("models/Q_Rim_Basketball_724_JZ.rknn");

// 可选配置
detector.set_confidence_threshold(0.4f);  // 置信度阈值
detector.set_nms_threshold(0.5f);         // NMS阈值

// 检测循环
cv::VideoCapture cap(2);  // 篮筐检测摄像头
cv::Mat frame;
while (cap.read(frame)) {
    auto results = detector.detect(frame);
    
    for (const auto& result : results) {
        printf("类别: %s, 置信度: %.2f\n", result.class_name.c_str(), result.confidence);
        printf("边界框: (%d,%d,%d,%d)\n", result.bbox.x, result.bbox.y, result.bbox.width, result.bbox.height);
        
        if (result.class_id == 0) {  // basketball
            printf("到篮筐距离: %.1f\n", result.distance_to_rim);
            printf("是否在篮筐ROI: %s\n", result.is_in_rim_roi ? "是" : "否");
        }
    }
}
```

### RimBasketballResult 结构体说明

```cpp
struct RimBasketballResult {
    int class_id;                    // 类别ID: 0=basketball, 1=rim
    std::string class_name;          // 类别名称: "basketball" 或 "rim"
    float confidence;                // 检测置信度 [0-1]
    cv::Rect bbox;                   // 边界框 (x, y, width, height)
    cv::Point2f center;              // 中心点坐标
    float distance_to_rim;           // 篮球到最近篮筐的距离 (仅basketball有效)
    bool is_in_rim_roi;             // 篮球是否在篮筐ROI区域内
};
```

## 性能测量

用户可以轻松测量推理时间：

```cpp
auto start = std::chrono::high_resolution_clock::now();
auto results = detector.detect(frame);
auto end = std::chrono::high_resolution_clock::now();

auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
printf("推理耗时: %ld ms\n", inference_time.count());
```

## 重要注意事项

### ⚠️ 延迟初始化行为
- **首次调用耗时**：首次调用 `detect()` 时会进行初始化，可能需要1-3秒
- **错误处理**：初始化失败时返回空的 `std::vector`，不会抛出异常
- **预热建议**：建议在程序启动时进行一次"预热"调用

#### 为什么需要预热？
延迟初始化在首次调用时需要执行以下耗时操作：
1. **RKNN模型加载** (~500ms): 从文件加载.rknn模型到NPU
2. **NPU内存分配** (~300ms): 分配输入输出零拷贝内存
3. **NPU上下文初始化** (~200ms): 创建推理上下文，绑定内存
4. **ByteTracker初始化** (~100ms): 仅PoseDetector需要
5. **NPU预编译** (~500ms): NPU首次运行模型时的编译优化

总计约1-3秒的初始化时间。如果在实际检测时才进行初始化，会导致：
- **第一帧检测卡顿**：用户体验差
- **实时性能下降**：影响FPS统计
- **超时风险**：某些场景下可能被误判为程序卡死

```cpp
// 推荐：程序启动时预热
PoseDetector detector("models/Q_yolov8_pose.rknn");
cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);
printf("正在预热检测器...\n");
detector.detect(dummy_frame);  // 预热调用，忽略返回值
printf("预热完成，后续检测将快速响应\n");
// 后续调用 detect() 将在10-30ms内返回
```

### ⚠️ 线程安全
- 每个检测器实例**不是**线程安全的
- 如需多线程，请为每个线程创建独立的检测器实例
- NPU资源有限，建议同时最多使用2个检测器实例

### ⚠️ 摄像头分辨率与标定
- Homography标定与摄像头分辨率严格对应
- 更改分辨率后必须重新标定
- 建议固定使用1920x1080分辨率

## 错误处理

### 详细错误处理示例

```cpp
// 1. 创建检测器时的错误处理
PoseDetector detector("models/invalid_model.rknn");

// 2. 检测时的错误处理
cv::Mat frame;  // 确保frame不为空
auto results = detector.detect(frame);

// 3. 判断结果状态
if (results.empty()) {
    if (!detector.is_initialized()) {
        printf("错误: 检测器初始化失败，可能原因:\n");
        printf("  - 模型文件路径错误或文件不存在\n");
        printf("  - NPU设备权限不足 (需要root或video组权限)\n");
        printf("  - 模型格式不兼容 (非RK3588平台.rknn文件)\n");
        printf("  - 内存不足\n");
    } else {
        printf("信息: 当前帧未检测到目标 (正常情况)\n");
    }
} else {
    printf("成功: 检测到 %zu 个目标\n", results.size());
}
```

### 常见错误码对应
- **返回空vector + is_initialized()=false**: 初始化失败
- **返回空vector + is_initialized()=true**: 无检测目标
- **程序崩溃**: 通常是NPU权限问题，运行`sudo chmod 666 /dev/dri/renderD*`

## 编译说明

将以下文件加入CMakeLists.txt：

```cmake
# 头文件
include_directories(include)

# 源文件
add_library(PoseDetector src/PoseDetector.cc)
add_library(RimBasketballDetector src/RimBasketballDetector.cc)

# 依赖库
target_link_libraries(PoseDetector rknn_api opencv_core opencv_imgproc)
target_link_libraries(RimBasketballDetector rknn_api opencv_core opencv_imgproc)
```

## 示例程序

参考 `examples/simple_detection_example.cc` 获取完整的使用示例。

---

**总结**: 这套API将复杂的NPU检测逻辑封装为2个函数调用，用户体验极佳：
1. `detector.detect(frame)` - 核心接口
2. `detector.load_calibration()` - 可选标定

从几百行复杂代码简化为几行代码，大幅降低使用门槛！