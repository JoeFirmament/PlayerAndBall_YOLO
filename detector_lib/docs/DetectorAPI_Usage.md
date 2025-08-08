# DetectorAPI 使用指南 - 坐标映射系统 ⭐

## 🎯 概述

坐标映射系统提供两套坐标系统：**笛卡尔坐标系**和**极坐标系**，可以将图像中的像素坐标转换为真实世界坐标（毫米单位），满足不同应用场景的需求。

**核心优势**：
- 🎯 **一行代码启用** - `detector.load_calibration("file.json")`
- 📐 **双坐标系统** - 同时支持笛卡尔坐标(x,y)和极坐标(r,θ)
- 🤖 **自动化处理** - 无需手动计算脚部位置，自动使用ROI底部中点
- 📏 **毫米级精度** - 真实测试精度可达毫米级
- 🔄 **无缝集成** - 完全集成到现有API，零学习成本

## 🚀 快速开始

### 最简使用（4行代码）

```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

int main() {
    // 智能查找模型文件
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    // 智能查找标定文件并启用坐标映射
    std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
    detector.load_calibration(calib_path);    // 启用坐标映射
    
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    // 直接获取世界坐标（笛卡尔+极坐标）
    for (const auto& pose : results) {
        // 笛卡尔坐标
        if (pose.has_ground_position) {
            printf("笛卡尔坐标: (%.1f, %.1f)mm\\n", 
                   pose.ground_position.x, pose.ground_position.y);
        }
        
        // 极坐标
        if (pose.has_polar_position) {
            printf("极坐标: 距离=%.1fmm, 角度=%.1f°\\n", 
                   pose.polar_position.r, pose.polar_position.theta_degrees());
        }
    }
    
    return 0;
}
```

## 📐 工作原理

### 1. ROI底部中点定位
```cpp
// 系统自动计算ROI框底部中点作为人员脚部位置
cv::Point2f foot_position(
    pose.bbox.x + pose.bbox.width / 2.0f,   // 中心X坐标
    pose.bbox.y + pose.bbox.height          // 底部Y坐标
);
```

### 2. 双坐标系统

#### 笛卡尔坐标系 (Cartesian)
```cpp
// 传统的X-Y坐标系统
cv::Point2f cartesian = pose.ground_position;
float x = cartesian.x;  // 水平位置 (mm)
float y = cartesian.y;  // 垂直位置 (mm)
```

#### 极坐标系 (Polar)
```cpp
// 距离+角度的坐标系统
detector::PolarCoordinate polar = pose.polar_position;
double distance = polar.r;                    // 距离 (mm)
double angle_rad = polar.theta;               // 角度 (弧度, -π到π)
double angle_deg = polar.theta_degrees();     // 角度 (度数, -180°到180°)
```

#### 极坐标配置
```cpp
// 方式1: JSON文件自动配置
detector.load_calibration("data/calibration.json");

// 方式2: 手动配置极坐标系统
detector.set_polar_coordinate_system(
    true,      // 启用极坐标
    100.0f,    // 原点X偏移 (mm)
    200.0f     // 原点Y偏移 (mm)
);
```

### 3. Homography变换
```cpp
// 使用3x3变换矩阵将像素坐标转换为世界坐标
cv::Point2f world_pos = apply_homography(foot_position);
pose.ground_position = world_pos;
pose.has_ground_position = true;
```

### 3. 结果输出
```
人员[0] ROI框: (624, 276, 96, 264)
ROI底部中点: (672.0, 540.0)      ← 像素坐标
世界坐标: (35.2, 3929.4)mm       ← 真实世界坐标
```

## 📄 标定文件格式

### 标准JSON格式
```json
{
    "timestamp": "2025-08-06T15:39:15.447713",
    "matrix": [
        [-3.2720398953723757, -0.006616969830473663, 2185.3722002814093],
        [-0.07920249932550606, 0.6201388621485532, -2183.270680916352],
        [2.0578777115434938e-05, -0.0027736686912052497, 1.0]
    ],
    "points": [
        {
            "pixel": [263.45631067961165, 574.1359223300971],
            "world": [-2275.0, 3185.0],
            "id": 0
        },
        {
            "pixel": [666.0970873786408, 719.9482200647249],
            "world": [0.0, 1820.0],
            "id": 2
        }
    ],
    "point_count": 11,
    "origin_offset": [0.0, 0.0],
    "use_polar": true
}
```

### 字段说明
- **`matrix`**: 3x3 Homography变换矩阵（核心数据）
- **`points`**: 标定点对，用于验证精度
- **`pixel`**: 图像像素坐标 [x, y]
- **`world`**: 真实世界坐标 [x, y]，单位毫米
- **`timestamp`**: 标定时间戳
- **`point_count`**: 标定点数量
- **`origin_offset`**: 极坐标原点偏移量 [x_offset, y_offset]（可选）
- **`use_polar`**: 是否启用极坐标计算 true/false（可选）

## 🎮 实际测试演示

### 运行测试程序 (v1.0.3 最新)
```bash
# 零配置使用 - 解压即用
cd detector_lib/bin/

# 基础姿态检测 (智能路径查找)
./pose_image

# Homography坐标映射测试
./pose_image_with_homography  

# 极坐标系统测试 (推荐⭐)
./pose_image_with_polar

# 篮筐篮球检测
./rim_basketball_image
```

### 输出结果
```
=== 姿态检测图片测试 ===
✓ 检测器创建成功
✓ 启用跟踪功能
✓ Homography标定加载成功
✓ 成功加载图片: ../imgs/pose.jpg (1280x720)

开始姿态检测...
检测完成，推理时间: 49ms
检测到 1 个人

人员[0] ROI框: (624, 276, 96, 264)
ROI底部中点: (672.0, 540.0)
世界坐标: (35.2, 3929.4)mm

✅ 检测结果已保存到: pose_test_result.jpg
=== 测试完成 ===
```

### 可视化结果
生成的 `pose_test_result.jpg` 包含：
- ✅ 绿色ROI边界框
- ✅ 红色底部中点标记
- ✅ 完整的坐标信息标注
- ✅ 世界坐标数值显示

## 🛠 高级用法

### 动态切换坐标映射 (智能路径查找)
```cpp
#include "detector_path_utils.h"
using namespace detector;

// 智能查找模型文件
std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
PoseDetectorLib detector(model_path);

// 方案A：不使用坐标映射
auto results1 = detector.detect(image);
// results1[0].has_ground_position == false

// 方案B：启用坐标映射 (智能查找标定文件)
std::string calib1 = PathUtils::find_calibration("calib1.json");
detector.load_calibration(calib1);
auto results2 = detector.detect(image);
// results2[0].has_ground_position == true
// results2[0].has_polar_position == true (如果JSON中启用)

// 方案C：切换不同标定文件
std::string calib2 = PathUtils::find_calibration("calib2.json");
detector.load_calibration(calib2);
auto results3 = detector.detect(image);
// 使用新的变换矩阵和极坐标配置
```

### 批量处理 (智能路径查找)
```cpp
#include "detector_path_utils.h"
using namespace detector;

// 智能查找模型文件和标定文件
std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");

PoseDetectorLib detector(model_path);
detector.load_calibration(calib_path);

std::vector<std::string> image_files = {"img1.jpg", "img2.jpg", "img3.jpg"};

for (const auto& filename : image_files) {
    cv::Mat image = cv::imread(filename);
    auto results = detector.detect(image);
    
    for (const auto& pose : results) {
        if (pose.has_ground_position) {
            printf("%s: 笛卡尔(%.1f, %.1f)mm", 
                   filename.c_str(), 
                   pose.ground_position.x, pose.ground_position.y);
                   
            if (pose.has_polar_position) {
                printf(", 极坐标(%.1fmm, %.1f°)", 
                       pose.polar_position.r, pose.polar_position.theta_degrees());
            }
            printf("\n");
        }
    }
}
```

### 实时视频处理 (智能路径查找)
```cpp
#include "detector_path_utils.h"
using namespace detector;

// 智能查找模型文件和标定文件
std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");

PoseDetectorLib detector(model_path);
detector.load_calibration(calib_path);

cv::VideoCapture cap(0);
cv::Mat frame;

while (cap.read(frame)) {
    auto results = detector.detect(frame);
    
    for (const auto& pose : results) {
        printf("ID[%d] 实时位置: 笛卡尔(%.1f, %.1f)mm", 
               pose.person_id, 
               pose.ground_position.x, pose.ground_position.y);
               
        if (pose.has_polar_position) {
            printf(", 极坐标(%.1fmm, %.1f°)", 
                   pose.polar_position.r, pose.polar_position.theta_degrees());
        }
        printf("\n");
    }
    
    // 可以根据坐标进行实时分析、告警等
}
```

## 📊 精度与性能

### 测试环境
- **硬件**: Orange Pi 5 Plus (RK3588)
- **分辨率**: 1280x720 (姿态检测)
- **标定点**: 11个标定点

### 性能数据
| 项目 | 数值 | 说明 |
|------|------|------|
| 推理时间 | 49ms | 包含坐标映射计算 |
| 坐标精度 | 毫米级 | 实测转换精度 |
| 初始化 | <1ms | 标定文件加载时间 |
| 内存开销 | 零增量 | 不影响原有性能 |

### 实测精度验证
```
标定点验证:
像素坐标 (666.1, 719.9) → 世界坐标 (0.0, 1820.0)mm ✓
像素坐标 (263.5, 574.1) → 世界坐标 (-2275.0, 3185.0)mm ✓

实际检测:
像素坐标 (672.0, 540.0) → 世界坐标 (35.2, 3929.4)mm
```

## 🔧 故障排除

### 常见问题

**Q: 标定文件加载失败？**
```cpp
bool success = detector.load_calibration("calib.json");
if (!success) {
    printf("标定文件不存在或格式错误\\n");
    // 系统会优雅降级，不影响基本检测功能
}
```

**Q: 坐标映射结果异常？**
```cpp
for (const auto& pose : results) {
    if (pose.has_ground_position) {
        // 检查坐标合理性
        if (abs(pose.ground_position.x) > 10000 || 
            abs(pose.ground_position.y) > 10000) {
            printf("警告：坐标异常，请检查标定文件\\n");
        }
    } else {
        printf("提示：未启用坐标映射或标定失败\\n");
    }
}
```

**Q: 不同分辨率如何处理？**
- 标定文件必须与推理图片分辨率匹配
- 如果分辨率不同，需要重新进行标定
- 建议为每个分辨率准备独立的标定文件

## 🎯 应用场景

### 体育分析
```cpp
// 篮球场上球员位置追踪
for (const auto& pose : results) {
    if (pose.has_ground_position) {
        float x = pose.ground_position.x;  // 距离场地中心的距离
        float y = pose.ground_position.y;  // 距离底线的距离
        
        // 判断球员位置区域
        if (y < 2000) {
            printf("球员[%d] 在三秒区内\\n", pose.person_id);
        }
    }
}
```

### 安全监控
```cpp
// 危险区域入侵检测
for (const auto& pose : results) {
    if (pose.has_ground_position) {
        // 定义危险区域边界（毫米）
        if (pose.ground_position.x > 5000 && pose.ground_position.y < 1000) {
            printf("警告：检测到人员进入危险区域！\\n");
        }
    }
}
```

### 空间分析
```cpp
// 人员密度分析
std::map<std::pair<int,int>, int> grid_count;
for (const auto& pose : results) {
    if (pose.has_ground_position) {
        // 将空间划分为1m×1m网格
        int grid_x = (int)(pose.ground_position.x / 1000);
        int grid_y = (int)(pose.ground_position.y / 1000);
        grid_count[{grid_x, grid_y}]++;
    }
}
```

## 🎉 总结

Homography坐标映射功能让DetectorAPI库从"图像检测"升级为"空间感知"，为用户提供了：

✅ **极简API** - 一行代码启用复杂功能  
✅ **自动化处理** - 无需手动计算，智能定位  
✅ **毫米级精度** - 真实测试验证的高精度  
✅ **零学习成本** - 完全集成到现有接口  
✅ **实用性强** - 适用于多种实际应用场景  

让AI检测不仅能"看到"目标，更能"定位"目标在真实世界中的精确位置！

---

**版本**: v1.0.3  
**更新日期**: 2025-08-07  
**测试平台**: RK3588 (Orange Pi 5 Plus)  
**新特性**: 智能路径查找 + 极坐标系统