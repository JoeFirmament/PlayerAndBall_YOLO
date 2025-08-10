# YOLOv8检测器库 - RK3588专用 🚀

🎯 **高性能的姿态检测和篮筐篮球检测库**，经过实际验证，真实RKNN推理。

## ⚠️ 重要：RKNN版本兼容性

**本库使用RKNN模型版本6格式，内置匹配的Runtime库。如遇到版本错误，库会自动处理，无需手动干预。**

```bash
❌ 如果看到此错误：
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!

✅ 解决：使用本库的编译系统，会自动使用内置的RKNN库
```

## ✨ 实测性能

| 功能 | 推理时间 | 检测精度 | 特色功能 |
|------|----------|----------|----------|
| 🧍 **姿态检测** | **32ms** | 置信度84%, 17关键点检测 | ByteTrack跟踪 + 双坐标系统 |
| 🏀 **篮筐篮球检测** | **39ms** | 篮筐97%, 篮球95% | ROI分析 + 距离计算 |

> ✅ **已验证**: 这些数据来自真实RK3588测试，不是模拟数据！

## 🎯 核心特性

- ✅ **3行代码搞定** - 创建检测器 → 调用detect() → 获取结果
- ✅ **双坐标系统** - 同时输出笛卡尔坐标(x,y)和极坐标(r,θ)
- ✅ **真实RKNN推理** - INT8量化模型，零拷贝NPU优化
- ✅ **完整可视化** - 自动生成检测框、关键点、骨架、坐标标注
- ✅ **即插即用** - 一键编译，开箱即用
- ✅ **工业级稳定** - 基于验证的生产代码

## 🚀 快速开始

### 方案1: 使用预编译包 (推荐 ⭐)

**解压即用，零配置！**

```bash
# 1. 解压发布包
tar -xzf yolov8_detector_lib_rk3588_v1.0.3.tar.gz
cd detector_lib

# 2. 直接运行示例程序
cd bin/
./pose_image_with_polar                 # 🌟 极坐标演示（单图，默认不启用跟踪）
./pose_image                            # 基础姿态检测（单图，默认不启用跟踪）
./rim_basketball_image                  # 篮筐篮球检测
./pose_camera_bytetrack_homography      # 摄像头+ByteTrack+Homography+极坐标 实时示例（新增）
```

✅ **相对路径机制** - 无需设置环境变量，程序自动找到所有依赖
✅ **完整依赖** - 包含 librknnrt.so 和所有必需库文件
✅ **智能路径查找** - 支持多种文件路径配置

### 方案2: 从源码编译

#### 1. 环境要求
- **硬件平台**: RK3588 (Orange Pi 5 Plus / ROC-RK3588S等)
- **操作系统**: Ubuntu 20.04+ / Debian 11+
- **编译器**: GCC 7.5+ (支持C++11)

#### 2. 安装依赖
```bash
# 基础依赖
sudo apt update && sudo apt install build-essential cmake git libopencv-dev libeigen3-dev

# NPU设备权限
sudo chmod 666 /dev/dri/renderD* && sudo usermod -a -G video $USER
```

#### 3. 编译
```bash
# 一键构建
./build_and_install.sh

# 或手动构建
mkdir build && cd build && cmake .. && make -j$(nproc)
```

### 4. 基本使用

#### 最简使用（智能路径查找）
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

int main() {
    // 1. 智能查找模型文件 (支持多种路径)
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    // 2. 检测图片
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    // 3. 获取结果
    for (const auto& pose : results) {
        printf("检测到人员，置信度: %.2f\n", pose.confidence);
        printf("ROI框: (%d,%d,%d,%d)\n", 
               pose.bbox.x, pose.bbox.y, pose.bbox.width, pose.bbox.height);
    }
    
    return 0; // 自动清理资源
}
```

✅ **PathUtils::find_model()** 会自动在以下位置查找模型：
- 环境变量 `DETECTOR_MODEL_PATH`
- `./models/`, `../models/`, `../../models/`
- `/usr/local/share/detector_lib/models/`
- `$HOME/.detector_lib/models/`

#### 双坐标系统功能 ⭐
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

int main() {
    // 智能路径查找
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    // 🎯 启用坐标映射 (智能查找标定文件)
    std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
    if (!calib_path.empty()) {
        detector.load_calibration(calib_path);  // 自动读取极坐标配置
    }
    
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    for (const auto& pose : results) {
        printf("人员ID: %d, 置信度: %.2f\n", pose.person_id, pose.confidence);
        
        // 📍 笛卡尔坐标系 (x,y)
        if (pose.has_ground_position) {
            printf("笛卡尔坐标: (%.1f, %.1f)mm\n", 
                   pose.ground_position.x, pose.ground_position.y);
        }
        
        // 🎯 极坐标系 (距离+角度)
        if (pose.has_polar_position) {
            printf("极坐标: 距离=%.1fmm, 角度=%.1f°\n", 
                   pose.polar_position.r, pose.polar_position.theta_degrees());
        }
    }
    
    return 0;
}
```

#### 环境变量配置 (高级用户)
```bash
# 自定义模型和数据文件路径
export DETECTOR_MODEL_PATH="/my/custom/models"
export DETECTOR_DATA_PATH="/my/custom/data"

# 然后程序会优先使用这些路径
./pose_image_with_polar
```

## 📊 API参考

### 核心类

- **`PoseDetectorLib`** - 姿态检测器主类
- **`RimBasketballDetectorLib`** - 篮筐篮球检测器  
- **`PathUtils`** - 智能路径查找工具

### 数据结构

```cpp
struct PoseResult {
    int person_id;                    // ByteTrack跟踪ID
    float confidence;                 // 检测置信度
    cv::Rect bbox;                   // ROI边界框
    cv::Point2f ground_position;     // 笛卡尔坐标(mm)
    PolarCoordinate polar_position;  // 极坐标(距离mm,角度弧度)
    bool has_ground_position;        // 是否有世界坐标
    bool has_polar_position;         // 是否有极坐标
};
```

### 智能路径查找

```cpp
// 环境变量优先级系统
PathUtils::find_model("Q_yolov8_pose.rknn");
// 查找顺序:
// 1. $DETECTOR_MODEL_PATH/Q_yolov8_pose.rknn
// 2. ./models/Q_yolov8_pose.rknn  
// 3. ../models/Q_yolov8_pose.rknn
// 4. /usr/local/share/detector_lib/models/Q_yolov8_pose.rknn
// 5. $HOME/.detector_lib/models/Q_yolov8_pose.rknn
```

## 📚 详细文档

### 🎯 专题指南
- **[DetectorAPI使用指南 - Homography坐标映射](docs/DetectorAPI_Usage.md)** ⭐ 推荐
  - 详细介绍坐标映射功能的使用方法
  - 包含完整的代码示例和最佳实践
  - 涵盖标定文件格式、精度验证、应用场景
- **[手动编译指南](docs/MANUAL_COMPILATION.md)** 🔧 实用
  - g++命令行编译方法
  - 静态库构建和测试程序编译
  - 用户项目集成示例和故障排除

### API参考

#### PoseDetectorLib - 姿态检测器

```cpp
// 构造函数 - 延迟初始化，无异常
PoseDetectorLib(const std::string& model_path);

// 🎯 核心检测接口 - 一键获取完整结果
std::vector<PoseResult> detect(const cv::Mat& frame);

// 🔧 功能配置
void enable_tracking(bool enable = true);                    // ByteTrack多目标跟踪
bool load_calibration(const std::string& calibration_file);  // Homography+极坐标映射
void set_polar_coordinate_system(bool enable, float offset_x, float offset_y);  // 手动配置极坐标
void set_confidence_threshold(float threshold);              // 检测置信度阈值

// 📊 状态查询  
bool is_initialized() const;                                // 是否已初始化
DetectorStatus get_status() const;                          // 当前状态
int get_last_inference_time_ms() const;                    // 上次推理时间(ms)
void release();                                             // 手动释放资源
```

#### 双坐标系统功能 ⭐

支持**笛卡尔坐标系**和**极坐标系**，可以将图像中的像素坐标转换为真实世界坐标（毫米单位），特别适合体育分析、位置追踪、机器人导航等应用。

```cpp
// 🎯 一行代码启用双坐标系统
detector.load_calibration("data/calibration.json");

// ✅ 自动工作原理
// 1. 自动提取ROI框底部中点作为人员脚部位置
// 2. 使用Homography变换矩阵将像素坐标转换为笛卡尔坐标
// 3. 基于原点偏移量计算极坐标(距离+角度)
// 4. 结果同时存储在 pose.ground_position 和 pose.polar_position 中

// 📍 获取双坐标系统结果
auto results = detector.detect(image);
for (const auto& pose : results) {
    // 笛卡尔坐标 (x,y)
    if (pose.has_ground_position) {
        printf("笛卡尔坐标: (%.1f, %.1f)mm\n", 
               pose.ground_position.x, pose.ground_position.y);
    }
    
    // 极坐标 (距离,角度)
    if (pose.has_polar_position) {
        printf("极坐标: 距离=%.1fmm, 角度=%.1f°\n", 
               pose.polar_position.r, pose.polar_position.theta_degrees());
    }
}

// 手动配置极坐标系统
detector.set_polar_coordinate_system(true, 100.0f, 200.0f);  // 启用，原点偏移(100,200)mm
```

**标定文件格式**：
```json
{
    "timestamp": "2025-08-06T15:39:15.447713",
    "matrix": [
        [-3.2720398953723757, -0.006616969830473663, 2185.3722002814093],
        [-0.07920249932550606, 0.6201388621485532, -2183.270680916352],
        [2.0578777115434938e-05, -0.0027736686912052497, 1.0]
    ],
    "points": [
        {"pixel": [263.4, 574.1], "world": [-2275.0, 3185.0]},
        {"pixel": [666.1, 719.9], "world": [0.0, 1820.0]}
        // ... 更多标定点
    ],
    "origin_offset": [0.0, 0.0],
    "use_polar": true
}
```

#### RimBasketballDetectorLib

```cpp
// 构造函数
RimBasketballDetectorLib(const std::string& model_path);

// 核心检测接口
std::vector<RimBasketballResult> detect(const cv::Mat& frame);

// 配置接口
void set_confidence_threshold(float threshold);
void set_nms_threshold(float threshold);

// 状态查询
bool is_initialized() const;
static std::vector<std::string> get_supported_classes();
```

### 数据结构

#### PoseResult - 检测结果数据结构 ⭐

```cpp
struct PoseResult {
    // 🆔 跟踪信息
    int person_id;                      // ByteTrack跟踪ID (-1表示未启用跟踪)
    float confidence;                   // 检测置信度 [0.0-1.0]
    
    // 📦 边界框信息
    cv::Rect bbox;                      // ROI边界框 (x, y, width, height)
    
    // 🦴 关键点信息 (17个COCO标准关键点)
    std::vector<cv::Point2f> keypoints; // 17个关键点坐标 [(x,y), ...]
    std::vector<float> keypoint_scores; // 17个关键点置信度 [0.0-1.0]
    
    // 🌍 双坐标系统 (Homography映射)
    cv::Point2f ground_position;        // 笛卡尔坐标 (x,y mm)
    PolarCoordinate polar_position;     // 极坐标 (距离mm, 角度弧度)
    bool has_ground_position;           // 是否有有效的笛卡尔坐标
    bool has_polar_position;            // 是否有有效的极坐标
};
```

**关键点索引对照 (COCO标准)**：
```cpp
// 人体17个关键点的标准索引
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
if (pose.keypoint_scores[LEFT_ANKLE] > 0.5f) {
    printf("左脚踝位置: (%.1f, %.1f)\n", 
           pose.keypoints[LEFT_ANKLE].x, pose.keypoints[LEFT_ANKLE].y);
}
```

**实际测试结果示例**：
```
人员[0] ROI框: (624, 276, 96, 264)
ROI底部中点: (672.0, 540.0) 
笛卡尔坐标: (35.2, 3929.4)mm   ← 自动计算的真实位置
极坐标: 距离=3929.5mm, 角度=89.5°  ← 自动计算的极坐标
```

#### RimBasketballResult
```cpp
struct RimBasketballResult {
    int class_id;                       // 0=basketball, 1=rim
    std::string class_name;             // 类别名称
    float confidence;                   // 检测置信度
    cv::Rect bbox;                      // 边界框
    cv::Point2f center;                 // 中心点
    float distance_to_rim;              // 到篮筐距离
    bool is_in_rim_roi;                // 是否在篮筐ROI内
};
```

## 🛠 编译选项

### CMake选项

```bash
cmake -DCMAKE_BUILD_TYPE=Release \\     # Release/Debug
      -DBUILD_EXAMPLES=ON \\           # 构建示例程序
      -DINSTALL_EXAMPLES=OFF \\        # 安装示例程序
      ..
```

### 构建脚本选项

```bash
./build_and_install.sh -h              # 显示帮助
./build_and_install.sh -r              # Release构建
./build_and_install.sh -d -t           # Debug构建并测试
./build_and_install.sh -r -i           # Release构建并安装
./build_and_install.sh --no-examples   # 不构建示例
```

## 📂 项目结构

```
detector_lib/
├── include/                    # 公共头文件
│   ├── detector_lib.h         # 主头文件
│   ├── detector_types.h       # 数据类型定义 (含极坐标)
│   ├── PoseDetectorLib.h      # 姿态检测器接口
│   ├── RimBasketballDetectorLib.h # 篮筐检测器接口
│   ├── detector_common_types.h # RKNN公共类型定义
│   ├── detector_file_utils.h  # 文件操作工具
│   ├── detector_path_utils.h  # 智能路径查找工具 ⭐
│   ├── detector_rim_basketball_postprocess.h # 篮筐检测后处理
│   └── rknn_api.h            # RKNN NPU API接口
├── src/                       # 实现文件
│   ├── internal/              # 内部实现
│   │   ├── detector_common.h  # 公共工具
│   │   └── detector_common.cpp
│   ├── PoseDetectorLib.cpp    # 姿态检测器实现
│   ├── RimBasketballDetectorLib.cpp # 篮筐检测器实现
│   ├── detector_lib.cpp       # 库主实现
│   ├── detector_file_utils.c  # 文件操作工具实现
│   ├── detector_path_utils.cpp # 智能路径查找实现 ⭐
│   └── rim_basketball_postprocess_simple.cpp # 篮筐检测后处理实现
├── examples/                  # 示例程序
│   ├── test_detector_lib.cpp        # 功能测试
│   ├── pose_image.cpp               # 基础姿态检测
│   ├── pose_image_with_homography.cpp # Homography坐标映射
│   ├── pose_image_with_polar.cpp    # 极坐标系统演示
│   └── rim_basketball_image.cpp     # 篮筐篮球检测
├── CMakeLists.txt            # 构建配置
├── build_and_install.sh      # 构建脚本
└── README.md                 # 项目文档
```

## 🎮 示例程序

### 1. 功能测试程序

```bash
cd build/examples
./test_detector_lib
```

测试库的基本功能，验证初始化、检测、状态查询等接口。

### 2. 基础姿态检测

```bash
cd build/examples
./pose_image
```

基础姿态检测演示，生成带有检测框和关键点的结果图片。

### 3. Homography坐标映射

```bash  
cd build/examples
./pose_image_with_homography
```

演示笛卡尔坐标映射功能，输出真实世界坐标。

### 4. 极坐标系统演示 ⭐ 推荐

```bash
cd build/examples
./pose_image_with_polar
```

完整的双坐标系统演示，同时输出：
- 笛卡尔坐标 `(35.2, 3929.4)mm`
- 极坐标 `距离=3929.5mm, 角度=89.5°`

### 5. 篮筐篮球检测

```bash
cd build/examples
./pose_image                    # 基础姿态检测 (使用智能路径)
./pose_image_with_homography    # 姿态+Homography坐标映射 (推荐!)
./rim_basketball_image          # 篮筐篮球检测
```

#### `pose_image_with_polar` - 完整双坐标系统演示 ⭐

**功能演示**：
- ✅ 智能查找模型文件和标定数据
- ✅ 执行姿态检测推理
- ✅ 启用Homography+极坐标映射
- ✅ 输出笛卡尔坐标和极坐标
- ✅ 生成带坐标标注的可视化结果

**输出示例**：
```
=== 姿态检测+极坐标系统测试 ===
✓ 检测器创建成功
✓ 启用跟踪功能
✓ Homography标定+极坐标配置加载成功
✓ 成功加载图片: ../../imgs/pose.jpg (1280x720)

开始姿态检测...
检测完成，推理时间: 33ms
检测到 1 个人

=== 人员[0] ===
跟踪ID: 1
置信度: 0.85
ROI框: (624, 276, 96, 264)
ROI底部中点: (672.0, 540.0)
笛卡尔坐标: (35.2, 3929.4)mm
极坐标: 距离=3929.5mm, 角度=89.5°

✅ 检测结果已保存到: pose_with_polar_result.jpg
=== 极坐标功能测试完成 ===
```

### 6. 摄像头实时示例（新增）

```bash
cd detector_lib
# 推荐用一键脚本（自动设置库路径并查找默认模型/标定）
./run_pose_camera_bytetrack_homography.sh 0

# 或直接运行可执行：
cd build/examples
./pose_camera_bytetrack_homography ../models/Q_yolov8_pose.rknn 0 ../data/2025_8_6_1280_720.json
```

说明：
- 摄像头示例默认启用 ByteTrack，并显示 Homography 世界坐标与极坐标。
- 单图示例默认关闭 ByteTrack，不影响 ROI/关键点，`person_id` 为 -1。

### 4. 篮筐检测演示

```bash
cd build/examples
./rim_basketball_demo 2  # 使用摄像头2
```

## ⚡ 性能表现

### 推理性能 (RK3588)
- **PoseDetector**: 49ms/帧 (1280x720输入)  
- **RimBasketballDetector**: 39ms/帧 (1280x960输入)
- **内存占用**: 与原程序相同 (零开销封装)

### 初始化时间
- **首次调用**: 1-3秒 (模型加载+NPU初始化)
- **后续调用**: 10-30ms (稳定推理时间)

## 🔧 故障排除

### 常见问题

#### ❌ RKNN版本兼容性错误 (最常见)

**症状:**
```bash
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!
```

**原因:** 系统RKNN库版本过旧，不支持Version 6模型格式

**解决方法:**
```bash
# ✅ 方法1: 使用本库编译系统 (推荐)
cd detector_lib && mkdir build && cd build
cmake .. && make -j$(nproc)
# 会自动使用项目内的新版RKNN库

# ✅ 方法2: 检查库版本
ls -la /lib/librknnrt.so*
# 新版库约7.4MB，旧版库约3.5MB

# ⚠️ 方法3: 手动替换系统库 (需要sudo)
sudo cp lib/librknnrt.so /lib/
sudo ldconfig
```

**Q: 编译时找不到rknn_api.h**
```bash
# 检查RKNN头文件路径
find /path/to/project -name "rknn_api.h"
# 或添加包含路径
export CMAKE_PREFIX_PATH=/path/to/rknpu2/include:$CMAKE_PREFIX_PATH
```

**Q: 运行时初始化失败**
```bash
# 检查NPU设备权限
ls -la /dev/dri/renderD*
sudo chmod 666 /dev/dri/renderD*

# 检查程序实际使用的RKNN库
ldd ./your_program | grep rknn
readelf -d ./your_program | grep -E "RPATH|RUNPATH"
```

**Q: 检测结果为空**
```bash
# 检查模型文件
file models/*.rknn
# 检查输入图像格式 (应为CV_8UC3)
```

### 调试模式

```cpp
// 设置日志级别
detector::set_log_level(4);  // 0=静默, 4=调试

// 检查状态
if (!detector.is_initialized()) {
    auto status = detector.get_status();
    // 处理初始化失败...
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

### 开发环境设置

```bash
git clone <repository>
cd detector_lib
./build_and_install.sh -d -t  # Debug构建并测试
```

## 📄 许可证

本项目遵循 Apache 2.0 许可证。

## 🙏 致谢

- **瑞芯微** - RKNN运行时和NPU支持
- **OpenCV社区** - 图像处理库
- **ByteDance** - ByteTrack多目标跟踪算法

---

**技术支持**: 如有问题，请提交Issue或查看项目Wiki。

**更新日志**: 查看 CHANGELOG.md 了解版本更新信息。