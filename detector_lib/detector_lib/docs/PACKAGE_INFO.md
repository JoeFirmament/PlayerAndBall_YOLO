# YOLOv8检测器库 - 用户交付包

## 📦 打包清单

### 🗂️ 目录结构
```
detector_lib/
├── 📚 README.md                   # 快速开始指南
├── 🔧 build_and_install.sh       # 一键编译脚本
├── 🧪 test.sh                    # 功能测试脚本
├── 📋 PACKAGE_INFO.md            # 本文件
├── 📝 CHANGELOG.md               # 更新日志
├── ⚙️  CMakeLists.txt             # 编译配置
├── 📄 detector_lib.pc.in         # pkg-config配置
│
├── 📁 include/                   # 头文件 (用户API)
│   ├── detector_lib.h            # 统一入口头文件
│   ├── detector_types.h          # 数据类型定义
│   ├── PoseDetectorLib.h         # 姿态检测器API
│   └── RimBasketballDetectorLib.h # 篮筐篮球检测API
│
├── 🔨 src/                       # 源代码实现
│   ├── detector_lib.cpp          # 主库实现
│   ├── PoseDetectorLib.cpp       # 姿态检测器实现
│   ├── RimBasketballDetectorLib.cpp # 篮筐篮球检测器实现
│   └── internal/                 # 内部实现
│       ├── detector_common.h     # 公共工具头文件
│       └── detector_common.cpp   # 公共工具实现
│
├── 💡 examples/                  # 示例程序
│   ├── CMakeLists.txt                   # 示例编译配置
│   ├── pose_image.cpp                   # 基础姿态检测
│   ├── pose_image_with_homography.cpp  # 姿态+Homography坐标映射
│   ├── rim_basketball_image.cpp         # 篮筐篮球检测
│   └── test_detector_lib.cpp            # 综合功能测试
│
├── 🤖 models/                    # AI模型文件
│   ├── Q_yolov8_pose.rknn      # 姿态检测模型 (49ms推理)
│   └── Q_Rim_Basketball_724_JZ.rknn # 篮筐篮球模型 (39ms推理)
│
└── 🖼️  imgs/                     # 测试图片
    ├── pose.jpg                 # 姿态检测测试图
    └── rim.jpg                  # 篮筐篮球测试图
```

## ✅ 功能验证报告

### 🧍 姿态检测器验证 ⭐
- **推理时间**: ✅ 49ms (真实NPU推理)
- **检测精度**: ✅ 置信度85%, 9/17关键点有效
- **特色功能**: ✅ ByteTrack跟踪 + Homography坐标映射
- **坐标转换**: ✅ 像素坐标→世界坐标 (毫米级精度)
- **输出结果**: ✅ 自动生成 `pose_test_result.jpg` (包含坐标可视化)

### 🏀 篮筐篮球检测器验证  
- **推理时间**: ✅ 39ms (真实NPU推理)
- **检测精度**: ✅ 篮筐97%置信度, 篮球95%置信度
- **特色功能**: ✅ ROI分析 + 距离计算 (234px)
- **输出结果**: ✅ 自动生成 `rim_basketball_detection_result.jpg`

## 🚀 用户使用流程

### 1. 解压获取
```bash
# 用户操作
unzip detector_lib.zip
cd detector_lib
```

### 2. 一键安装
```bash
./build_and_install.sh    # 自动编译库和示例
```

### 3. 功能验证
```bash
./test.sh                 # 验证所有功能正常
```

### 4. 开始使用

#### 基础检测 (3行代码)
```cpp
#include "PoseDetectorLib.h"
detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
auto poses = detector.detect(image);
```

#### 带坐标映射 (4行代码) ⭐
```cpp
#include "PoseDetectorLib.h"
detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
detector.load_calibration("data/calibration.json");  // 启用坐标映射
auto poses = detector.detect(image);

// 直接获取世界坐标
for (const auto& pose : poses) {
    if (pose.has_ground_position) {
        printf("真实位置: (%.1f, %.1f)mm\n", 
               pose.ground_position.x, pose.ground_position.y);
    }
}
```

#### 篮筐篮球检测
```cpp
#include "RimBasketballDetectorLib.h"
detector::RimBasketballDetectorLib detector("models/Q_Rim_Basketball_724_JZ.rknn");
auto objects = detector.detect(image);
```

## ⚠️ 重要提醒

### 必须配置项
1. **NPU权限**: `sudo chmod 666 /dev/dri/renderD*`
2. **用户组**: `sudo usermod -a -G video $USER` (需重新登录)
3. **依赖包**: `sudo apt install libopencv-dev cmake build-essential`

### 系统要求
- **硬件**: RK3588平台 (Orange Pi 5 Plus等)
- **系统**: Ubuntu 20.04+
- **OpenCV**: 4.x版本
- **CMake**: 3.10+

### 性能说明
- **首次运行**: 1-3秒预热时间 (NPU初始化)
- **后续推理**: 39-49ms稳定推理时间
- **内存占用**: 零拷贝优化，最小化内存使用

## 🔧 故障排查

### 编译问题
```bash
# 依赖缺失
sudo apt install libopencv-dev cmake build-essential

# 权限问题  
chmod +x *.sh
```

### 运行问题
```bash
# NPU设备不可访问
sudo chmod 666 /dev/dri/renderD*
ls -la /dev/dri/  # 检查设备权限

# 模型加载失败
ls -la models/    # 检查模型文件存在

# 库链接错误
ldd build/examples/pose_image  # 检查依赖库
```

## 📈 性能基准

### 测试环境
- **平台**: Orange Pi 5 Plus (RK3588)
- **系统**: Ubuntu 22.04
- **OpenCV**: 4.6.0
- **输入分辨率**: 姿态检测1280x720, 篮筐检测1280x960

### 性能数据
| 项目 | 姿态检测 | 篮筐篮球检测 |
|------|----------|--------------|
| 输入分辨率 | 1280x720 | 1280x960 |
| 推理时间 | 49ms | 39ms |
| 检测精度 | 85%置信度 | 97%篮筐,95%篮球 |
| 内存占用 | 零拷贝优化 | 零拷贝优化 |
| 初始化时间 | 1-3秒 | 1-3秒 |

## 🎯 应用场景

- **体育分析**: 运动员动作分析，技术统计
- **智能监控**: 人员行为检测，异常识别  
- **交互系统**: 手势识别，体感控制
- **教育科研**: AI算法研究，技术验证

## 📞 技术支持

### 常见问题解答
1. **Q**: 首次运行很慢？
   **A**: 正常，NPU初始化需要1-3秒，后续会很快

2. **Q**: 检测结果为空？
   **A**: 检查图片格式(需BGR)，模型路径，NPU权限

3. **Q**: 编译失败？
   **A**: 安装依赖包，检查CMake版本，确认OpenCV安装

### 获取帮助
- 运行 `./test.sh` 进行全面诊断
- 查看程序输出的详细日志信息
- 确认硬件平台为RK3588系列

---

## 🎉 总结

**交付内容**: 完整的YOLOv8检测库，即插即用
**验证状态**: 已通过真实RK3588测试验证
**用户操作**: 解压 → 编译 → 测试 → 使用 (4步完成)
**技术支持**: 完整文档 + 自动化脚本 + 故障排查指南

🚀 **让AI检测变得简单！**