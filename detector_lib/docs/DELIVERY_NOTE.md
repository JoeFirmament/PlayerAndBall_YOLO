# YOLOv8检测器库 - 交付说明

## 📦 交付文件

1. **主文件**: `yolov8_detector_lib_rk3588_v1.0.0_20250806.tar.gz` (8.5MB)
2. **校验文件**: `yolov8_detector_lib_rk3588_v1.0.0_20250806.tar.gz.md5`

## ✅ 验证完整性

```bash
# 验证MD5校验和
md5sum -c yolov8_detector_lib_rk3588_v1.0.0_20250806.tar.gz.md5
```

## 🚀 用户快速开始

```bash
# 1. 解压文件
tar -xzf yolov8_detector_lib_rk3588_v1.0.0_20250806.tar.gz

# 2. 进入目录
cd detector_lib

# 3. 一键编译安装
./build_and_install.sh

# 4. 功能验证
./test.sh

# 5. 查看示例结果
cd build/examples
ls *_result.jpg
```

## 📋 包含内容

- ✅ **完整源代码**: 姿态检测和篮筐篮球检测的封装实现
- ✅ **预训练模型**: 两个RKNN量化模型 (INT8优化)
- ✅ **示例程序**: 7个示例程序，覆盖所有使用场景
- ✅ **测试图片**: 验证用的测试图片
- ✅ **RKNN运行库**: librknnrt.so (专用版本)
- ✅ **自动化脚本**: 编译、测试、安装一键完成
- ✅ **完整文档**: API文档、使用指南、故障排查

## 🎯 核心功能

### 姿态检测 ⭐
- **推理时间**: 49ms (真实NPU测试)
- **检测精度**: 85%置信度，17个COCO关键点
- **特色功能**: ByteTrack跟踪 + Homography坐标映射
- **坐标转换**: 像素→世界坐标 (毫米级精度)

### 篮筐篮球检测  
- **推理时间**: 39ms (真实NPU测试)
- **检测精度**: 篮筐97%，篮球95%
- **特色功能**: ROI分析 + 距离计算

## 💡 使用示例

#### 基础姿态检测 (3行代码)
```cpp
#include "PoseDetectorLib.h"
detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
auto poses = detector.detect(image);
```

#### 带坐标映射 (4行代码) ⭐ 推荐
```cpp
#include "PoseDetectorLib.h"
detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
detector.load_calibration("data/calibration.json");
auto poses = detector.detect(image);

// 直接获取毫米级世界坐标
if (poses[0].has_ground_position) {
    printf("真实位置: (%.1f, %.1f)mm\n", 
           poses[0].ground_position.x, poses[0].ground_position.y);
}
```

#### 篮筐篮球检测 (3行代码)
```cpp
#include "RimBasketballDetectorLib.h"
detector::RimBasketballDetectorLib detector("models/Q_Rim_Basketball_724_JZ.rknn");
auto objects = detector.detect(image);
```

## ⚠️ 系统要求

- **硬件**: RK3588平台 (Orange Pi 5 Plus等)
- **系统**: Ubuntu 20.04+
- **依赖**: OpenCV 4.x, CMake 3.10+
- **权限**: NPU设备访问权限

## 📞 技术支持

如遇到问题：
1. 运行 `./test.sh` 进行自动诊断
2. 查看 `README.md` 的故障排查章节
3. 检查 `PACKAGE_INFO.md` 的详细说明

---

**交付时间**: 2025-08-06  
**版本号**: v1.0.0  
**文件大小**: 8.5MB  
**MD5**: 8599fac8e1588290c0930977450f2d5d