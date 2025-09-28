# 相机标定文件格式指南

## 📋 关于NPZ格式

### NPZ是什么？
NPZ（NumPy Zipped）是NumPy库的专用压缩格式，用于存储多个numpy数组。

### 为什么使用NPZ？
- **原生支持**：NumPy原生格式，无需额外依赖
- **高效压缩**：自动压缩，节省存储空间
- **类型保持**：完全保持numpy数据类型和精度
- **快速加载**：内存映射加载，速度快

### NPZ的缺点
- **Python专用**：主要用于Python生态系统
- **跨语言困难**：其他语言需要特殊处理
- **可读性差**：二进制格式，人眼无法直接阅读

## 🔄 NPZ转换为其他格式

是的！NPZ格式可以转换为其他多种格式。我已经为您创建了一个转换工具。

### 支持的格式

1. **JSON格式** - 人类可读，语言无关
```json
{
  "camera_matrix": [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
  "dist_coeffs": [-0.1, 0.05, 0.0, 0.0, 0.0]
}
```

2. **YAML格式** - 结构清晰，支持注释
```yaml
camera_matrix:
  - [800.0, 0.0, 640.0]
  - [0.0, 800.0, 360.0]
  - [0.0, 0.0, 1.0]
dist_coeffs: [-0.1, 0.05, 0.0, 0.0, 0.0]
```

3. **XML格式** - 结构化数据存储
4. **二进制格式** - 自定义高效格式
5. **MATLAB .mat格式** - MATLAB兼容格式

## 📖 使用转换工具

### 基本用法
```bash
# 转换为JSON
python npz_converter.py camera_calibration.npz json

# 转换为YAML并指定输出路径
python npz_converter.py camera_calibration.npz yaml -o calib_config.yaml

# 查看支持的格式
python npz_converter.py --list
```

### C++使用示例
```cpp
#include "cpp_calibration_loader.h"

int main() {
    CalibrationLoader loader;

    // 加载标定文件
    if (loader.loadCameraCalibration("camera_calibration.npz")) {
        // 使用标定数据
        cv::Mat undistorted = loader.undistortImage(input_image);
    }
}
```

## 🎯 Image Display功能详解

### Calibration Bench右侧的Image Display是干什么用的？

**Image Display** 是相机标定工作台的核心可视化组件，主要功能包括：

### 1. **图像预览功能**
- **选择文件夹后**：自动显示找到的标定图像数量
- **图像信息显示**：显示当前图像的分辨率、格式等信息
- **状态提示**：显示"Select image folder to preview images"等提示信息

### 2. **相机预览功能**（Camera标签页）
右侧的预览区域用于：
- **实时相机预览**：连接相机后实时显示摄像头画面
- **分辨率预览**：显示当前设置的分辨率效果
- **拍摄状态显示**：显示拍摄进度和状态信息

### 3. **标定验证功能**
- **验证完成后**：显示标定质量报告
- **错误可视化**：显示重投影误差统计
- **质量评估**：显示EXCELLENT/GOOD/POOR等质量等级

### 4. **地面标定预览**
- **图像选择**：预览地面标定图像
- **角点检测显示**：显示检测到的棋盘格角点
- **原点设置**：支持交互式选择地面坐标原点

## 💡 为什么需要Image Display？

1. **质量控制**：确保标定图像质量足够
2. **参数验证**：验证相机设置是否正确
3. **结果验证**：验证标定结果的准确性
4. **调试支持**：帮助诊断标定过程中的问题
5. **用户反馈**：提供直观的视觉反馈

## 🎯 推荐格式选择

### Python开发
- **推荐**：NPZ格式（原生支持，最方便）

### C++/其他语言开发
- **推荐**：JSON或YAML（人类可读，语言无关）

### 网络传输
- **推荐**：JSON（轻量级，广泛支持）

### 存档/备份
- **推荐**：NPZ（完整数据保持）+ JSON（可读备份）

## 🚀 实际应用建议

1. **开发阶段**：使用NPZ，享受完整功能
2. **生产环境**：转换为JSON，便于跨平台部署
3. **数据交换**：使用JSON，确保兼容性
4. **长期存档**：保留NPZ格式，确保数据完整性