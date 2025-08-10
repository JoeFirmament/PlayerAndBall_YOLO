# DetectorLib 更新日志

## [v1.0.4] - 2025-08-10

### 🚀 新增功能 - NPU核心分配支持

#### ⚡ NPU资源管理
- **NPU核心分配接口**: 为用户暴露NPU核心选择权，避免多检测器资源冲突
- **RK3588S平台优化**: 支持3个NPU核心（0、1、2）的显式分配
- **向后兼容设计**: 默认参数-1保持自动分配行为，不影响现有代码

#### 🔧 API增强
- **PoseDetectorLib构造函数扩展**:
  ```cpp
  explicit PoseDetectorLib(const std::string& model_path, int npu_core = -1);
  ```
- **RimBasketballDetectorLib构造函数扩展**:
  ```cpp
  explicit RimBasketballDetectorLib(const std::string& model_path, int npu_core = -1);
  ```
- **NPU核心分配策略**:
  - `-1` (默认): 自动分配，由RKNN Runtime决定
  - `0`: 强制使用NPU核心0
  - `1`: 强制使用NPU核心1  
  - `2`: 强制使用NPU核心2

#### 🛠️ 工具和监控
- **NPU监控脚本** (`scripts/monitor_npu.sh`):
  - 实时显示NPU负载率、频率、温度
  - 需要sudo权限访问 `/sys/kernel/debug/rknpu/load`
- **NPU信息检测脚本** (`scripts/check_npu_info.sh`):
  - 检测NPU设备状态和驱动版本
  - 显示可用频率和调频策略
- **C++ NPU工具类** (`include/npu_utils.h`):
  - `NPUUtils::get_npu_info()`: 获取NPU状态信息
  - `NPUUtils::get_recommended_core()`: 智能推荐空闲核心

#### 📋 示例程序
- **dual_camera_with_npu**: 双摄像头NPU优化示例
  - 演示如何为不同检测器分配不同NPU核心
  - 包含错误处理和性能监控
  - 支持自动分配和手动指定模式
- **test_npu_allocation**: NPU分配测试程序
  - 验证NPU核心分配功能
  - 测试多检测器并发场景

#### 🔍 使用场景
双摄像头系统最佳实践：
```cpp
// 摄像头0 -> 姿态检测 -> NPU核心0
PoseDetectorLib pose_detector(pose_model_path, 0);

// 摄像头2 -> 篮筐检测 -> NPU核心1
RimBasketballDetectorLib rim_detector(rim_model_path, 1);
```

#### 🧠 RKNN Runtime智能调度机制揭秘

**重要发现**：通过深度测试验证，RKNN Runtime内置智能负载均衡机制！

**智能调度验证数据**：
```
测试场景                    | 系统吞吐量  | 说明
---------------------------|-----------|-------------------
自动分配（默认-1）          | 77.5 FPS  | Runtime智能调度
相同NPU核心（0+0）          | 49.4 FPS  | 资源竞争基线
不同NPU核心（0+1）          | 77.5 FPS  | 手动最优分配
```

**核心API接口**：
- **`rknn_init()`**: 默认启用 `RKNN_NPU_CORE_AUTO` 智能调度
- **`RKNN_NPU_CORE_AUTO = 0`**: 定义在 `rknn_api.h:247`，实现负载均衡
- **`rknn_set_core_mask()`**: 可覆盖智能调度，手动指定NPU核心

**重要结论**：
- ✅ **v1.0.3已经享受NPU智能调度的性能优势**
- ✅ **v1.0.4提供精确控制能力，满足高级用户需求**
- 🎯 **智能调度 ≈ 手动最优分配性能**，证明Runtime调度机制非常有效

#### ⚠️ 注意事项
- NPU核心分配仅在初始化时设置，运行时不可更改
- 多个检测器使用相同NPU核心可能导致性能下降
- **大部分用户使用默认智能调度即可获得最佳性能**
- 建议高级用户在特殊场景下显式分配不同核心

## [v1.0.3] - 2025-08-08

### 🚀 重大更新 - RKNN版本兼容性 + 相对路径机制

#### ⚠️ RKNN版本兼容性修复
- **RKNN模型版本6支持**: 内置匹配的RKNN Runtime库 (7.4MB)
- **版本冲突解决**: 自动避免与系统旧版RKNN库(3.5MB)冲突
- **RPATH强制链接**: 程序运行时自动使用项目内RKNN库
- **CMake库查找优化**: 优先查找项目内库，避免系统库回退
- **错误处理增强**: 如果找不到匹配库版本会报错而非静默失败

**修复的错误:**
```bash
❌ 修复前:
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!

✅ 修复后:
✓ RKNN初始化成功，正在查询模型信息...
✓ 模型信息: 输入=1, 输出=4
```

#### 🔧 技术实现
- **库查找路径**: `${CMAKE_CURRENT_SOURCE_DIR}/lib` 优先级最高
- **RPATH设置**: `BUILD_RPATH` + `INSTALL_RPATH` 双重保障  
- **版本隔离**: 项目库与系统库完全隔离，避免版本冲突

#### ✨ 新增功能
- **相对路径机制**: 实现RPATH相对路径支持，用户无需设置环境变量
  - 程序自动从 `$ORIGIN/../lib` 查找动态库
  - 支持零配置使用：解压即用，无需复杂配置
- **智能路径查找系统** (`detector_path_utils.h`)
  - 自动查找模型文件和标定数据
  - 支持多级路径搜索和环境变量覆盖
  - 函数：`PathUtils::find_model()`, `PathUtils::find_calibration()`
- **极坐标系统增强**
  - 完善的双坐标系统：笛卡尔坐标 + 极坐标
  - 自动角度单位转换：`theta_degrees()` 方法
  - JSON配置文件支持极坐标原点偏移

#### 🔧 重要改进
- **C++17兼容性支持**: 升级构建系统支持C++17标准
  - CMakeLists.txt: `CMAKE_CXX_STANDARD 11` → `CMAKE_CXX_STANDARD 17`
  - 完全向后兼容，无需修改现有API代码
  - 支持现代C++特性：结构化绑定、if constexpr等
- **文件命名规范化** (避免命名冲突)
  - `common.h` → `detector_common_types.h`
  - `file_utils.h` → `detector_file_utils.h`
  - `file_utils.c` → `detector_file_utils.c`
  - `rim_basketball_postprocess.h` → `detector_rim_basketball_postprocess.h`
- **CMake构建系统优化**
  - RPATH设置：`INSTALL_RPATH "$ORIGIN/../lib"`
  - RKNN库查找路径优化和强制版本匹配
  - 智能依赖查找和路径管理
  - 更好的跨平台兼容性

#### 📦 打包和分发
- **用户友好的发布包结构**
  ```
  detector_lib/
  ├── bin/          # 预编译可执行程序
  ├── lib/          # 库文件 (包含 librknnrt.so)
  ├── include/      # 完整头文件
  ├── models/       # AI模型文件
  ├── data/         # 标定数据
  └── examples/     # 源码示例
  ```
- **零配置使用体验**
  - 解压后直接运行，无需安装步骤
  - 自动依赖解析，包含所有必需库

#### 🎯 示例程序更新
- 新增 `pose_camera_bytetrack_homography.cpp`：摄像头+ByteTrack+Homography+极坐标实时示例
  - 支持命令行：`<model> <cam_index> [calibration]`
  - 同步提供一键脚本 `run_pose_camera_bytetrack_homography.sh`
- 单图示例（`pose_image*`）默认关闭 ByteTrack，仅输出检测与坐标；摄像头示例默认启用 ByteTrack
- 所有示例统一使用 `PathUtils` 智能路径查找

#### 📚 文档全面更新
- **README.md**: 新增零配置使用指南，突出相对路径机制
- **USER_GUIDE.md**: 更新所有代码示例使用智能路径查找
- **项目结构文档**: 反映重命名后的文件结构

#### ⚠️ 破坏性变更
- 头文件重命名：需要更新 `#include` 语句
- API命名空间：推荐使用 `using namespace detector;`
- 路径配置：从硬编码路径迁移到智能路径查找

## [v1.0.2] - 2025-08-06

### ✨ 功能增强
- 新增极坐标系统支持
- 完善Homography坐标映射功能
- ByteTracker多目标跟踪集成

### 🔧 优化改进
- NPU内存管理优化
- 推理性能提升
- 错误处理机制完善

### 📝 文档更新
- API参考文档
- 使用示例和最佳实践

## [v1.0.1] - 2025-08-05

### 初始版本
- 基础姿态检测功能
- 篮筐篮球检测功能
- RKNN NPU加速支持
- CMake构建系统

---

## 版本说明

- **主版本号**: 重大架构变更或不兼容更新
- **次版本号**: 新功能添加，向下兼容
- **修订版本号**: Bug修复和小幅改进

## 升级指南

### 从 v1.0.2 升级到 v1.0.3

1. **更新头文件引用**:
   ```cpp
   // 旧版本
   #include "common.h"
   #include "file_utils.h"
   
   // 新版本
   #include "detector_common_types.h"
   #include "detector_file_utils.h"
   #include "detector_path_utils.h"  // 新增
   ```

2. **使用智能路径查找**:
   ```cpp
   // 旧版本 - 硬编码路径
   PoseDetectorLib detector("/path/to/model.rknn");
   
   // 新版本 - 智能路径查找
   std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
   PoseDetectorLib detector(model_path);
   ```

3. **利用相对路径机制**:
   - 使用新的发布包结构
   - 删除环境变量设置（如有）
   - 享受零配置使用体验

## 技术支持

如有问题或建议，请提交Issue到项目仓库。

---
*本更新日志遵循 [Keep a Changelog](https://keepachangelog.com/) 规范*