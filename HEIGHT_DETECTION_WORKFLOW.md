# 身高检测系统开发流程

## 📋 项目概述

**项目名称**: Height Detection System (身高检测系统)  
**核心功能**: 基于YOLOv8姿态检测的实时身高测量系统  
**应用场景**: 智能监控、体育分析、健康评估

## 🏗️ 项目架构

```
yolov8_pose_basketball/
├── pose_analysis/          # 🔬 研发环境 (实验室)
├── detector_lib/          # 🚀 生产环境 (工厂)
│   ├── tests/            # 🧪 质量控制中心
│   └── examples/         # 🎯 应用示例
└── HEIGHT_DETECTION_WORKFLOW.md
```

## 🔄 三阶段开发流程

### 阶段1: 🔬 研发环境 (`pose_analysis/`)

**目标**: 快速原型开发和算法验证  
**环境**: 完整调试工具 + 灵活配置

#### 1.1 功能开发

```bash
# 进入研发环境
cd pose_analysis/

# 编辑核心模块
vim src/height_detector.cpp          # 身高检测算法
vim include/height_detector.h        # 接口定义
vim include/pose_analysis_types.h    # 数据结构
```

**核心模块职责:**
- `HeightDetector`: 身高计算引擎
- `BallRequestDetector`: 要球手势检测 (辅助功能)
- `IDPriorityManager`: ID管理和跟踪
- `PoseAnalyzer`: 统一分析接口

#### 1.2 快速验证

```bash
# 构建研发版本 (包含调试工具)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# 运行调试示例
./examples/simple_pose_analysis_example
./examples/height_measurement_test

# 查看调试输出
./examples/pose_analysis_with_images  # 图片批处理验证
```

**研发阶段特性:**
- ✅ 丰富的调试信息输出
- ✅ 实时参数调整和可视化
- ✅ 数据记录和分析工具
- ✅ 多种测试图片和场景

#### 1.3 配置优化

```bash
# 编辑配置文件
vim data/pose_analysis_config.json

# 测试不同参数组合
{
  "height_detection": {
    "filter_type": "kalman",           # median/kalman/moving_average
    "window_size": 15,
    "stability_threshold_mm": 50.0
  }
}
```

---

### 阶段2: 🧪 质量控制 (`detector_lib/tests/`)

**目标**: 全面测试验证和质量保证  
**环境**: 完备测试框架 + 性能分析

#### 2.1 单元测试

```bash
cd detector_lib/tests/

# 构建测试套件
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

# 运行身高检测专项测试
./pose_analysis_tests --gtest_filter="HeightDetectorTest.*"
```

**测试覆盖:**
```cpp
// test_height_detector.cpp
TEST(HeightDetectorTest, BasicCalculation) {
    // 基础身高计算精度测试
}

TEST(HeightDetectorTest, MultiFrameStability) {
    // 多帧稳定性测试
}

TEST(HeightDetectorTest, EdgeCaseHandling) {
    // 边缘情况处理测试
}
```

#### 2.2 性能测试

```bash
# 性能基准测试
make pose_analysis_benchmarks
./pose_analysis_benchmarks

# 内存泄漏检测
make memcheck
cat valgrind-out.txt

# 线程安全测试
cmake .. -DENABLE_TSAN=ON
make run_tests
```

#### 2.3 集成测试

```bash
# 完整工作流测试
./test_integration

# 长时间稳定性测试
./pose_analysis_perf_tests --duration=3600  # 1小时测试
```

**质量指标:**
- 测试覆盖率 > 90%
- 内存泄漏 = 0
- 身高测量误差 < 2cm
- 处理延迟 < 10ms

---

### 阶段3: 🚀 生产集成 (`detector_lib/`)

**目标**: 优化部署和生产应用  
**环境**: 高性能 + 稳定性优先

#### 3.1 代码同步

```bash
# 从研发环境同步最新代码
rsync -av pose_analysis/src/height_detector.cpp detector_lib/src/
rsync -av pose_analysis/include/height_detector.h detector_lib/include/

# 检查API兼容性
diff -u pose_analysis/include/pose_analysis_types.h \
        detector_lib/include/pose_analysis_types.h
```

#### 3.2 生产构建

```bash
cd detector_lib/

# 优化构建 (移除调试开销)
mkdir build_release && cd build_release
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG_TOOLS=OFF
make -j$(nproc)

# 生成部署包
make package
```

#### 3.3 集成验证

```bash
# 运行生产级示例
./examples/height_measurement_roi_based
./examples/yolov8_pose_with_analysis

# 实际场景测试
./dual_camera_detector ../models/Q_yolov8_pose.rknn \
                      ../models/Q_Rim_Basketball_724_JZ.rknn
```

**生产特性:**
- ⚡ 零拷贝内存优化
- 🎯 精简API接口
- 📊 实时性能监控
- 🛡️ 异常恢复机制

---

## 🔧 开发工具和脚本

### 自动化脚本

```bash
# 研发环境快速构建
./pose_analysis/scripts/build.sh -d --asan

# 完整测试套件
./detector_lib/scripts/build_pose_analysis.sh --enable-tests

# 生产部署
./detector_lib/scripts/package.sh --release
```

### 调试工具

```bash
# 性能分析
perf record -g ./height_detection_test
perf report

# 内存分析  
valgrind --tool=massif ./height_detection_test
ms_print massif.out.xxx
```

---

## 📊 质量控制检查清单

### 研发阶段 ✅
- [ ] 算法精度验证 (±2cm误差范围)
- [ ] 多种测试场景覆盖
- [ ] 参数配置优化完成
- [ ] 调试信息完整清晰

### 测试阶段 ✅  
- [ ] 单元测试通过率 100%
- [ ] 代码覆盖率 ≥ 90%
- [ ] 内存泄漏检测通过
- [ ] 性能基准达标 (≤10ms延迟)
- [ ] 线程安全验证通过

### 生产阶段 ✅
- [ ] API接口稳定
- [ ] 部署包完整
- [ ] 实际场景测试通过
- [ ] 性能优化达标
- [ ] 错误处理健壮

---

## 🚀 版本发布流程

### 版本命名规范
```
v1.0.0 - 基础身高检测功能
v1.1.0 - 多帧平滑优化
v1.2.0 - 云台追踪集成
v2.0.0 - 架构重构
```

### 发布检查
1. **代码审查** - 所有变更经过review
2. **测试验证** - 完整测试套件通过
3. **性能回归** - 确保无性能下降
4. **文档更新** - API文档和使用说明
5. **部署测试** - 生产环境验证

---

## 📚 相关文档

- [技术方案设计](./pose_analysis/doc/新功能记录.md)
- [API参考文档](./detector_lib/include/height_detector.h)  
- [部署指南](./CLAUDE.md)
- [测试报告](./detector_lib/tests/test_reports/)

---

## 🎯 下一步计划

### 短期目标 (1-2周)
- [ ] 完善现有测试用例
- [ ] 优化卡尔曼滤波算法
- [ ] 增强边缘情况处理

### 中期目标 (1个月)  
- [ ] 多人身高同时检测
- [ ] 云台追踪自动化
- [ ] 实时校准功能

### 长期目标 (3个月)
- [ ] 深度学习身高预测
- [ ] 多摄像头融合
- [ ] 云端部署支持

---

**备注**: 本文档应随开发进度持续更新，确保流程的时效性和准确性。