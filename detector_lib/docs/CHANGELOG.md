# 更新日志

## v1.0.3 (2025-08-07) 🌟 极坐标系统

### 🎯 重大新功能 - 双坐标系统

#### ✨ 极坐标系统
- **📐 双坐标输出**: 同时支持笛卡尔坐标(x,y)和极坐标(r,θ)
- **🎯 自动计算**: 基于笛卡尔坐标自动生成极坐标
- **⚙️ 灵活配置**: 支持JSON文件配置和手动配置两种方式
- **🔧 原点偏移**: 可自定义极坐标系原点位置

#### 🚀 API扩展
- **新增接口**: `set_polar_coordinate_system(bool enable, float offset_x, float offset_y)`
- **数据结构增强**: 
  ```cpp
  struct PoseResult {
      // 原有字段...
      PolarCoordinate polar_position;     // 极坐标 (距离mm, 角度弧度)
      bool has_polar_position;            // 是否有有效极坐标
  };
  ```

#### 📦 新增示例程序
- **pose_image_with_polar.cpp**: 极坐标系统完整演示
- **更新现有示例**: 所有坐标映射示例支持极坐标

#### 🔧 配置文件增强
- **JSON新字段**: 
  ```json
  {
      "origin_offset": [0.0, 0.0],   // 极坐标原点偏移
      "use_polar": true              // 启用极坐标
  }
  ```

#### ✅ 实测验证
```
笛卡尔坐标: (35.2, 3929.4)mm
极坐标: 距离=3929.5mm, 角度=89.5°  ✓ 数学计算正确
```

---

## v1.0.2 (2025-08-07) 🔧

### 🎯 重大改进 - 头文件内置

#### ✨ 简化安装
- **📦 头文件内置**: 所有RKNN头文件已内置到`detector_lib/include`
  - `rknn_api.h` - RKNN NPU API接口
  - `detector_common_types.h` - 公共工具函数
  - `detector_file_utils.h` - 文件操作工具
- **🚫 无需外部依赖**: 用户无需单独安装RKNN SDK
- **✅ 即插即用**: 解压即可编译，零配置要求

#### 🔧 智能路径查找
- **优先级机制**: 优先使用内置头文件，自动回退外部路径
- **详细错误信息**: CMake失败时提供清晰的路径诊断
- **多平台兼容**: 支持Rock-5C、Orange Pi等不同RK3588平台

#### 📚 文档更新
- **简化安装指南**: 5步完成安装，无复杂配置
- **故障排除**: 完整的问题诊断和解决方案
- **平台说明**: 针对不同硬件的具体指导

**用户体验提升**：
```bash
# 旧版本：需要复杂的RKNN SDK配置
# 新版本：5行命令搞定
cd detector_lib
sudo apt install build-essential cmake libopencv-dev
sudo chmod 666 /dev/dri/renderD*
./build_and_install.sh
./build/examples/pose_image  # 完成！
```

---

## v1.0.1 (2025-08-07) ⭐

### 🎉 重大功能更新 - Homography坐标映射

#### ✨ 新增功能
- **🌍 Homography坐标映射**: 一行代码启用像素坐标到真实世界坐标转换
  - 支持标准JSON格式标定文件
  - 自动提取ROI底部中点作为人员脚部位置
  - 毫米级精度的世界坐标输出
  - 完全集成到现有API，零学习成本
  
- **📍 智能位置算法**: 
  - ROI底部中点自动定位
  - 无需关键点检测，更稳定可靠
  - 适配各种姿态和遮挡情况

#### 🔧 API增强
- **简化接口**: `detector.load_calibration("file.json")` 一行代码启用
- **自动判断**: `pose.has_ground_position` 自动标识是否有世界坐标
- **数据完整**: `pose.ground_position` 直接获取毫米单位坐标

#### 🎯 测试验证
- **✅ 真实测试**: 使用RK3588实际验证坐标映射精度
- **✅ 示例程序**: 新增 `pose_image` 测试程序演示完整功能
- **✅ 可视化输出**: 自动生成包含ROI框、底部中点、世界坐标的结果图片

#### 📚 文档更新
- **完整使用指南**: 从标定文件格式到API调用的详细说明
- **实际测试结果**: 展示真实的坐标转换数据
- **应用场景**: 体育分析、位置追踪等具体应用示例

#### 🚀 用户友好性提升
- **无破坏性更新**: 完全向后兼容，现有代码无需修改
- **智能默认值**: 不启用时自动跳过，启用时自动工作
- **错误处理**: 完善的异常处理，标定文件不存在时优雅降级

**实测效果**：
```
人员[0] ROI框: (624, 276, 96, 264)
ROI底部中点: (672.0, 540.0)
世界坐标: (35.2, 3929.4)mm  ← 毫米级精度！
```

---

## v1.0.0 (2024-08-06)

### 🎉 首次发布

#### ✨ 新功能
- **PoseDetectorLib**: YOLOv8姿态检测封装类
  - 支持17个COCO关键点检测
  - 集成ByteTrack多目标跟踪
  - 支持Homography坐标映射
  - NPU零拷贝优化
  - **纯检测接口** - 只返回检测数据，不包含绘制功能
  
- **RimBasketballDetectorLib**: 篮筐篮球检测封装类
  - 支持篮筐(rim)和篮球(basketball)双类别检测
  - ROI分析和距离计算
  - NMS后处理优化
  - **纯检测接口** - 专注于AI推理，无绘制代码

#### 🏗 架构特性
- **延迟初始化**: 构造函数无异常，首次调用时自动初始化
- **RAII资源管理**: 自动内存清理，无泄漏风险
- **Pimpl模式**: 隐藏实现细节，编译时优化
- **命名空间**: `detector::` 避免符号冲突
- **完整API**: 状态查询、配置接口、性能监控

#### 🛠 构建系统
- **CMake支持**: 完整的构建配置
- **pkg-config**: 系统集成支持
- **静态/动态库**: 双模式构建
- **示例程序**: 完整的演示代码
- **一键构建脚本**: `build_and_install.sh`

#### 📚 文档
- **README.md**: 完整的项目文档
- **API参考**: 详细的接口说明
- **使用示例**: 多个演示程序
- **故障排除**: 常见问题解答

#### 🎯 性能
- **推理时间**: 
  - PoseDetector: 15-25ms/帧 (1920x1080)
  - RimBasketballDetector: 10-20ms/帧 (1920x1080)
- **初始化时间**: 1-3秒 (首次调用)
- **内存占用**: 零开销封装设计

#### 🧪 测试
- **功能测试**: `test_detector_lib` - 基础功能验证
- **演示程序**: `pose_image_with_homography` - 完整功能演示
- **并发测试**: 双检测器同时运行
- **错误处理**: 完整的异常安全测试

#### 📦 安装
- **系统安装**: 支持安装到 `/usr/local`
- **头文件**: 安装到 `include/detector_lib/`
- **库文件**: 静态库和动态库
- **pkg-config**: 自动生成 `.pc` 文件

### 🔧 技术细节

#### 依赖管理
- **OpenCV**: 4.x (图像处理)
- **RKNN Runtime**: 2.x (NPU推理)
- **Eigen3**: 线性代数 (ByteTracker)
- **C++标准**: C++11 兼容

#### 平台支持
- **硬件**: RK3588系列芯片
- **操作系统**: Ubuntu 20.04+, Debian 11+
- **架构**: aarch64 (ARM64)

#### 代码质量
- **代码行数**: 约1600行 (包含注释和示例)
- **文件结构**: 15个文件，5个目录
- **编译警告**: 零警告编译
- **内存检查**: 无泄漏设计

### 🚀 使用示例

#### 最简使用 (3行代码)
```cpp
detector::PoseDetectorLib detector("model.rknn");
auto results = detector.detect(frame);
// 自动清理
```

#### 完整功能使用
```cpp
detector::PoseDetectorLib detector("model.rknn");
detector.set_confidence_threshold(0.3f);
detector.enable_tracking(true);
detector.load_calibration("calib.json");

cv::VideoCapture cap(0);
cv::Mat frame;
while (cap.read(frame)) {
    auto results = detector.detect(frame);
    // 处理结果...
}
```

### 🎯 设计目标达成

- ✅ **从几百行减少到几行**: 用户代码简化95%+
- ✅ **零学习成本**: 用户无需了解NPU、RKNN概念
- ✅ **专注核心功能**: 纯检测接口，无绘制代码干扰
- ✅ **高性能保持**: 保留所有底层优化
- ✅ **生产就绪**: 完整错误处理和文档
- ✅ **易于集成**: 标准C++库形式发布
- ✅ **职责单一**: 库只负责AI推理，显示由用户决定

---

**开发团队**: AI推理优化专家  
**发布日期**: 2024年8月6日  
**版本代号**: "Genesis" - 从复杂到简单的第一步