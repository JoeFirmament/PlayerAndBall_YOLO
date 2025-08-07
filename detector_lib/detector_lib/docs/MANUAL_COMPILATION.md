# 手动编译指南

本文档介绍如何使用 `g++` 命令手动编译和使用 YOLOv8 检测器封装库。

## 📋 目录
- [环境准备](#环境准备)
- [编译静态库](#编译静态库)
- [编译测试程序](#编译测试程序)
- [使用封装库](#使用封装库)
- [完整示例](#完整示例)
- [故障排除](#故障排除)

---

## 🔧 环境准备

### 系统要求
- **平台**: RK3588 (Linux aarch64)
- **编译器**: GCC 7+ 支持 C++11
- **依赖库**: OpenCV 4.x, RKNN Runtime

### 检查依赖
```bash
# 检查 OpenCV
pkg-config --cflags --libs opencv4

# 检查 RKNN 库
ls -la libs/librknnrt.so

# 检查编译器版本
g++ --version
```

---

## 📚 编译静态库

### 1. 编译源文件为对象文件

```bash
# 设置编译选项
CXXFLAGS="-std=c++11 -O3 -fPIC -Wall"
INCLUDES="-Iinclude -Isrc -I/usr/include/eigen3"
OPENCV_FLAGS="$(pkg-config --cflags opencv4)"

# 编译核心库文件
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS -c src/detector_lib.cpp -o detector_lib.o
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS -c src/PoseDetectorLib.cpp -o PoseDetectorLib.o  
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS -c src/RimBasketballDetectorLib.cpp -o RimBasketballDetectorLib.o
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS -c src/internal/detector_common.cpp -o detector_common.o

# 编译依赖文件
gcc -std=c99 -O3 -fPIC $INCLUDES -c src/detector_file_utils.c -o detector_file_utils.o
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS -c src/rim_basketball_postprocess_simple.cpp -o rim_basketball_postprocess.o
```

### 2. 创建静态库

```bash
# 打包成静态库
ar rcs libdetector_lib.a detector_lib.o PoseDetectorLib.o RimBasketballDetectorLib.o detector_common.o detector_file_utils.o rim_basketball_postprocess.o

# 验证静态库
ls -la libdetector_lib.a
nm libdetector_lib.a | grep detector  # 查看符号表
```

---

## 🧪 编译测试程序

### 基础姿态检测程序

```bash
# 编译 pose_image
g++ -std=c++11 -O2 \
    -Iinclude -Isrc \
    $(pkg-config --cflags opencv4) \
    examples/pose_image.cpp \
    -o pose_image \
    libdetector_lib.a \
    $(pkg-config --libs opencv4) \
    -lrknnrt -lpthread

# 运行测试
./pose_image
```

### 姿态+Homography坐标映射程序

```bash  
# 编译 pose_image_with_homography
g++ -std=c++11 -O2 \
    -Iinclude -Isrc \
    $(pkg-config --cflags opencv4) \
    examples/pose_image_with_homography.cpp \
    -o pose_image_with_homography \
    libdetector_lib.a \
    $(pkg-config --libs opencv4) \
    -lrknnrt -lpthread

# 运行测试  
./pose_image_with_homography
```

### 篮筐篮球检测程序

```bash
# 编译 rim_basketball_image
g++ -std=c++11 -O2 \
    -Iinclude -Isrc \
    $(pkg-config --cflags opencv4) \
    examples/rim_basketball_image.cpp \
    -o rim_basketball_image \
    libdetector_lib.a \
    $(pkg-config --libs opencv4) \
    -lrknnrt -lpthread

# 运行测试
./rim_basketball_image
```

### 综合功能测试程序

```bash
# 编译 test_detector_lib  
g++ -std=c++11 -O2 \
    -Iinclude -Isrc \
    $(pkg-config --cflags opencv4) \
    examples/test_detector_lib.cpp \
    -o test_detector_lib \
    libdetector_lib.a \
    $(pkg-config --libs opencv4) \
    -lrknnrt -lpthread

# 运行测试
./test_detector_lib
```

---

## 💡 使用封装库

### 创建你的项目

```bash
mkdir my_detection_project && cd my_detection_project
```

### 基础姿态检测示例 (`my_pose_app.cpp`)

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector_lib.h"

int main() {
    // 1. 创建检测器
    detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
    
    // 2. 加载图片  
    cv::Mat image = cv::imread("test_image.jpg");
    if (image.empty()) {
        std::cerr << "无法加载图片" << std::endl;
        return -1;
    }
    
    // 3. 执行检测
    auto results = detector.detect(image);
    
    // 4. 处理结果
    std::cout << "检测到 " << results.size() << " 个人" << std::endl;
    for (const auto& pose : results) {
        std::cout << "人员ID: " << pose.person_id 
                  << ", 置信度: " << pose.confidence << std::endl;
    }
    
    return 0;
}
```

### 编译你的项目

```bash
# 复制库文件和头文件到你的项目
cp /path/to/detector_lib/libdetector_lib.a .
cp -r /path/to/detector_lib/include .

# 编译你的应用程序
g++ -std=c++11 -O2 \
    -Iinclude \
    $(pkg-config --cflags opencv4) \
    my_pose_app.cpp \
    -o my_pose_app \
    libdetector_lib.a \
    $(pkg-config --libs opencv4) \
    -lrknnrt -lpthread

# 运行
./my_pose_app
```

### 带Homography坐标映射的示例 (`my_advanced_app.cpp`)

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector_lib.h"

int main() {
    // 1. 创建检测器
    detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
    
    // 2. 启用跟踪和坐标映射
    detector.enable_tracking(true);
    detector.load_calibration("data/calibration.json");
    
    // 3. 检测并获取世界坐标
    cv::Mat image = cv::imread("test_image.jpg");
    auto results = detector.detect(image);
    
    for (const auto& pose : results) {
        std::cout << "人员ID: " << pose.person_id << std::endl;
        
        if (pose.has_ground_position) {
            std::cout << "世界坐标: (" 
                      << pose.ground_position.x << ", " 
                      << pose.ground_position.y << ")mm" << std::endl;
        }
    }
    
    return 0;
}
```

### 篮筐篮球检测示例 (`my_basketball_app.cpp`)

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "detector_lib.h"

int main() {
    // 1. 创建篮筐篮球检测器
    detector::RimBasketballDetectorLib detector("models/Q_Rim_Basketball_724_JZ.rknn");
    
    // 2. 配置参数
    detector.set_confidence_threshold(0.5f);
    detector.set_nms_threshold(0.4f);
    
    // 3. 执行检测
    cv::Mat image = cv::imread("basketball_scene.jpg");
    auto results = detector.detect(image);
    
    // 4. 分析结果
    for (const auto& obj : results) {
        std::cout << "检测到: " << obj.class_name 
                  << ", 置信度: " << obj.confidence << std::endl;
                  
        if (obj.class_id == 0) {  // basketball
            std::cout << "  距篮筐: " << obj.distance_to_rim << "px" << std::endl;
            std::cout << "  在ROI内: " << (obj.is_in_rim_roi ? "是" : "否") << std::endl;
        }
    }
    
    return 0;
}
```

---

## 📖 完整示例

### 一键编译脚本 (`build_my_app.sh`)

```bash
#!/bin/bash

echo "编译自定义检测应用程序..."

# 设置变量
LIB_PATH="../../detector_lib"
INCLUDES="-I${LIB_PATH}/include"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"
CXXFLAGS="-std=c++11 -O2 -Wall"

# 编译应用程序
g++ $CXXFLAGS $INCLUDES $OPENCV_FLAGS \
    my_pose_app.cpp \
    -o my_pose_app \
    ${LIB_PATH}/libdetector_lib.a \
    -lrknnrt -lpthread

if [ $? -eq 0 ]; then
    echo "✅ 编译成功!"
    echo "运行: ./my_pose_app"
else
    echo "❌ 编译失败!"
    exit 1
fi
```

### Makefile 示例

```makefile
# Makefile for YOLOv8 Detection App

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall
INCLUDES = -Iinclude
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)
LIBS = -lrknnrt -lpthread

# 目标和源文件
TARGETS = my_pose_app my_basketball_app
LIB = libdetector_lib.a

all: $(TARGETS)

my_pose_app: my_pose_app.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $< -o $@ $(LIB) $(LIBS)

my_basketball_app: my_basketball_app.cpp $(LIB)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPENCV_FLAGS) $< -o $@ $(LIB) $(LIBS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
```

---

## 🔍 故障排除

### 常见编译错误

#### 1. 找不到头文件
```bash
fatal error: detector_lib.h: No such file or directory
```
**解决方案**: 确保 `-I` 参数指向正确的头文件目录
```bash
-Iinclude -Isrc
```

#### 2. 找不到 OpenCV
```bash  
fatal error: opencv2/opencv.hpp: No such file or directory
```
**解决方案**: 安装 OpenCV 并验证 pkg-config
```bash
sudo apt install libopencv-dev
pkg-config --cflags opencv4
```

#### 3. 链接 RKNN 库失败
```bash
undefined reference to 'rknn_init'
```
**解决方案**: 确保 RKNN 库路径正确
```bash
# 检查库文件
ls -la libs/librknnrt.so

# 添加库路径 (如果需要)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./libs
```

#### 4. 符号未定义
```bash
undefined reference to 'detector::PoseDetectorLib::detect'
```
**解决方案**: 确保静态库包含所有必要的对象文件
```bash
# 验证静态库内容
nm libdetector_lib.a | grep PoseDetectorLib
```

### 运行时错误

#### NPU 权限不足
```bash
❌ 错误: RKNN初始化失败 (错误码: -1)
```
**解决方案**: 设置 NPU 设备权限
```bash
sudo chmod 666 /dev/dri/renderD*
```

#### 模型文件找不到
```bash  
❌ 错误: 找不到模型文件
```
**解决方案**: 确保模型文件路径正确
```bash
ls -la models/Q_yolov8_pose.rknn
```

### 调试技巧

#### 查看库符号
```bash
# 查看静态库导出的符号
nm -D libdetector_lib.a

# 查看可执行文件依赖
ldd my_pose_app
```

#### 编译详细信息
```bash
# 添加 -v 查看详细编译过程
g++ -v -std=c++11 ... 

# 只编译不链接，检查语法
g++ -c -std=c++11 ...
```

#### 运行时调试
```bash
# 设置详细日志
export RKNN_LOG_LEVEL=1

# 使用 gdb 调试
gdb ./my_pose_app
```

---

## 📝 总结

通过本指南，你可以：

1. **手动编译静态库** - 完全控制编译过程
2. **编译测试程序** - 验证库功能正常
3. **创建自己的应用** - 集成检测功能到项目中
4. **解决常见问题** - 快速排除编译和运行错误

### 推荐的开发流程
1. 先使用 CMake 构建系统验证环境
2. 了解 g++ 手动编译方法
3. 根据项目需求选择合适的编译方式
4. 参考示例代码开发自己的应用

### 进一步参考
- [README.md](../README.md) - 库使用指南
- [API文档](DetectorAPI_Usage.md) - 详细接口说明  
- [examples/](../examples/) - 完整示例代码