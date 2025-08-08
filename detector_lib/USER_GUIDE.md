# Detector Lib 用户使用指南 📚

## ⚠️ RKNN版本兼容性重要说明

**本库使用最新的RKNN模型Version 6格式，已内置匹配的Runtime库。**

如果您的系统出现以下错误：
```bash
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!
```

**解决方法：** 
- ✅ 使用提供的预编译程序，已自动配置
- ✅ 重新编译项目，CMake会自动使用内置RKNN库
- ❌ 不要手动修改系统RKNN库

## 🎯 v1.0.3 - 相对路径机制使用指南

### 🚀 零配置使用 (推荐 ⭐)

#### 解压即用，无需复杂配置！

**发布包结构 (2025年最新)**：
```
detector_lib/
├── bin/                          # 可执行程序 (预编译)
│   ├── pose_image               # 基础姿态检测
│   ├── pose_image_with_polar    # 极坐标演示 ⭐ 新功能
│   ├── pose_image_with_homography # Homography坐标映射
│   └── rim_basketball_image     # 篮筐篮球检测
├── lib/                          # 库文件 (包含所有依赖)
│   ├── libdetector_lib.so*      # 我们的封装库
│   └── librknnrt.so            # RKNN运行时库 ⭐ 内置
├── include/                      # 完整头文件
│   ├── PoseDetectorLib.h        # 主接口 ⭐
│   ├── detector_path_utils.h    # 智能路径查找 ⭐
│   └── detector_types.h         # 数据类型 (含极坐标)
├── models/                       # AI模型文件
│   ├── Q_yolov8_pose.rknn
│   └── Q_Rim_Basketball_724_JZ.rknn
├── data/                         # 标定数据
│   └── 2025_8_6_1280_720.json  # 极坐标配置
└── examples/                     # 源码示例
    └── *.cpp
```

**用户使用步骤**:
```bash
# 1. 解压 (唯一步骤)
tar -xzf yolov8_detector_lib_rk3588_v1.0.3.tar.gz

# 2. 直接运行 (零配置！)
cd detector_lib/bin/
./pose_image_with_polar    # 🌟 极坐标功能演示
./pose_image               # 基础姿态检测  
./rim_basketball_image     # 篮筐篮球检测

# 就这么简单！无需任何环境变量或路径配置
```

✅ **RPATH相对路径机制** - 程序自动找到 ../lib/ 中的所有库
✅ **智能文件查找** - 模型和数据文件自动定位
✅ **完整依赖包含** - 无需额外安装RKNN SDK

#### 1.2 用户安装步骤

**Step 1: 安装基础依赖**
```bash
# Ubuntu/Debian系统
sudo apt update
sudo apt install libopencv-dev libeigen3-dev

# 或根据用户的Linux发行版提供相应命令
```

**Step 2: 安装库文件**
```bash
# 解压并进入目录
tar -xzf detector_lib_package.tar.gz
cd detector_lib_package

# 一键安装（推荐）
sudo ./install.sh

# 或手动安装
sudo cp lib/* /usr/local/lib/
sudo cp include/* /usr/local/include/
sudo cp models/* /usr/local/share/detector_lib/models/
sudo ldconfig  # 更新动态库缓存
```

**Step 3: 测试安装**
```bash
# 编译测试程序
g++ examples/sample_code.cpp -ldetector_lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -o test_detector

# 运行测试
./test_detector test_image.jpg
```

### 方案2: 使用pkg-config配置 (开发者推荐)

#### 2.1 创建pkg-config文件
```bash
# 用户系统上创建 /usr/local/lib/pkgconfig/detector_lib.pc
sudo tee /usr/local/lib/pkgconfig/detector_lib.pc << 'EOF'
prefix=/usr/local
exec_prefix=${prefix}
libdir=${exec_prefix}/lib
includedir=${prefix}/include

Name: DetectorLib
Description: YOLOv8 Pose and Basketball Detection Library for RK3588
Version: 1.0.0
Requires: opencv4
Libs: -L${libdir} -ldetector_lib -lrknn_api -lpthread
Cflags: -I${includedir}
EOF

# 更新pkg-config缓存
sudo ldconfig
```

#### 2.2 用户编译程序
```bash
# 简单编译（推荐方式）
g++ your_program.cpp $(pkg-config --cflags --libs detector_lib) -o your_program

# 查看编译选项
pkg-config --cflags --libs detector_lib
```

### 方案3: Docker容器化部署

#### 3.1 创建Dockerfile
```dockerfile
FROM arm64v8/ubuntu:20.04

# 安装依赖
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制库文件
COPY lib/* /usr/local/lib/
COPY include/* /usr/local/include/
COPY models/* /usr/local/share/detector_lib/models/

# 更新动态库缓存
RUN ldconfig

# 设置工作目录
WORKDIR /app

# 示例：运行用户程序
# COPY user_app .
# CMD ["./user_app"]
```

#### 3.2 用户使用Docker
```bash
# 构建镜像
docker build -t detector_lib:1.0 .

# 运行容器
docker run -it --privileged \
  -v /dev/dri:/dev/dri \
  -v $(pwd):/app \
  detector_lib:1.0 bash
```

## 📝 用户代码示例 (v1.0.3最新)

### 示例1: 智能路径姿态检测 ⭐ (推荐)
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>

using namespace detector;

int main() {
    // 1. 智能查找模型文件 (无需硬编码路径)
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    // 2. 检测图片
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    // 3. 处理结果
    for (const auto& pose : results) {
        printf("检测到人员 ID:%d，置信度:%.2f\n", 
               pose.person_id, pose.confidence);
        printf("位置: (%d,%d,%d,%d)\n", 
               pose.bbox.x, pose.bbox.y, 
               pose.bbox.width, pose.bbox.height);
    }
    
    return 0;
}
```

### 示例2: 带可视化的检测
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>

using namespace detector;

int main() {
    // 智能查找模型文件
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    // 绘制检测结果
    cv::Mat display_image = image.clone();
    for (const auto& pose : results) {
        // 绘制边界框
        cv::rectangle(display_image, pose.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 绘制关键点
        for (size_t i = 0; i < pose.keypoints.size(); i++) {
            if (pose.keypoint_scores[i] > 0.5) {  // 只显示高置信度关键点
                cv::circle(display_image, pose.keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // 显示跟踪ID和置信度
        std::string text = "ID:" + std::to_string(pose.person_id) + 
                          " Conf:" + std::to_string(pose.confidence).substr(0, 4);
        cv::putText(display_image, text, 
                   cv::Point(pose.bbox.x, pose.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }
    
    // 保存结果
    cv::imwrite("detection_result.jpg", display_image);
    printf("检测完成，结果保存到 detection_result.jpg\n");
    
    return 0;
}
```

### 示例3: 双坐标系统 ⭐ 极坐标功能
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>

using namespace detector;

int main() {
    // 智能路径查找
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    // 启用双坐标系统 (智能查找标定文件)
    std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
    if (!calib_path.empty()) {
        detector.load_calibration(calib_path);
    }
    
    cv::Mat image = cv::imread("test.jpg");
    auto results = detector.detect(image);
    
    for (const auto& pose : results) {
        printf("人员 ID:%d，置信度:%.2f\n", pose.person_id, pose.confidence);
        
        // 笛卡尔坐标系
        if (pose.has_ground_position) {
            printf("笛卡尔坐标: (%.1f, %.1f)mm\\n", 
                   pose.ground_position.x, pose.ground_position.y);
        }
        
        // 极坐标系 (距离+角度)
        if (pose.has_polar_position) {
            printf("极坐标: 距离=%.1fmm, 角度=%.1f°\\n", 
                   pose.polar_position.r, pose.polar_position.theta_degrees());
        }
    }
    
    return 0;
}
```

### 示例4: 篮筐篮球检测
```cpp
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>

using namespace detector;

int main() {
    // 智能查找模型文件
    std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
    RimBasketballDetectorLib detector(model_path);
    
    cv::Mat image = cv::imread("basketball_court.jpg");
    auto results = detector.detect(image);
    
    for (const auto& result : results) {
        printf("检测到 %s，置信度:%.2f\n", 
               result.class_name.c_str(), result.confidence);
        printf("位置: (%d,%d,%d,%d)\n", 
               result.bbox.x, result.bbox.y, 
               result.bbox.width, result.bbox.height);
    }
    
    return 0;
}
```

### 示例5: 视频实时检测
```cpp
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"
#include <opencv2/opencv.hpp>

using namespace detector;

int main() {
    // 智能查找模型文件
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    PoseDetectorLib detector(model_path);
    
    cv::VideoCapture cap(0);  // 摄像头
    cv::Mat frame;
    
    while (cap.read(frame)) {
        auto start = cv::getTickCount();
        
        // 检测
        auto results = detector.detect(frame);
        
        // 计算FPS
        auto end = cv::getTickCount();
        double fps = cv::getTickFrequency() / (end - start);
        
        // 显示结果
        std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 4);
        cv::putText(frame, fps_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        
        cv::imshow("实时检测", frame);
        if (cv::waitKey(1) == 27) break;  // ESC退出
    }
    
    return 0;
}
```

## 🔧 编译配置

### Makefile示例
```makefile
CXX = g++
CXXFLAGS = -std=c++11 -O2
INCLUDES = -I/usr/local/include
LIBS = -L/usr/local/lib -ldetector_lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

# 目标文件
TARGET = my_detector_app
SOURCES = main.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) $(LIBS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean
```

### CMakeLists.txt示例
```cmake
cmake_minimum_required(VERSION 3.10)
project(MyDetectorApp)

set(CMAKE_CXX_STANDARD 11)

# 查找依赖
find_package(PkgConfig REQUIRED)
pkg_check_modules(DETECTOR_LIB REQUIRED detector_lib)

find_package(OpenCV REQUIRED)

# 创建可执行文件
add_executable(my_detector_app main.cpp)

# 链接库
target_link_libraries(my_detector_app 
    ${DETECTOR_LIB_LIBRARIES}
    ${OpenCV_LIBS}
)

# 包含头文件
target_include_directories(my_detector_app PRIVATE 
    ${DETECTOR_LIB_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
```

## 🛠️ 用户故障排除

### 常见错误及解决方案

#### ❌ 错误1: RKNN版本不兼容 (最常见)
```bash
# 错误信息:
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!

# 诊断步骤:
ldd ./your_program | grep rknn
# 如果显示系统库: librknnrt.so => /lib/librknnrt.so

# ✅ 解决方案1: 重新编译 (推荐)
cd detector_lib && rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)

# ✅ 解决方案2: 检查RPATH设置
readelf -d ./your_program | grep -E "RPATH|RUNPATH"
# 应该看到: Library runpath: [$ORIGIN/../lib]

# ⚠️ 解决方案3: 手动库替换 (需要谨慎)
sudo cp detector_lib/lib/librknnrt.so /lib/
sudo ldconfig
```

**错误2: 找不到动态库**
```bash
# 错误信息: error while loading shared libraries: libdetector_lib.so.1
# 解决方案:
sudo ldconfig
# 或添加库路径
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

**错误3: NPU设备权限问题**
```bash
# 错误信息: RKNN 初始化失败
# 解决方案:
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video $USER
# 重新登录生效
```

**错误4: 模型文件找不到**
```bash
# 错误信息: 无法加载模型文件
# 解决方案: 检查文件路径和权限
ls -la /usr/local/share/detector_lib/models/
sudo chmod 644 /usr/local/share/detector_lib/models/*.rknn
```

## 📦 分发建议

### 给用户的分发包结构
```bash
# 创建分发包脚本
./create_distribution_package.sh
```

分发包应包含：
1. **编译好的库文件** (动态库 + 静态库)
2. **头文件** (detector_lib.h)
3. **RKNN模型文件**
4. **示例代码和测试程序**
5. **一键安装脚本**
6. **详细的README和故障排除指南**

这样用户就能够：
- 🚀 **快速上手**: 3行代码即可使用
- 🔧 **灵活集成**: 支持多种编译方式
- 🐳 **容器化部署**: 提供Docker支持
- 🛠️ **故障排除**: 完整的调试指南

用户无需了解RKNN底层细节，只需要关注业务逻辑的实现。