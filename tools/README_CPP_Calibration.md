# C++ 相机标定工具使用指南

本指南介绍如何在C++项目中使用Python相机标定工具生成的npz文件。

## 📋 目录

- [功能概述](#功能概述)
- [系统要求](#系统要求)
- [编译和安装](#编译和安装)
- [使用方法](#使用方法)
- [API 参考](#api-参考)
- [示例代码](#示例代码)
- [故障排除](#故障排除)

## 🎯 功能概述

该C++工具提供以下功能：

- **相机标定结果加载**：读取Python工具生成的相机标定npz文件
- **地面标定结果加载**：读取地面单应矩阵标定结果
- **图像矫正**：去除镜头畸变
- **坐标转换**：图像坐标 ↔ 地面坐标转换
- **距离计算**：计算地面上两点间的距离
- **验证功能**：验证标定结果的质量
- **原点设置**：设置地面坐标系的原点

## 📋 系统要求

### 必需依赖

- **OpenCV 4.x** 或更高版本
- **CMake 3.10** 或更高版本
- **C++11** 兼容的编译器

### 可选依赖

- **Eigen3**：用于高级矩阵运算（推荐）

### 支持的操作系统

- Linux (Ubuntu, CentOS, etc.)
- macOS (10.12+)
- Windows (10+ with MSVC 2017+)

## 🔧 编译和安装

### 1. 安装依赖

#### Ubuntu/Debian
```bash
# 安装OpenCV
sudo apt update
sudo apt install libopencv-dev cmake build-essential

# 可选：安装Eigen
sudo apt install libeigen3-dev
```

#### CentOS/RHEL/Fedora
```bash
# 安装OpenCV (使用包管理器或从源码编译)
sudo yum install opencv-devel cmake gcc-c++

# 或使用dnf (新版本)
sudo dnf install opencv-devel cmake gcc-c++

# 可选：安装Eigen
sudo yum install eigen3-devel
```

#### macOS
```bash
# 使用Homebrew安装
brew install opencv cmake eigen
```

#### Windows
使用vcpkg安装：
```bash
vcpkg install opencv eigen3
```

### 2. 编译项目

```bash
# 克隆或下载项目文件到本地目录
cd /path/to/calibration/tools

# 创建构建目录
mkdir build
cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 可选：安装到系统
sudo make install
```

### 3. 验证安装

```bash
# 运行示例程序
./calibration_example camera_calibration.npz ground_calibration.npz test_image.jpg

# 检查是否正常工作
echo $?
```

## 📖 使用方法

### 基本使用流程

```cpp
#include "cpp_calibration_loader.h"

int main() {
    // 1. 创建加载器实例
    CalibrationLoader loader;

    // 2. 加载相机标定结果
    if (!loader.loadCameraCalibration("camera_calibration.npz")) {
        std::cerr << "Failed to load camera calibration" << std::endl;
        return -1;
    }

    // 3. 加载地面标定结果
    if (!loader.loadGroundCalibration("ground_calibration.npz")) {
        std::cerr << "Failed to load ground calibration" << std::endl;
        return -1;
    }

    // 4. 使用功能
    cv::Mat input_image = cv::imread("input.jpg");
    cv::Mat undistorted;
    loader.undistortImage(input_image, undistorted);

    // 5. 坐标转换示例
    std::vector<cv::Point2f> image_points = {cv::Point2f(320, 240)};
    std::vector<cv::Point3f> ground_points;
    loader.imageToGround(image_points, ground_points, 0.0);

    return 0;
}
```

### 高级功能

#### 设置地面坐标系原点

```cpp
// 设置原点（图像坐标 -> 地面坐标）
cv::Point2f image_origin(320, 480);  // 图像中的原点
cv::Point2f ground_origin(0, 0);     // 对应的地面坐标
loader.setGroundOrigin(image_origin, ground_origin);
```

#### 验证标定质量

```cpp
// 验证标定结果
std::vector<std::string> test_images = {"test1.jpg", "test2.jpg"};
std::string report = loader.validateCalibration(test_images);
std::cout << report << std::endl;
```

#### 计算地面距离

```cpp
// 计算两点间的地面距离
cv::Point3f point1(100, 200, 0);  // 毫米单位
cv::Point3f point2(150, 250, 0);
double distance = loader.calculateGroundDistance(point1, point2);
std::cout << "Distance: " << distance << " mm" << std::endl;
```

## 📚 API 参考

### CalibrationLoader 类

#### 构造函数
```cpp
CalibrationLoader();
```

#### 主要方法

##### 加载标定结果
```cpp
bool loadCameraCalibration(const std::string& npz_path);
bool loadGroundCalibration(const std::string& npz_path);
```

##### 图像处理
```cpp
bool undistortImage(const cv::Mat& input_image, cv::Mat& output_image);
```

##### 坐标转换
```cpp
bool imageToGround(const std::vector<cv::Point2f>& image_points,
                  std::vector<cv::Point3f>& ground_points,
                  double z_height = 0.0);

bool groundToImage(const std::vector<cv::Point3f>& ground_points,
                  std::vector<cv::Point2f>& image_points);
```

##### 工具函数
```cpp
double calculateGroundDistance(const cv::Point3f& point1, const cv::Point3f& point2);
void setGroundOrigin(const cv::Point2f& origin_image_point,
                    const cv::Point2f& origin_ground_point = cv::Point2f(0, 0));
```

##### 验证功能
```cpp
std::string validateCalibration(const std::vector<std::string>& test_images);
```

##### 获取结果
```cpp
const CameraCalibrationResults& getCameraResults() const;
const GroundCalibrationResults& getGroundResults() const;
```

## 💡 示例代码

### 完整的篮球姿态检测应用

```cpp
#include "cpp_calibration_loader.h"
#include <opencv2/opencv.hpp>
#include <iostream>

class BasketballPoseDetector {
private:
    CalibrationLoader calibration_loader_;
    cv::CascadeClassifier ball_detector_;

public:
    BasketballPoseDetector() {}

    bool initialize(const std::string& camera_npz, const std::string& ground_npz) {
        // 加载标定文件
        if (!calibration_loader_.loadCameraCalibration(camera_npz)) {
            std::cerr << "Failed to load camera calibration" << std::endl;
            return false;
        }

        if (!calibration_loader_.loadGroundCalibration(ground_npz)) {
            std::cerr << "Failed to load ground calibration" << std::endl;
            return false;
        }

        // 设置地面原点（篮球场中心）
        cv::Point2f image_center(640, 360);  // 图像中心
        cv::Point2f ground_center(0, 0);     // 篮球场中心
        calibration_loader_.setGroundOrigin(image_center, ground_center);

        // 加载篮球检测器（如果有的话）
        // ball_detector_.load("basketball_cascade.xml");

        return true;
    }

    bool processFrame(const cv::Mat& input_frame,
                     cv::Mat& output_frame,
                     std::vector<cv::Point3f>& ball_positions) {

        // 1. 矫正图像畸变
        cv::Mat undistorted;
        if (!calibration_loader_.undistortImage(input_frame, undistorted)) {
            return false;
        }

        // 2. 检测篮球位置（这里使用简化的圆检测）
        std::vector<cv::Vec3f> circles;
        cv::Mat gray;
        cv::cvtColor(undistorted, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, 30, 200, 50, 10, 100);

        // 3. 转换为地面坐标
        std::vector<cv::Point2f> image_ball_positions;
        for (const auto& circle : circles) {
            cv::Point2f center(circle[0], circle[1]);
            image_ball_positions.push_back(center);
        }

        // 转换为地面坐标
        ball_positions.clear();
        if (!calibration_loader_.imageToGround(image_ball_positions, ball_positions, 0.0)) {
            return false;
        }

        // 4. 在输出图像上绘制结果
        output_frame = undistorted.clone();
        for (size_t i = 0; i < circles.size(); ++i) {
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);

            // 绘制圆
            cv::circle(output_frame, center, radius, cv::Scalar(0, 255, 0), 3);

            // 显示地面坐标
            if (i < ball_positions.size()) {
                cv::Point3f ground_pos = ball_positions[i];
                std::string coord_text = cv::format("(%.1f, %.1f)mm",
                                                  ground_pos.x, ground_pos.y);
                cv::putText(output_frame, coord_text,
                           cv::Point(center.x - radius, center.y - radius - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        return true;
    }

    // 计算球场上的距离和角度
    double calculateDistanceToBasket(const cv::Point3f& ball_pos) {
        // 假设篮筐位置为(0, 5000)mm（球场前方5米）
        cv::Point3f basket_pos(0, 5000, 0);
        return calibration_loader_.calculateGroundDistance(ball_pos, basket_pos);
    }
};

int main() {
    BasketballPoseDetector detector;

    if (!detector.initialize("camera_calibration.npz", "ground_calibration.npz")) {
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }

    cv::Mat frame, processed_frame;
    std::vector<cv::Point3f> ball_positions;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (detector.processFrame(frame, processed_frame, ball_positions)) {
            // 显示检测到的篮球位置
            for (const auto& pos : ball_positions) {
                double distance_to_basket = detector.calculateDistanceToBasket(pos);
                std::cout << "Ball at: (" << pos.x << ", " << pos.y << ") mm, "
                         << "Distance to basket: " << distance_to_basket << " mm" << std::endl;
            }

            cv::imshow("Basketball Detection", processed_frame);
        }

        if (cv::waitKey(30) == 27) break;  // ESC to exit
    }

    return 0;
}
```

## 🔧 故障排除

### 常见问题

#### 1. 编译错误：找不到OpenCV

**问题**：
```
fatal error: opencv2/opencv.hpp: No such file or directory
```

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt install libopencv-dev

# CentOS/RHEL
sudo yum install opencv-devel

# 检查安装
pkg-config --modversion opencv
```

#### 2. 运行时错误：无法读取npz文件

**问题**：
```
Failed to load camera calibration
```

**解决方案**：
- 检查npz文件路径是否正确
- 确认npz文件是由Python工具生成的
- 检查文件权限
- 验证OpenCV版本支持npz格式

#### 3. 坐标转换结果不准确

**问题**：
地面坐标转换结果偏差较大

**解决方案**：
- 验证地面标定的质量
- 检查原点设置是否正确
- 确保单应矩阵计算准确
- 考虑重新进行地面标定

#### 4. 内存不足错误

**问题**：
处理大图像时出现内存错误

**解决方案**：
- 减小图像分辨率
- 使用流式处理
- 增加系统内存
- 优化算法使用更少的内存

### 调试技巧

#### 启用调试模式
```cpp
#define DEBUG_MODE
// 在代码中添加调试输出
std::cout << "Debug: Matrix size = " << matrix.size() << std::endl;
```

#### 验证标定结果
```cpp
// 使用验证功能检查标定质量
std::vector<std::string> test_images = {"test1.jpg", "test2.jpg"};
std::string report = loader.validateCalibration(test_images);
std::cout << report << std::endl;
```

## 📞 支持

如果遇到问题，请：

1. 检查此文档的故障排除部分
2. 查看示例代码
3. 验证系统依赖是否正确安装
4. 查看控制台错误消息

## 📝 许可证

本工具遵循与Python版本相同的许可证。

---

**版本**: 1.0.0
**更新日期**: 2024年
**作者**: 基于Python相机标定工具的C++实现
