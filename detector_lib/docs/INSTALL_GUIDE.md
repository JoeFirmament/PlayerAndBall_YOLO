# DetectorLib 安装指南

## ⚠️ RKNN版本兼容性重要说明

**本库内置最新RKNN Runtime库，支持Version 6模型格式。**

如果您在其他系统上遇到版本兼容问题：
```bash
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!
```

**原因：** 您的系统RKNN库版本过旧。  
**解决：** 使用本库提供的编译系统，会自动处理版本问题。

## 🎆 v1.0.3 重大更新 - 相对路径机制 + 零配置使用

**重大改进！** v1.0.3版本实现了真正的零配置使用：

✅ **RPATH相对路径机制** - 程序自动查找依赖库  
✅ **智能文件查找** - 模型文件和数据文件自动定位  
✅ **完整依赖打包** - 包含匹配版本的librknnrt.so等所有必需库  
✅ **版本隔离** - 避免与系统RKNN库版本冲突  
✅ **解压即用** - 无需安装步骤，无需设置环境变量

## 🚀 零配置使用 (v1.0.3 最新 ⭐)

### 方案1：直接使用预编译包 (推荐)

```bash
# 1. 解压即可使用！
tar -xzf yolov8_detector_lib_rk3588_v1.0.3.tar.gz
cd detector_lib

# 2. 设置NPU权限 (仅需一次)
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video $USER

# 3. 直接运行示例程序
cd bin/
./pose_image_with_polar    # 极坐标功能演示
./pose_image               # 基础姿态检测  
./rim_basketball_image     # 篮筐篮球检测
```

**就这么简单！** 无需编译，无需安装，无需配置环境变量。

### 方案2：从源码编译

```bash
# 1. 解压detector_lib
tar -xzf detector_lib_source.tar.gz
cd detector_lib

# 2. 安装系统依赖
sudo apt update
sudo apt install build-essential cmake libopencv-dev libeigen3-dev

# 3. 设置NPU权限
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video $USER

# 4. 一键编译
./build_and_install.sh

# 5. 测试功能
cd build/examples
./pose_image_with_polar    # 最新功能
```

---

如果你仍然遇到编译错误，请参考下面的解决方案。

## 🚨 遇到问题的解决方案

如果你遇到了以下错误（通常在旧版本中）：
```
CMake Error at CMakeLists.txt:XX (message):
  RKNN header not found
```

这表示系统无法找到RKNN头文件，请按以下步骤解决：

## 🛠 解决方案

### 方案1: 创建RKNN SDK目录结构（推荐）

在detector_lib同级目录创建3rdparty结构：

```bash
cd ~/Qworkspace
mkdir -p 3rdparty/rknpu2/include
mkdir -p 3rdparty/rknpu2/Linux/aarch64

# 将RKNN头文件复制到正确位置
cp /path/to/your/rknn_sdk/include/* 3rdparty/rknpu2/include/
cp /path/to/your/rknn_sdk/lib/* 3rdparty/rknpu2/Linux/aarch64/
```

### 方案2: 系统安装RKNN SDK

将RKNN SDK安装到系统路径：

```bash
sudo mkdir -p /usr/include/rknn
sudo cp /path/to/your/rknn_sdk/include/* /usr/include/rknn/
sudo cp /path/to/your/rknn_sdk/lib/* /usr/lib/
```

### 方案3: 使用环境变量

设置RKNN SDK路径：

```bash
export RKNN_SDK_PATH=/path/to/your/rknn_sdk
export CMAKE_PREFIX_PATH=$RKNN_SDK_PATH:$CMAKE_PREFIX_PATH
```

## 📋 完整安装步骤

### 1. 检查系统要求
```bash
# 确认平台
uname -m  # 应该输出 aarch64

# 确认系统版本
cat /etc/os-release
```

### 2. 安装基础依赖
```bash
sudo apt update
sudo apt install -y build-essential cmake git
sudo apt install -y libopencv-dev libeigen3-dev
```

### 3. 设置NPU权限
```bash
sudo chmod 666 /dev/dri/renderD*
sudo usermod -a -G video $USER
# 重新登录或运行: newgrp video
```

### 4. 获取RKNN SDK

#### 选项A: 从官方下载
```bash
# 下载瑞芯微RKNN SDK
wget https://github.com/rockchip-linux/rknpu2/releases/download/v1.4.0/rknpu2_sdk_v1.4.0.tar.bz2
tar -xjf rknpu2_sdk_v1.4.0.tar.bz2
```

#### 选项B: 使用项目提供的SDK（如果有）
```bash
# 如果压缩包中包含了RKNN文件，直接解压即可
tar -xzf detector_lib.tar.gz
cd detector_lib
```

### 5. 配置路径

根据你的RKNN SDK位置，选择以下方法之一：

#### 方法1: 标准项目结构（推荐）
```bash
cd ~/Qworkspace
# 创建标准目录结构
mkdir -p 3rdparty/rknpu2/include
mkdir -p 3rdparty/rknpu2/Linux/aarch64

# 复制RKNN文件到正确位置
cp /your/rknn/sdk/path/include/* 3rdparty/rknpu2/include/
cp /your/rknn/sdk/path/lib/* 3rdparty/rknpu2/Linux/aarch64/

# 验证文件存在
ls 3rdparty/rknpu2/include/rknn_api.h
ls 3rdparty/rknpu2/Linux/aarch64/librknnrt.so
```

#### 方法2: 系统路径安装
```bash
sudo mkdir -p /usr/include/rknn
sudo cp /your/rknn/sdk/path/include/* /usr/include/rknn/
sudo cp /your/rknn/sdk/path/lib/* /usr/lib/
sudo ldconfig
```

### 6. 编译库
```bash
cd detector_lib
./build_and_install.sh
```

### 7. 验证安装
```bash
cd build/examples
./pose_image  # 测试姿态检测
./rim_basketball_image  # 测试篮筐检测
```

## 🔍 故障排除

### ❌ 问题1: RKNN版本不兼容 (最常见)

```bash
错误信息:
E RKNN: Invalid RKNN model version 6
E RKNN: rknn_init, load model failed!
```

**原因:** 系统RKNN库版本过旧，不支持Version 6模型格式

**解决方法:**
```bash
# ✅ 方法1: 重新编译 (推荐)
cd detector_lib && rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)
# CMake会自动找到并使用项目内的新版RKNN库

# ✅ 方法2: 检查库链接
ldd ./your_program | grep rknn
readelf -d ./your_program | grep -E "RPATH|RUNPATH"

# ⚠️ 方法3: 手动替换系统库 (谨慎使用)
sudo cp detector_lib/lib/librknnrt.so /lib/
sudo ldconfig
```

### 问题2: rknn_api.h未找到
```
错误: RKNN header not found
```

**解决**:
1. 检查文件是否存在：`find /home -name "rknn_api.h" 2>/dev/null`
2. 将找到的文件复制到正确位置
3. 或修改CMakeLists.txt中的POSSIBLE_RKNN_PATHS

### 问题3: librknnrt.so未找到 (编译时)
```
错误: cannot find -lrknnrt
```

**注意:** v1.0.3已内置RKNN库，此错误应该不再出现。如果出现：

**解决**:
```bash
# 确保库文件存在
ls -la detector_lib/lib/librknnrt.so

# 重新生成构建文件
rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)
```

### 问题3: NPU设备权限错误
```
错误: RKNN init failed
```

**解决**:
```bash
# 检查设备文件
ls -la /dev/dri/renderD*

# 设置权限
sudo chmod 666 /dev/dri/renderD*

# 添加用户到video组
sudo usermod -a -G video $USER
newgrp video
```

### 问题4: OpenCV版本不兼容
```
错误: OpenCV version mismatch
```

**解决**:
```bash
# 检查OpenCV版本
pkg-config --modversion opencv4

# 如果版本<4.0，升级OpenCV
sudo apt remove libopencv-dev
sudo apt install libopencv-dev=4.6.0*
```

## 💡 针对不同平台的说明

### Rock-5C / Radxa平台
```bash
# Radxa通常有预装的RKNN环境
# 检查是否已有RKNN库
dpkg -l | grep rknn
ls /usr/lib/*rknn*

# 如果已安装，可能只需要头文件
sudo apt install rockchip-npu-dev  # 如果有这个包的话
```

### Orange Pi 5系列
```bash
# Orange Pi通常需要手动安装RKNN SDK
# 按照上面的标准流程安装
```

### 通用RK3588平台
```bash
# 大多数RK3588系统都支持RKNN
# 关键是找到正确的SDK版本和路径
```

## 📞 获得帮助

如果仍然遇到问题：

1. **收集信息**:
```bash
uname -a
cat /proc/cpuinfo | grep -i rockchip
ls -la /dev/dri/
dpkg -l | grep -i rockchip
```

2. **查看详细错误**:
```bash
cd detector_lib
rm -rf build/
./build_and_install.sh > build.log 2>&1
cat build.log
```

3. **联系支持**:
   - 提供完整的build.log
   - 说明你的硬件平台型号
   - 说明你的系统版本

---

**记住**: DetectorLib是为RK3588平台优化的，确保你的硬件支持RKNN NPU！