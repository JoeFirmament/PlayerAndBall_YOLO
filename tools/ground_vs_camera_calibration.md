# 🎯 Ground Calibration vs Camera Calibration

## **核心区别**

### **📹 Camera Calibration (相机标定)**
- **目的**: 确定相机内部参数和畸变校正
- **方法**: 使用棋盘格在不同位置、角度拍摄
- **结果**: 相机内参矩阵、畸变系数、相对位置信息
- **高度信息**: **相对高度** (相对于标定板坐标系)

### **🌍 Ground Calibration (地面标定)**
- **目的**: 建立精确的世界坐标系基准
- **方法**: 棋盘格平放在地面上，计算相机到地面的关系
- **结果**: 地面Homography矩阵、绝对高度基准
- **高度信息**: **绝对高度** (相对于实际地面)

## **Ground Calibration 能提供什么？**

### **1. 绝对高度基准**
```
Ground Calibration 结果:
├── ground_homography_matrix: 3x3变换矩阵
├── reprojection_error: 重投影误差
└── 隐含的Z=0基准面 (地面)
```

### **2. 坐标系转换**
```
图像像素坐标 → 地面实际坐标 (毫米/厘米)

例如:
- 图像中点的像素坐标: (320, 240)
- 转换为地面坐标: (1500mm, 2000mm)
- 实际含义: 距离篮筐中心点150cm, 200cm位置
```

### **3. 精确的Z轴参考**
```
Ground Calibration的优势:
✅ 建立Z=0的精确基准 (实际地面)
✅ 所有测量都相对于真实地面
✅ 适合篮球场等需要精确地面坐标的应用
```

## **实际应用场景**

### **🏀 篮球场应用**
```python
# Ground Calibration 结果用于:
# 1. 计算运动员相对于篮筐的位置
# 2. 测量投篮角度和距离
# 3. 分析运动员移动轨迹
# 4. 确定罚球线、3分线位置
```

### **📏 精确测量应用**
```python
# Ground Calibration 提供:
# - 像素到实际距离的精确转换
# - 角度测量 (相对于地面法线)
# - 速度计算 (考虑实际距离)
```

## **Ground Calibration 的工作原理**

### **1. 棋盘格放置**
```
地面放置棋盘格:
- 棋盘格完全平放在地面上
- Z坐标 = 0 (地面基准)
- 建立世界坐标系的原点
```

### **2. 计算Homography矩阵**
```python
# 从棋盘格角点计算变换矩阵
# 像素坐标 ↔ 地面实际坐标 (mm)

H = ground_homography_matrix  # 3x3变换矩阵

# 像素点转换为地面坐标
ground_point = H * pixel_point
```

### **3. 高度基准建立**
```
Ground Calibration的优势:
- Z=0 对应实际地面
- 所有高度测量都有绝对参考
- 无需额外的基准点测量
```

## **与Camera Calibration的对比**

| 特性 | Camera Calibration | Ground Calibration |
|------|-------------------|-------------------|
| **坐标系** | 标定板坐标系 | 实际地面坐标系 |
| **高度基准** | 相对标定板高度 | 绝对地面高度 |
| **适用场景** | 相机参数校正 | 实际距离测量 |
| **输出结果** | 内参矩阵 + 相对位置 | Homography矩阵 |
| **测量精度** | 相机参数精度 | 地面坐标精度 |

## **推荐使用流程**

### **1. 先进行Camera Calibration**
```python
# 获取相机内参和畸变校正
camera_matrix, dist_coeffs = calibrate_camera(...)
```

### **2. 再进行Ground Calibration**
```python
# 建立地面坐标系基准
ground_homography = calibrate_ground(...)

# 现在可以进行精确的地面测量了!
```

### **3. 实际应用**
```python
# 像素坐标 → 地面实际坐标
pixel_point = [320, 240]  # 图像中心点
ground_point = apply_homography(pixel_point, ground_homography)

print(f"实际地面位置: {ground_point} mm")
print(f"距离原点: {np.linalg.norm(ground_point)} mm")
```

## **总结**

**Ground Calibration 确实能提供相机Z轴的绝对高度信息！**

- ✅ **建立Z=0的精确基准** (实际地面)
- ✅ **提供像素到实际距离的精确转换**
- ✅ **适合需要绝对坐标的应用场景**
- ✅ **与Camera Calibration配合使用效果最佳**

Ground Calibration 是建立精确世界坐标系的关键步骤，让你能够从图像像素坐标转换为实际的地面坐标，包括精确的高度信息。
