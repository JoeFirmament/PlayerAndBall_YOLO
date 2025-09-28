# 相机标定内参和外参矫正指南

## 📋 概述

本指南介绍如何对相机标定的内参（Intrinsic Parameters）和外参（Extrinsic Parameters）进行矫正。内参主要涉及图像去畸变，外参涉及3D坐标变换。

## 🎯 内参矫正（Intrinsic Correction）

### 1. 内参矫正原理

内参矫正主要是**图像去畸变**（Undistortion），用于矫正镜头畸变：

- **径向畸变**：桶形畸变（Barrel Distortion）或枕形畸变（Pincushion Distortion）
- **切向畸变**：由镜头与图像平面不平行引起的畸变

### 2. 内参矫正方法

#### 2.1 图像去畸变

```python
from simple_corrector import SimpleCorrector

# 创建矫正器
corrector = SimpleCorrector()
corrector.load_calibration("example_calibration.npz")

# 读取图像
image = cv2.imread("input.jpg")

# 去畸变（alpha参数控制裁剪策略）
undistorted = corrector.undistort_image(image, alpha=0.5)

# 保存结果
cv2.imwrite("undistorted.jpg", undistorted)
```

**alpha参数说明：**
- `alpha=0.0`: 保持所有像素，可能有黑边
- `alpha=0.5`: 平衡选择（推荐）
- `alpha=1.0`: 裁剪到有效区域

#### 2.2 点坐标矫正

```python
# 矫正图像点坐标
original_points = [[100, 100], [200, 150], [300, 200]]
corrected_points = corrector.undistort_points(original_points)

print("矫正前后对比:")
for orig, corr in zip(original_points, corrected_points):
    print(f"{orig} -> {corr}")
```

### 3. 质量评估

```python
# 评估去畸变质量
quality = corrector.evaluate_undistortion_quality(original_image, undistorted)

print(f"平均像素差异: {quality['mean_difference']:.2f}")
print(f"结构相似度(SSIM): {quality['ssim_score']:.4f}")
print(f"质量等级: {quality['quality_assessment']}")
```

**质量等级标准：**
- `EXCELLENT`: SSIM > 0.95
- `GOOD`: SSIM > 0.85
- `FAIR`: SSIM > 0.7
- `POOR`: SSIM ≤ 0.7

### 4. 批量处理

```python
# 批量去畸变
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = corrector.batch_undistort(image_paths, "output_dir", alpha=0.5)
```

## 🔄 外参矫正（Extrinsic Correction）

### 1. 外参矫正原理

外参矫正涉及**3D坐标变换**，将图像坐标转换为世界坐标：

- **旋转矩阵**：描述相机坐标系相对于世界坐标系的旋转
- **平移向量**：描述相机中心相对于世界坐标系原点的位移

### 2. 外参矫正方法

#### 2.1 图像坐标到世界坐标转换

```python
# 加载外参数据
from calibration_corrector import CalibrationCorrector

corrector = CalibrationCorrector()
corrector.load_calibration("calibration_with_extrinsics.npz")

# 图像坐标点
image_points = np.array([[100, 100], [200, 150], [300, 200]])

# 转换为世界坐标（Z=0表示地面）
world_points = corrector.image_to_world_coordinates(
    image_points,
    corrector.rvecs[0],  # 旋转向量
    corrector.tvecs[0],  # 平移向量
    z=0.0               # 世界坐标Z值
)

print("坐标转换结果:")
for img_pt, world_pt in zip(image_points, world_points):
    print(f"图像: {img_pt} -> 世界: {world_pt}")
```

#### 2.2 3D点变换

```python
# 直接变换3D世界坐标点到相机坐标系
world_points_3d = np.array([
    [100, 200, 0],    # 地面点1
    [150, 250, 10],   # 空中点1
    [200, 300, 20]    # 空中点2
])

# 变换到相机坐标系
camera_points = corrector.transform_3d_points(
    world_points_3d,
    corrector.rvecs[0],
    corrector.tvecs[0]
)
```

## 📊 矫正效果验证

### 1. 内参矫正验证

```python
# 1. 检查焦距合理性
fx, fy = corrector.camera_matrix[0,0], corrector.camera_matrix[1,1]
print(f"焦距: fx={fx:.1f}, fy={fy:.1f}")

# 2. 检查主点位置
cx, cy = corrector.camera_matrix[0,2], corrector.camera_matrix[1,2]
print(f"主点: cx={cx:.1f}, cy={cy:.1f}")

# 3. 检查畸变系数
print(f"畸变系数: {corrector.dist_coeffs.flatten()}")

# 4. 验证矫正前后差异
original = cv2.imread("original.jpg")
corrected = corrector.undistort_image(original)
quality = corrector.evaluate_undistortion_quality(original, corrected)
```

### 2. 外参矫正验证

```python
# 1. 验证旋转矩阵正交性
R, _ = cv2.Rodrigues(corrector.rvecs[0])
identity_check = np.dot(R.T, R)
print(f"旋转矩阵正交性检查: {np.allclose(identity_check, np.eye(3))}")

# 2. 验证平移向量合理性
tvec = corrector.tvecs[0].flatten()
print(f"平移向量: {tvec}")

# 3. 重新投影验证
# 将世界坐标点重新投影到图像坐标，验证精度
```

## 🛠️ 工具使用指南

### 1. 简单内参矫正器

```bash
# 激活环境
source ~/calibration_env/bin/activate

# 运行内参矫正演示
python simple_corrector.py
```

**功能特点：**
- ✅ 图像去畸变
- ✅ 点坐标矫正
- ✅ 质量评估
- ✅ 批量处理
- ✅ 无需外参数据

### 2. 完整矫正器

```bash
# 运行完整矫正演示
python calibration_corrector.py
```

**功能特点：**
- ✅ 所有内参矫正功能
- ✅ 外参矫正（需要有效的外参数据）
- ✅ 3D坐标变换
- ✅ 高级质量评估

## 🎯 最佳实践

### 1. 内参矫正最佳实践

1. **选择合适的alpha值**
   - 一般应用：`alpha=0.5`
   - 高精度测量：`alpha=0.0`（保留所有像素）
   - 视觉效果：`alpha=1.0`（裁剪无效区域）

2. **质量监控**
   - 关注SSIM分数（>0.9为优秀）
   - 监控平均像素差异（<30为良好）
   - 定期验证矫正效果

3. **批量处理优化**
   - 使用相同的alpha值保持一致性
   - 监控处理成功率
   - 记录处理统计信息

### 2. 外参矫正最佳实践

1. **数据验证**
   - 验证旋转向量格式（应为3元素）
   - 验证平移向量合理性
   - 检查坐标系一致性

2. **精度验证**
   - 使用已知世界坐标进行验证
   - 比较多视角一致性
   - 监控重投影误差

## 🚨 常见问题

### 1. 内参矫正问题

**Q: 去畸变后图像有黑边怎么办？**
A: 增加alpha值（如改为0.8或1.0）来裁剪黑边区域。

**Q: SSIM分数很低怎么办？**
A: 检查标定质量，可能需要重新标定或调整畸变模型。

**Q: 批量处理失败率高怎么办？**
A: 检查图像格式一致性，确保所有图像分辨率相同。

### 2. 外参矫正问题

**Q: Rodrigues转换失败怎么办？**
A: 检查旋转向量格式，应为3x1或1x3数组。

**Q: 坐标转换结果不准确怎么办？**
A: 验证标定精度，可能需要重新标定外参。

**Q: 内存不足怎么办？**
A: 分批处理大量数据，避免一次性加载过多图像。

## 📈 性能优化

### 1. 内存优化

```python
# 分批处理大量图像
batch_size = 10
for i in range(0, len(image_paths), batch_size):
    batch = image_paths[i:i+batch_size]
    corrector.batch_undistort(batch, output_dir)
```

### 2. 速度优化

```python
# 预先计算矫正映射（对于固定分辨率）
map1, map2 = cv2.initUndistortRectifyMap(
    camera_matrix, dist_coeffs, None, camera_matrix,
    image.shape[:2][::-1], cv2.CV_32FC1
)

# 快速矫正
undistorted = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
```

## 🔗 相关工具

- `simple_corrector.py` - 简单内参矫正工具
- `calibration_corrector.py` - 完整矫正工具
- `npz_converter.py` - 格式转换工具
- `calibration_inspector.py` - 标定参数检验工具

## 📚 参考资料

1. **OpenCV文档**: Camera Calibration and 3D Reconstruction
2. **Zhang's Method**: Flexible Camera Calibration By Viewing a Plane From Unknown Orientations
3. **Brown's Model**: Close-range camera calibration

---

**版本**: v1.0
**更新日期**: 2024-08-29
**作者**: AI Assistant
