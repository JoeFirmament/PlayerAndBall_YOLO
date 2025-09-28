# 🎯 Ground Calibration工作流程指南

## **📋 问题背景**

当点击"Start Ground Calibration"按钮后，如果系统检测到没有相机标定结果，会弹出以下提示：

```
Camera calibration results not found. Would you like to proceed with ground calibration anyway?
```

**这是正确的流程设计**，因为Ground Calibration需要依赖Camera Calibration的结果来进行精确的地面坐标转换。

## **🔄 完整的工作流程**

### **第一步：执行Camera Calibration**

#### **1. 选择相机标定模式**
```
📷 Camera Calibration Tab → 选择标定模式
```

#### **2. 连接相机并拍摄标定图片**
```
🔗 连接相机 → 📸 拍摄标定图像 (至少10-15张不同角度的棋盘格图片)
```

#### **3. 运行相机标定**
```
▶️ Start Camera Calibration → 等待标定完成
```

#### **4. 保存标定结果**
```python
# 选择保存格式（推荐JSON格式）
文件类型：
• JSON文件 (*.json) - ✅ 推荐，易于查看和版本控制
• XML文件 (*.xml) - ✅ OpenCV兼容，适合C++程序
• NPZ文件 (*.npz) - ⚠️ Python专用，不易查看

保存的文件包含：
• camera_matrix: 相机内参矩阵 (3x3)
• dist_coeffs: 畸变系数 (1x5或1x8)
• rvecs: 旋转向量
• tvecs: 平移向量
• 标定精度信息
```

### **第二步：加载Camera Calibration结果**

#### **方法1：使用GUI加载（推荐）**
```
📂 File → Load Calibration File
```

#### **方法2：在Ground Calibration开始前加载**
```python
# Ground Calibration Tab中：
🔧 工具栏 → Load Camera Calibration
# 或者在弹出警告时选择"No"，然后加载文件
```

#### **支持的文件格式**
| 格式 | 扩展名 | 优势 | 适用场景 |
|------|--------|------|----------|
| **JSON** | `.json` | 人类可读，版本控制友好 | ✅ **推荐用于开发和调试** |
| **XML** | `.xml` | OpenCV标准格式 | ✅ **推荐用于生产环境** |
| **NPZ** | `.npz` | NumPy原生，数据完整 | ⚠️ **仅Python环境** |

### **第三步：执行Ground Calibration**

#### **1. 准备地面标定图片**
```
📁 选择包含地面棋盘格图片的文件夹
图片要求：
• 棋盘格平放在地面上
• 相机从不同角度拍摄
• 建议至少拍摄4-6张图片
```

#### **2. 设置地面标定参数**
```
棋盘格尺寸：
• 宽度: 通常6-9格
• 高度: 通常4-6格
• 方格大小: 实际尺寸(毫米或厘米)
```

#### **3. 开始Ground Calibration**
```
▶️ Start Ground Calibration
系统会自动：
• 检查相机标定结果 ✅
• 检测地面棋盘格角点
• 计算地面单应性矩阵
• 建立像素↔地面坐标映射
```

#### **4. 验证Ground Calibration**
```
✅ Validate Ground Calibration
验证内容：
• 角点检测准确性
• 坐标转换精度
• 重投影误差
```

#### **5. 保存Ground Calibration结果**
```
💾 Save Ground Calibration Results
保存的文件包含：
• ground_homography: 地面单应性矩阵
• camera_height_info: 相机高度信息
• 标定精度统计
```

## **📁 文件管理建议**

### **推荐的文件命名规范**
```
📂 calibration_results/
├── 📄 camera_calibration_20241201_143022.json    # 相机标定结果
├── 📄 ground_calibration_20241201_143522.json    # 地面标定结果
└── 📄 full_calibration_20241201_144022.json      # 完整标定数据
```

### **版本控制建议**
```bash
# 使用JSON格式便于版本控制
git add *.json
git commit -m "Add camera calibration results for session 2024-12-01"
```

## **🔍 故障排除**

### **问题1：找不到相机标定结果**
```
❌ Camera calibration results not found
✅ 解决方案：
1. 确保已完成Camera Calibration
2. 使用"Load Calibration File"加载保存的标定文件
3. 检查文件路径和格式
```

### **问题2：Ground Calibration精度不佳**
```
⚠️ 可能原因：
• 地面棋盘格放置不平整
• 相机标定结果不准确
• 拍摄角度不够多样
✅ 解决方案：
1. 重新进行Camera Calibration
2. 确保地面棋盘格完全平放
3. 增加不同角度的拍摄图片
```

### **问题3：文件加载失败**
```
❌ 加载失败
✅ 检查清单：
• 文件格式是否正确 (JSON/XML/NPZ)
• 文件是否损坏
• NumPy数组格式是否正确
• 相机参数是否完整
```

## **⚙️ 技术细节**

### **为什么需要Camera Calibration结果？**

Ground Calibration需要以下Camera Calibration数据：

1. **相机内参矩阵 (camera_matrix)**
   ```python
   # 3x3矩阵，包含焦距和主点信息
   camera_matrix = [
       [fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]
   ]
   ```

2. **畸变系数 (dist_coeffs)**
   ```python
   # 畸变校正参数
   dist_coeffs = [k1, k2, p1, p2, k3]  # 或更多参数
   ```

3. **坐标转换计算**
   ```python
   # Ground Calibration使用相机参数进行：
   # 像素坐标 → 相机坐标系 → 地面坐标系
   # 需要准确的相机内参来进行精确转换
   ```

### **Ground Calibration vs Camera Calibration**

| 特性 | Camera Calibration | Ground Calibration |
|------|-------------------|-------------------|
| **目的** | 确定相机内参和畸变 | 建立地面坐标系 |
| **输入** | 棋盘格在空间中的位置 | 棋盘格在地面上的位置 |
| **输出** | camera_matrix, dist_coeffs | ground_homography |
| **依赖关系** | 独立 | 需要Camera Calibration结果 |

## **🚀 快速开始脚本**

创建以下脚本快速测试完整流程：

```python
#!/usr/bin/env python3
"""
Ground Calibration完整流程演示
"""

import os
import cv2
import numpy as np
from pathlib import Path

def demo_calibration_workflow():
    """演示完整的标定工作流程"""

    print("🎯 Ground Calibration完整工作流程演示")
    print("=" * 50)

    # 1. 检查Camera Calibration结果
    camera_file = "camera_calibration_results.json"
    if os.path.exists(camera_file):
        print(f"✅ 找到相机标定文件: {camera_file}")
        # 加载相机标定数据
        with open(camera_file, 'r') as f:
            import json
            camera_data = json.load(f)

        print("📊 相机标定数据:")
        print(f"• 相机矩阵形状: {np.array(camera_data['camera_matrix']).shape}")
        print(f"• 畸变系数数量: {len(camera_data['dist_coeffs'])}")

    else:
        print(f"❌ 未找到相机标定文件: {camera_file}")
        print("请先完成Camera Calibration步骤")

    # 2. 准备Ground Calibration
    ground_images_dir = "ground_calibration_images/"
    if os.path.exists(ground_images_dir):
        images = list(Path(ground_images_dir).glob("*.jpg"))
        print(f"✅ 找到地面标定图片: {len(images)} 张")

        # 3. 执行Ground Calibration (简化演示)
        print("🔄 执行Ground Calibration...")
        print("• 检测棋盘格角点")
        print("• 计算地面单应性矩阵")
        print("• 建立坐标映射关系")

        # 模拟结果
        ground_homography = np.eye(3)  # 简化示例
        print("✅ Ground Calibration完成")
        print(f"• 生成地面单应性矩阵: {ground_homography.shape}")

    else:
        print(f"❌ 未找到地面标定图片目录: {ground_images_dir}")
        print("请拍摄地面棋盘格图片")

    print("\n🎉 工作流程演示完成!")

if __name__ == "__main__":
    demo_calibration_workflow()
```

## **📞 技术支持**

如果遇到问题，请检查：
1. **相机标定文件是否存在且格式正确**
2. **Ground Calibration图片是否包含清晰的棋盘格**
3. **所有参数设置是否合理**
4. **文件路径是否正确**

---

**🎯 总结**: Ground Calibration需要Camera Calibration结果是正确的设计决策，这样可以确保坐标转换的准确性和一致性。建议按照上述完整流程进行操作。
