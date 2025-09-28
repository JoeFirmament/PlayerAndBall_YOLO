# 相机标定高度详解

## 🎯 **相对高度 vs 绝对高度**

### **相对高度 (标定结果)**
标定过程中得到的高度是**相对高度**，以标定板为参考点：

```
相机实际高度 = 标定板高度 + 标定结果中的Z值
```

### **坐标系说明**

#### **标定板坐标系 (世界坐标系)**
- **原点**: 标定板左上角第一个内角点
- **X轴**: 沿标定板宽度方向
- **Y轴**: 沿标定板高度方向
- **Z轴**: 垂直于标定板平面向上

#### **相机位置表示**
```
相机位置 = [X, Y, Z] 相对于标定板坐标系

X: 相机在左右方向的偏移
Y: 相机在前后方向的偏移
Z: 相机在垂直方向的高度 (相对标定板)
```

## 📊 **实际应用示例**

### **场景1: 篮球场地面标定**
```
假设条件:
- 标定板放在篮球场地面上
- 标定板厚度忽略不计 (≈0mm)

标定结果: Z = 892.22 mm
实际含义: 相机距离地面约 892.22 mm (89.2 cm)
```

### **场景2: 标定板放在桌子上**
```
假设条件:
- 桌子高度: 750 mm
- 标定板放在桌子上

标定结果: Z = 892.22 mm
实际含义: 相机距离地面 = 750 + 892.22 = 1642.22 mm (164.2 cm)
```

## 🔧 **转换为绝对高度的方法**

### **方法1: 测量标定板位置**
```python
# 已知信息
标定板距地面高度 = 0  # 如果放在地面
# 或
标定板距地面高度 = 750  # 如果放在750mm高的桌子上

# 计算实际相机高度
标定结果_Z = 892.22  # 从标定文件中读取
实际相机高度 = 标定板距地面高度 + 标定结果_Z

print(f"实际相机高度: {实际相机高度} mm")
```

### **方法2: 多点标定**
如果在多个已知高度位置进行标定，可以建立更精确的映射关系。

## 📋 **标定数据的实际含义**

### **当前标定数据分析**
```
标定板参数:
- 方格尺寸: 25.0 mm
- 棋盘格: 9x6 (8x5个内角点)

相机位置范围:
- X范围: 168.84 到 375.13 mm (左右移动范围约20.6cm)
- Y范围: -338.09 到 110.58 mm (前后移动范围约44.9cm)
- Z范围: 611.02 到 892.22 mm (高度变化约28.1cm)
```

### **实际应用建议**

1. **篮球场应用**: 如果标定板放在地面，Z值≈相机距地面高度
2. **实验室应用**: 需要测量标定板的具体位置
3. **多相机系统**: 可用于计算相机间的相对位置关系

## 💡 **重要提醒**

- **标定结果都是相对测量值**
- **需要结合实际场景确定基准点**
- **同一相机在不同位置标定的Z值是可比的**
- **不同相机的Z值需要考虑各自的基准点**

## 🔍 **如何获取绝对高度**

```python
def get_absolute_camera_height(calibration_file, board_height_from_ground=0):
    """
    计算相机绝对高度

    参数:
    calibration_file: 标定文件路径
    board_height_from_ground: 标定板距离地面的高度 (mm)
    """
    import json

    with open(calibration_file, 'r') as f:
        data = json.load(f)

    tvecs = data['tvecs']

    absolute_heights = []
    for i, tvec in enumerate(tvecs):
        relative_height = tvec[2][0] if isinstance(tvec[2], list) else tvec[2]
        absolute_height = board_height_from_ground + relative_height
        absolute_heights.append(absolute_height)
        print(f"位置 {i+1}: 相对高度 {relative_height:.2f}mm, 绝对高度 {absolute_height:.2f}mm")

    return absolute_heights

# 使用示例
# heights = get_absolute_camera_height('calibration.json', board_height_from_ground=0)
```
