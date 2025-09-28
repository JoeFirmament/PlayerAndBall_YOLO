# 相机标定多格式保存系统

## 概述

相机标定工具现在支持保存标定结果为多种通用格式，方便查看和跨平台使用。

## 支持的格式

### 1. JSON 格式 (推荐) ⭐
- **特点**: 人类可读，跨平台兼容
- **优势**: 可以直接用文本编辑器查看，易于版本控制
- **文件大小**: 中等
- **使用场景**: 开发调试、数据分享、版本控制

```json
{
  "camera_matrix": [
    [800.0, 0.0, 320.0],
    [0.0, 800.0, 240.0],
    [0.0, 0.0, 1.0]
  ],
  "dist_coeffs": [0.1, -0.2, 0.01, 0.001, 0.0],
  "calibration_date": "2024-01-01T12:00:00",
  "image_size": [640, 480]
}
```

### 2. XML 格式 (OpenCV兼容)
- **特点**: OpenCV标准格式，C++程序最佳兼容
- **优势**: 与OpenCV C++ API完全兼容
- **文件大小**: 中等
- **使用场景**: C++应用程序开发

```xml
<opencv_storage>
  <camera_matrix type_id="opencv-matrix">
    <rows>3</rows><cols>3</cols><dt>f</dt>
    <data>800.0 0.0 320.0 0.0 800.0 240.0 0.0 0.0 1.0</data>
  </camera_matrix>
</opencv_storage>
```

### 3. NPZ 格式 (Python专用)
- **特点**: NumPy原生格式，包含所有数据类型
- **优势**: 性能最佳，数据完整性最好
- **文件大小**: 最小
- **使用场景**: Python应用程序内部使用

## 默认设置

- **主要格式**: JSON (默认选中)
- **辅助格式**: XML (默认选中)
- **可选格式**: NPZ (可选择)

当用户不选择任何格式时，系统会默认保存为JSON格式。

## 使用方法

### GUI界面使用

1. 完成相机标定后，点击"保存结果"按钮
2. 系统会询问是否保存到当前目录：
   - **默认目录**: `/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/`
   - 点击"是"保存到当前目录（推荐）
   - 点击"否"选择自定义目录
3. 在设置面板中选择要保存的格式：
   - ☑ JSON (推荐) - 人类可读格式
   - ☑ XML (OpenCV) - C++兼容格式
   - ☐ NPZ (Python) - 高性能格式
4. 系统会自动保存为选定的格式

### 默认保存路径

- **项目根目录**: `/home/orangepi/Qworkspace/yolov8_pose_basketball/`
- **默认保存目录**: `/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/`
- **文件名格式**: `YYYYMMDD_HHMMSS_calibration.{ext}`

### 文件名示例
```
20250829_151327_calibration.json  # JSON格式 (推荐)
20250829_151327_calibration.xml   # XML格式 (OpenCV兼容)
20250829_151327_calibration.npz   # NPZ格式 (Python专用)
```

### 命令行使用

```bash
# 使用文件管理器
python3 calibration_file_manager.py save --data calibration_data.npz --output ./results

# 转换格式
python3 calibration_file_manager.py convert --input calibration.json --format xml

# 加载文件
python3 calibration_file_manager.py load --input calibration.json
```

### Python API使用

```python
from calibration_file_manager import CalibrationFileManager

# 创建管理器
manager = CalibrationFileManager()

# 保存多种格式
calibration_data = {
    'camera_matrix': camera_matrix,
    'dist_coeffs': dist_coeffs,
    # ... 其他标定数据
}

saved_files = manager.save_calibration_multi_format(
    calibration_data,
    output_dir="./calibration_results",
    formats=['json', 'xml']
)

# 加载文件
data, format_type = manager.load_calibration_file('calibration.json')
```

## 文件格式对比

| 特性 | JSON | XML | NPZ |
|------|------|-----|-----|
| 人类可读 | ⭐⭐⭐ | ⭐⭐ | ❌ |
| 跨平台兼容 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| 文件大小 | 中等 | 中等 | 小 |
| 性能 | 快 | 中等 | 最快 |
| 版本控制友好 | ⭐⭐⭐ | ⭐⭐ | ❌ |
| C++兼容 | ❌ | ⭐⭐⭐ | ❌ |

## 最佳实践

### 开发阶段
- 使用JSON格式，便于调试和版本控制
- 配合XML格式，确保C++兼容性

### 生产环境
- 根据具体需求选择合适的格式
- C++应用程序推荐XML格式
- Python应用程序可以选择NPZ格式获得最佳性能

### 数据分享
- 优先使用JSON格式，通用性最好
- 如果需要与OpenCV C++程序交互，使用XML格式

## 故障排除

### Q: JSON文件无法查看？
A: 确保使用支持UTF-8编码的文本编辑器，如VS Code、Notepad++等。

### Q: XML文件在C++中加载失败？
A: 检查OpenCV版本是否支持XML格式加载，或使用FileStorage API。

### Q: 想要添加其他格式支持？
A: 可以扩展CalibrationFileManager类，参考现有的JSON/XML实现。

## 版本历史

- v1.0: 初始版本，支持NPZ、JSON、XML格式
- v1.1: 将JSON设为默认推荐格式
- v1.2: 移除YAML依赖，专注于核心格式
- v1.3: 改进用户界面和格式说明
- v1.4: 将默认保存路径改为当前工具目录 (tools/)
