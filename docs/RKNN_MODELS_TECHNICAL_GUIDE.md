# RKNN模型技术详解：YOLOv8 Pose与Rim Basketball检测

本文档详细解读两个RKNN模型的使用细节，特别是预处理、后处理的关键技术点。经过大量调试和验证，这里记录了所有重要的技术细节。

## 目录
1. [模型概述](#模型概述)
2. [预处理详解](#预处理详解)
3. [后处理详解（重点）](#后处理详解重点)
4. [关键技术要点](#关键技术要点)
5. [常见问题和解决方案](#常见问题和解决方案)

---

## 模型概述

### 1. YOLOv8 Pose模型
- **文件**: `Q_yolov8_pose.rknn`
- **输入**: `[1, 640, 640, 3]` NHWC格式
- **输出**: 3个检测层，每层包含位置(64维DFL)和类别(1类:person)
- **功能**: 人体姿态检测，17个关键点

### 2. Rim Basketball模型
- **文件**: `Q_Rim_Basketball_724_JZ.rknn`
- **输入**: `[1, 640, 640, 3]` NHWC格式
- **输出**: 6个独立张量（已经过DFL处理）
- **功能**: 篮筐(rim)和篮球(basketball)检测

---

## 预处理详解

### Letterbox处理

两个模型都使用相同的Letterbox预处理方式：

```cpp
// letterbox参数计算
float scale = std::min(640.0f/src_width, 640.0f/src_height);
int new_width = (int)(src_width * scale);
int new_height = (int)(src_height * scale);
int x_pad = (640 - new_width) / 2;
int y_pad = (640 - new_height) / 2;
```

### 颜色格式要求

⚠️ **关键点**：
1. **输入格式**: RGB顺序（不是BGR！）
2. **数据类型**: uint8_t
3. **填充颜色**: (114, 114, 114) 灰色背景

### NHWC格式说明

RKNN模型要求NHWC格式输入：
- **N**: Batch size (1)
- **H**: Height (640)
- **W**: Width (640)
- **C**: Channels (3, RGB顺序)

### 零拷贝优化

```cpp
// 1. 分配NPU内存
rknn_mem_create(&input_mem, rknn_ctx, 640*640*3, RKNN_NPU_MEM);

// 2. 获取虚拟地址
input_ptr = (uint8_t*)rknn_mem_get_virt_addr(&input_mem, 640*640*3);

// 3. 直接写入NPU内存（避免CPU↔NPU拷贝）
// 注意：必须按NHWC格式写入
for (int y = 0; y < dst_height; y++) {
    for (int x = 0; x < dst_width; x++) {
        int dst_idx = (y * dst_width + x) * 3;
        input_ptr[dst_idx + 0] = r;  // R
        input_ptr[dst_idx + 1] = g;  // G
        input_ptr[dst_idx + 2] = b;  // B
    }
}
```

---

## 后处理详解（重点）

### 1. YOLOv8 Pose模型后处理

#### 输出张量结构
- **3个检测层**: P3(80x80), P4(40x40), P5(20x20)
- **每层格式**: `[1, 65, H, W]` 或 `[1, 68, H, W]`（带关键点）
  - 前64维: DFL编码的边界框（16×4）
  - 第65维: 类别置信度（person）
  - 后3×17维: 关键点坐标（可选）

#### DFL解码过程
```cpp
// 1. 提取DFL分布（16个值）
float dfl_values[16];
for (int i = 0; i < 16; i++) {
    dfl_values[i] = deqnt_affine_to_f32(input[offset + i], zp, scale);
}

// 2. Softmax归一化
softmax(dfl_values, 16);

// 3. 计算期望值
float distance = 0;
for (int i = 0; i < 16; i++) {
    distance += dfl_values[i] * i;
}

// 4. 转换为边界框坐标
float x1 = (grid_x + 0.5 - left_distance) * stride;
float y1 = (grid_y + 0.5 - top_distance) * stride;
float x2 = (grid_x + 0.5 + right_distance) * stride;
float y2 = (grid_y + 0.5 + bottom_distance) * stride;
```

#### 关键点解码
```cpp
// 17个关键点，每个3维(x,y,confidence)
for (int k = 0; k < 17; k++) {
    float kp_x = deqnt_affine_to_f32(kp_data[k*3+0], zp, scale);
    float kp_y = deqnt_affine_to_f32(kp_data[k*3+1], zp, scale);
    float kp_conf = sigmoid(deqnt_affine_to_f32(kp_data[k*3+2], zp, scale));
    
    // 转换到原图坐标
    kp_x = (kp_x * 2.0 + grid_x) * stride;
    kp_y = (kp_y * 2.0 + grid_y) * stride;
}
```

### 2. Rim Basketball模型后处理

#### ⚠️ 关键差异：6个独立输出张量

这是最重要的区别！该模型已经在导出时完成了DFL处理：

```
输出[0]: [1, 1, 4, 6400] - P3层回归输出（已处理）
输出[1]: [1, 2, 80, 80] - P3层分类输出
输出[2]: [1, 1, 4, 1600] - P4层回归输出（已处理）
输出[3]: [1, 2, 40, 40] - P4层分类输出
输出[4]: [1, 1, 4, 400]  - P5层回归输出（已处理）
输出[5]: [1, 2, 20, 20] - P5层分类输出
```

#### 类别定义（重要！）
```cpp
// 正确的类别顺序
static const char* class_names[2] = {"basketball", "rim"};
// basketball = 0
// rim = 1
```

#### 简化的后处理流程

由于DFL已经在模型中处理，后处理大大简化：

```cpp
// 1. 遍历每层
for (int layer = 0; layer < 3; layer++) {
    int8_t* reg_data = (int8_t*)outputs[layer*2].buf;     // 回归输出
    int8_t* cls_data = (int8_t*)outputs[layer*2+1].buf;   // 分类输出
    
    // 2. 遍历网格点
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            // 3. 获取分类置信度
            float basketball_conf = sigmoid(deqnt(cls_data[0], zp1, scale1));
            float rim_conf = sigmoid(deqnt(cls_data[1], zp1, scale1));
            
            // 4. 选择最高置信度类别
            if (max_conf > conf_threshold) {
                // 5. 直接读取边界框距离（已经过DFL处理）
                float left = deqnt(reg_data[0], zp0, scale0);
                float top = deqnt(reg_data[1], zp0, scale0);
                float right = deqnt(reg_data[2], zp0, scale0);
                float bottom = deqnt(reg_data[3], zp0, scale0);
                
                // 6. 计算实际坐标
                float x1 = (x + 0.5f - left) * stride;
                float y1 = (y + 0.5f - top) * stride;
                float x2 = (x + 0.5f + right) * stride;
                float y2 = (y + 0.5f + bottom) * stride;
            }
        }
    }
}
```

### NMS（非极大值抑制）

两个模型都使用相同的NMS策略：

```cpp
// 1. 按置信度排序
std::sort(detections.begin(), detections.end(), 
    [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });

// 2. NMS处理
for (int i = 0; i < detections.size(); i++) {
    if (suppressed[i]) continue;
    
    for (int j = i + 1; j < detections.size(); j++) {
        if (suppressed[j]) continue;
        
        float iou = calculate_iou(detections[i], detections[j]);
        if (iou > nms_threshold) {
            suppressed[j] = true;
        }
    }
}
```

推荐阈值：
- **置信度阈值**: 0.25
- **NMS阈值**: 0.1（篮球场景较低的NMS阈值效果更好）

---

## 关键技术要点

### 1. 量化与反量化

```cpp
// INT8反量化公式
float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

// 置信度阈值量化（用于加速）
int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    return (int8_t)fmaxf(-128, fminf(127, dst_val));
}
```

### 2. 坐标转换

从letterbox坐标转换回原图坐标：
```cpp
// 缩放因子（来自预处理）
float scale = 640.0f / std::max(src_width, src_height);
float x_offset = (640 - src_width * scale) / 2;
float y_offset = (640 - src_height * scale) / 2;

// 转换公式
float orig_x = (detect_x - x_offset) / scale;
float orig_y = (detect_y - y_offset) / scale;
```

### 3. 性能优化技巧

1. **提前计算量化阈值**：避免每个网格点都计算
2. **使用fast_exp替代exp**：sigmoid计算优化
3. **零拷贝内存**：直接操作NPU内存
4. **多线程处理**：不同层并行处理

---

## 常见问题和解决方案

### Q1: 为什么检测结果偏移？
**A**: 检查颜色格式，确保是RGB而不是BGR。

### Q2: 为什么篮球和篮筐类别反了？
**A**: 确认类别定义顺序：basketball=0, rim=1。

### Q3: 为什么Pose模型检测框不准？
**A**: 检查DFL解码是否正确，特别是softmax和期望值计算。

### Q4: 为什么Basketball模型后处理简单但之前很复杂？
**A**: 因为模型导出时已经完成DFL处理，输出的是距离值而不是分布。

### Q5: 如何调试后处理？
**A**: 
1. 打印量化参数(zp, scale)
2. 检查第一个检测结果的原始值
3. 对比Python推理结果
4. 使用`modern_dual_comparator.py`验证

---

## 总结

1. **预处理统一性**：两个模型使用相同的letterbox预处理
2. **后处理差异性**：
   - Pose模型需要完整的DFL解码
   - Basketball模型输出已处理，直接使用
3. **关键细节**：
   - RGB顺序（不是BGR）
   - NHWC格式
   - 正确的类别映射
   - 适当的NMS阈值

通过理解这些技术细节，可以避免在后处理上花费过多时间调试。记住：**模型的输出格式决定了后处理的复杂度**。

---

*最后更新：2025-01-25*