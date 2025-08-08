/*-------------------------------------------
        篮球框和篮球检测后处理模块 - 简化版
        
基于详细文档YOLOV8_RK3588_OUTPUT_TENSORS_DETAILED_GUIDE.md:
- 支持6输出格式 (reg1, cls1, reg2, cls2, reg3, cls3)
- 2类检测: rim(0), basketball(1)
- 完全基于文档的后处理逻辑
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>

// RKNN
#include "rknn_api.h"
#include "detector_rim_basketball_postprocess.h"

// 类别名称 (根据模型实际训练顺序)
static const char* class_names[2] = {"basketball", "rim"};

// 模型参数
static const int IMG_SIZE = 640;
static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};  // P3, P4, P5层的步长
static const int MAP_SIZES[3][2] = {{80, 80}, {40, 40}, {20, 20}};

// 内部检测结果结构体
typedef struct {
    float xmin, ymin, xmax, ymax;
    float score;
    int class_id;
} DetectRect;

// 辅助函数：快速指数函数
static inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v;
    v.i = (uint32_t)(12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

// 辅助函数：sigmoid激活
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

// 标准的反量化函数（参考pose模型）
static inline float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

// 置信度阈值量化函数（参考pose模型）
static inline int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    return (int8_t)fmaxf(-128, fminf(127, dst_val));
}

// unsigmoid函数（参考pose模型）
static inline float unsigmoid(float y) {
    return -logf((1.0f / y) - 1.0f);
}

// 辅助函数：计算IoU
static float calculate_iou(float xmin1, float ymin1, float xmax1, float ymax1,
                          float xmin2, float ymin2, float xmax2, float ymax2) {
    float xmin = fmaxf(xmin1, xmin2);
    float ymin = fmaxf(ymin1, ymin2);
    float xmax = fminf(xmax1, xmax2);
    float ymax = fminf(ymax1, ymax2);
    
    float inter_width = xmax - xmin;
    float inter_height = ymax - ymin;
    
    if (inter_width <= 0 || inter_height <= 0) return 0.0f;
    
    float intersection = inter_width * inter_height;
    float area1 = (xmax1 - xmin1) * (ymax1 - ymin1);
    float area2 = (xmax2 - xmin2) * (ymax2 - ymin2);
    float union_area = area1 + area2 - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

// 基于详细文档的6输出模型后处理
static int postprocess_6_outputs_basketball_rim(rknn_output* outputs, rknn_tensor_attr* output_attrs,
                                              float conf_threshold, float nms_threshold,
                                              RimBasketballDetectionResult* result) {
    result->count = 0;
    
    // 模型参数
    const int strides[3] = {8, 16, 32};
    const int map_sizes[3][2] = {{80, 80}, {40, 40}, {20, 20}};
    
    // 获取输出指针和量化参数
    int8_t* reg_outputs[3] = {(int8_t*)outputs[0].buf, (int8_t*)outputs[2].buf, (int8_t*)outputs[4].buf};
    int8_t* cls_outputs[3] = {(int8_t*)outputs[1].buf, (int8_t*)outputs[3].buf, (int8_t*)outputs[5].buf};
    
    
    
    // 统计每帧检测数量
    int frame_detections = 0;
    int rim_count = 0;
    int basketball_count = 0;
    
    std::vector<DetectRect> detect_rects;
    
    // 处理3个检测层
    for (int layer = 0; layer < 3; layer++) {
        int stride = strides[layer];
        int height = map_sizes[layer][0];
        int width = map_sizes[layer][1];
        
        int8_t* reg_data = reg_outputs[layer];
        int8_t* cls_data = cls_outputs[layer];
        
        // 量化参数
        int reg_zp = output_attrs[layer * 2].zp;
        float reg_scale = output_attrs[layer * 2].scale;
        int cls_zp = output_attrs[layer * 2 + 1].zp;
        float cls_scale = output_attrs[layer * 2 + 1].scale;
        
        
        
        // 遍历网格
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // 获取最高置信度的类别
                float max_conf = 0.0f;
                int best_class = 0;
                
                // 使用量化阈值进行快速筛选（参考pose模型的优化）
                int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(conf_threshold), cls_zp, cls_scale);
                
                for (int c = 0; c < 2; c++) { // 2个类别：篮球(0), 篮筐(1)
                    int cls_idx = c * height * width + h * width + w;
                    
                    // 先用量化阈值快速筛选
                    if (cls_data[cls_idx] >= thres_i8) {
                        // 只对可能的候选进行反量化和sigmoid计算
                        float raw_score = deqnt_affine_to_f32(cls_data[cls_idx], cls_zp, cls_scale);
                        float conf = sigmoid(raw_score);
                        
                        // 调试：显示量化后的数值分布
                        static int sample_count = 0;
                        if (sample_count < 10 && layer == 0 && h < 3 && w < 3) {
                            printf("量化调试[%d]: 网格[%d,%d] 类别%d raw_int8=%d, thres_i8=%d, raw_float=%.4f, sigmoid=%.4f\n", 
                                   sample_count, h, w, c, cls_data[cls_idx], thres_i8, raw_score, conf);
                            sample_count++;
                        }
                        
                        if (conf > max_conf) {
                            max_conf = conf;
                            best_class = c;
                        }
                    }
                }
                
                
                
                // 检查是否超过阈值
                if (max_conf > conf_threshold) {
                    // 根据输出形状[1,1,4,H*W]，reg_data实际包含4个坐标值
                    // 修正访问方式：reg_data的布局应该是[4][H*W]
                    int grid_pos = h * width + w;
                    int hw_size = height * width;
                    
                    // 安全检查索引范围
                    if (grid_pos >= hw_size) continue;
                    
                    // 按照[4, H*W]格式访问
                    float left_dist   = deqnt_affine_to_f32(reg_data[0 * hw_size + grid_pos], reg_zp, reg_scale);
                    float top_dist    = deqnt_affine_to_f32(reg_data[1 * hw_size + grid_pos], reg_zp, reg_scale);
                    float right_dist  = deqnt_affine_to_f32(reg_data[2 * hw_size + grid_pos], reg_zp, reg_scale);
                    float bottom_dist = deqnt_affine_to_f32(reg_data[3 * hw_size + grid_pos], reg_zp, reg_scale);
                    
                    // 计算anchor center
                    float anchor_x = (w + 0.5f) * stride;
                    float anchor_y = (h + 0.5f) * stride;
                    
                    // 基于DFL处理后的距离计算边界框（距离值需要乘以stride）
                    float x1 = anchor_x - left_dist * stride;
                    float y1 = anchor_y - top_dist * stride;
                    float x2 = anchor_x + right_dist * stride;
                    float y2 = anchor_y + bottom_dist * stride;
                    
                    
                    
                    // 边界检查
                    x1 = fmaxf(0.0f, fminf(x1, 640.0f));
                    y1 = fmaxf(0.0f, fminf(y1, 640.0f));
                    x2 = fmaxf(0.0f, fminf(x2, 640.0f));
                    y2 = fmaxf(0.0f, fminf(y2, 640.0f));
                    
                    if (x1 < x2 && y1 < y2) {
                        DetectRect rect;
                        rect.xmin = x1 / 640.0f;  // 归一化
                        rect.ymin = y1 / 640.0f;
                        rect.xmax = x2 / 640.0f; 
                        rect.ymax = y2 / 640.0f;
                        rect.score = max_conf;
                        rect.class_id = best_class;
                        
                        detect_rects.push_back(rect);
                        frame_detections++;
                        if (best_class == 0) basketball_count++;
                        else if (best_class == 1) rim_count++;
                        
                        
                    }
                }
            }
        }
    }
    
    
    
    if (detect_rects.empty()) {
        return 0;
    }
    
    // 按置信度排序
    std::sort(detect_rects.begin(), detect_rects.end(),
              [](const DetectRect& a, const DetectRect& b) {
                  return a.score > b.score;
              });
    
    // NMS处理
    std::vector<bool> suppressed(detect_rects.size(), false);
    
    for (int i = 0; i < detect_rects.size(); i++) {
        if (suppressed[i]) continue;
        
        const DetectRect& rect_i = detect_rects[i];
        
        for (int j = i + 1; j < detect_rects.size(); j++) {
            if (suppressed[j]) continue;
            
            const DetectRect& rect_j = detect_rects[j];
            
            // 同类别才进行NMS
            if (rect_i.class_id == rect_j.class_id) {
                float iou = calculate_iou(rect_i.xmin, rect_i.ymin, rect_i.xmax, rect_i.ymax,
                                        rect_j.xmin, rect_j.ymin, rect_j.xmax, rect_j.ymax);
                
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    // 构建最终结果
    for (int i = 0; i < detect_rects.size() && result->count < MAX_DETECTIONS; i++) {
        if (suppressed[i]) continue;
        
        const DetectRect& rect = detect_rects[i];
        RimBasketballDetection* det = &result->detections[result->count];
        
        // 转换回像素坐标
        det->x = (rect.xmin + rect.xmax) / 2.0f * 640;  // center_x
        det->y = (rect.ymin + rect.ymax) / 2.0f * 640;  // center_y
        det->w = (rect.xmax - rect.xmin) * 640;         // width
        det->h = (rect.ymax - rect.ymin) * 640;         // height
        det->confidence = rect.score;
        det->class_id = rect.class_id;
        det->class_name = class_names[rect.class_id];
        
        result->count++;
    }
    
    
    
    return 0;
}

// 主要的后处理函数
int process_rim_basketball_outputs(rknn_output* outputs, rknn_tensor_attr* output_attrs, 
                                  float conf_threshold, float nms_threshold,
                                  RimBasketballDetectionResult* result) {
    if (!outputs || !output_attrs || !result) {
        printf("❌ 后处理参数错误\n");
        return -1;
    }
    
    // 获取输出数量
    int num_outputs = 0;
    while (num_outputs < 10 && output_attrs[num_outputs].n_dims > 0) {
        num_outputs++;
    }
    
    
    
    // 打印输出维度信息
    
    
    // 支持6输出格式
    if (num_outputs == 6) {
        
        return postprocess_6_outputs_basketball_rim(outputs, output_attrs, conf_threshold, nms_threshold, result);
    } else {
        printf("❌ 当前仅支持6输出模型格式，检测到: %d个输出\n", num_outputs);
        result->count = 0;
        return -1;
    }
}