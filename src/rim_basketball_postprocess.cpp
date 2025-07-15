/*-------------------------------------------
        篮球框和篮球检测后处理模块
        
处理9输出模型结构：
- 3个尺度: 80x80, 40x40, 20x20
- 每个尺度3个输出: 64通道特征, 2通道分类, 1通道置信度
- INT8量化反算
- 2类检测: rim(0), basketball(1)
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>

// OpenCV
#include <opencv2/opencv.hpp>

// RKNN
#include "rknn_api.h"

// 类别定义
#define RIM_CLASS_ID 0
#define BASKETBALL_CLASS_ID 1
#define MAX_DETECTIONS 100

// 检测结果结构体
typedef struct {
    float x, y, w, h;       // 边界框坐标 (center_x, center_y, width, height)
    float confidence;       // 置信度
    int class_id;           // 类别ID: 0=rim, 1=basketball
    const char* class_name; // 类别名称
} RimBasketballDetection;

typedef struct {
    RimBasketballDetection detections[MAX_DETECTIONS];
    int count;
} RimBasketballDetectionResult;

// 类别名称
extern const char* class_names[2];

// anchor配置（基于YOLO标准）
static const int ANCHOR_GRIDS[3] = {80, 40, 20};  // 3个尺度
static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};  // 对应步长

// DFL (Distribution Focal Loss) 长度
static const int DFL_LEN = 16;  // 64通道 / 4 = 16

// INT8量化反算
static inline float dequantize_int8(int8_t quantized_value, float scale, int32_t zero_point) {
    return (quantized_value - zero_point) * scale;
}

// Softmax函数
static void softmax(float* input, int length, float* output) {
    float max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// DFL解码：将分布转换为坐标偏移
static float decode_dfl(float* dfl_data, int dfl_len) {
    float result = 0.0f;
    for (int i = 0; i < dfl_len; i++) {
        result += i * dfl_data[i];
    }
    return result;
}

// 处理单个尺度的输出
static int process_single_scale(
    void* feature_output,     // 64通道特征 [1,64,H,W]
    void* class_output,       // 2通道分类 [1,2,H,W]
    void* conf_output,        // 1通道置信度 [1,1,H,W]
    int grid_size,            // H=W (80, 40, 20)
    float stride,             // 步长 (8, 16, 32)
    rknn_tensor_attr* feature_attr,
    rknn_tensor_attr* class_attr,
    rknn_tensor_attr* conf_attr,
    float conf_threshold,
    std::vector<RimBasketballDetection>& detections
) {
    // INT8指针
    int8_t* feature_ptr = (int8_t*)feature_output;
    int8_t* class_ptr = (int8_t*)class_output;
    int8_t* conf_ptr = (int8_t*)conf_output;
    
    // 量化参数
    float feature_scale = feature_attr->scale;
    int32_t feature_zp = feature_attr->zp;
    float class_scale = class_attr->scale;
    int32_t class_zp = class_attr->zp;
    float conf_scale = conf_attr->scale;
    int32_t conf_zp = conf_attr->zp;
    
    // 只在第一个尺度输出量化参数
    static bool first_call = true;
    if (first_call && grid_size == 80) {  // 第一个尺度
        printf("[量化参数] 特征: scale=%.6f, zp=%d | 分类: scale=%.6f, zp=%d | 置信度: scale=%.6f, zp=%d\n",
               feature_scale, feature_zp, class_scale, class_zp, conf_scale, conf_zp);
        
        // 采样前几个位置的原始数据进行调试
        printf("[数据采样] 前5个位置的原始INT8值:\n");
        for (int sample_idx = 0; sample_idx < 5; sample_idx++) {
            printf("  位置%d: 特征=[%d,%d,%d,%d] 分类=[%d,%d] 置信度=[%d]\n", 
                   sample_idx,
                   feature_ptr[sample_idx], feature_ptr[grid_size*grid_size + sample_idx], 
                   feature_ptr[2*grid_size*grid_size + sample_idx], feature_ptr[3*grid_size*grid_size + sample_idx],
                   class_ptr[sample_idx], class_ptr[grid_size*grid_size + sample_idx],
                   conf_ptr[sample_idx]);
        }
        
        first_call = false;
    }
    
    int valid_detections = 0;
    
    // 遍历网格
    for (int y = 0; y < grid_size; y++) {
        for (int x = 0; x < grid_size; x++) {
            int grid_idx = y * grid_size + x;
            
            // 1. 获取置信度 (objectness) - 快速预过滤
            int8_t conf_i8 = conf_ptr[grid_idx];
            
            // 快速过滤：避免浮点运算
            // INT8范围[-128,127]，需要根据实际量化参数调整
            if (conf_i8 < 0) {  // 更严格的预过滤
                continue;
            }
            
            float objectness = dequantize_int8(conf_i8, conf_scale, conf_zp);
            objectness = 1.0f / (1.0f + expf(-objectness));  // sigmoid
            
            if (objectness < conf_threshold) {
                continue;  // 跳过低置信度
            }
            
            // 2. 获取类别概率
            float class_probs[2];
            for (int c = 0; c < 2; c++) {
                int class_idx = c * grid_size * grid_size + grid_idx;
                int8_t class_i8 = class_ptr[class_idx];
                class_probs[c] = dequantize_int8(class_i8, class_scale, class_zp);
            }
            
            // Softmax归一化
            float class_probs_norm[2];
            softmax(class_probs, 2, class_probs_norm);
            
            // 找到最大概率的类别
            int best_class = (class_probs_norm[0] > class_probs_norm[1]) ? 0 : 1;
            float class_conf = class_probs_norm[best_class];
            
            // 3. 计算最终置信度
            float final_conf = objectness * class_conf;
            if (final_conf < conf_threshold) {
                continue;
            }
            
            // 4. 解析边界框坐标 (假设前4个通道是直接回归)
            float bbox_raw[4] = {0};  // [dx, dy, dw, dh] 或者 [x1, y1, x2, y2]
            
            for (int i = 0; i < 4; i++) {  // 只取前4个通道
                int feature_idx = i * grid_size * grid_size + grid_idx;
                int8_t feature_i8 = feature_ptr[feature_idx];
                bbox_raw[i] = dequantize_int8(feature_i8, feature_scale, feature_zp);
            }
            
            // 5. 转换为实际坐标 (尝试两种解释方式)
            float center_x = (x + 0.5f) * stride;
            float center_y = (y + 0.5f) * stride;
            
            // 方式1: 直接偏移量 (YOLOv8常用)
            float bbox_center_x = center_x + bbox_raw[0] * stride;
            float bbox_center_y = center_y + bbox_raw[1] * stride;
            float width = expf(bbox_raw[2]) * stride;
            float height = expf(bbox_raw[3]) * stride;
            
            // 方式2: 如果方式1结果异常，尝试直接坐标
            if (width > 640 || height > 640 || width < 1 || height < 1) {
                // 可能是直接坐标格式
                bbox_center_x = bbox_raw[0];
                bbox_center_y = bbox_raw[1];
                width = bbox_raw[2] - bbox_raw[0];
                height = bbox_raw[3] - bbox_raw[1];
            }
            
            // 基本合理性检查
            if (width <= 0 || height <= 0 || width > 640 || height > 640) {
                continue;
            }
            
            // 6. 创建检测结果
            RimBasketballDetection detection;
            detection.x = bbox_center_x;
            detection.y = bbox_center_y;
            detection.w = width;
            detection.h = height;
            detection.confidence = final_conf;
            detection.class_id = best_class;
            detection.class_name = class_names[best_class];
            
            detections.push_back(detection);
            valid_detections++;
            
            // 临时保护：限制单个尺度最大检测数防止爆炸
            if (valid_detections > 20) {
                printf("[警告] 尺度%dx%d检测数量超过20，可能存在问题，截断处理\n", grid_size, grid_size);
                goto scale_done;  // 跳出双重循环
            }
        }
    }
    
scale_done:
    return valid_detections;
}

// NMS (Non-Maximum Suppression)
static float calculate_iou(const RimBasketballDetection& a, const RimBasketballDetection& b) {
    float left = std::max(a.x - a.w/2, b.x - b.w/2);
    float top = std::max(a.y - a.h/2, b.y - b.h/2);
    float right = std::min(a.x + a.w/2, b.x + b.w/2);
    float bottom = std::min(a.y + a.h/2, b.y + b.h/2);
    
    if (right <= left || bottom <= top) {
        return 0.0f;
    }
    
    float intersection = (right - left) * (bottom - top);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection;
    
    return intersection / union_area;
}

static void apply_nms(std::vector<RimBasketballDetection>& detections, float nms_threshold) {
    // 按置信度排序
    std::sort(detections.begin(), detections.end(), 
              [](const RimBasketballDetection& a, const RimBasketballDetection& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            // 同类别才进行NMS
            if (detections[i].class_id == detections[j].class_id) {
                float iou = calculate_iou(detections[i], detections[j]);
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    // 移除被抑制的检测
    std::vector<RimBasketballDetection> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) {
            filtered.push_back(detections[i]);
        }
    }
    
    detections = filtered;
}

// 主后处理函数
extern "C" int process_rim_basketball_outputs(
    rknn_output* outputs,           // 9个RKNN输出
    rknn_tensor_attr* output_attrs, // 9个输出属性
    float conf_threshold,           // 置信度阈值
    float nms_threshold,            // NMS阈值
    RimBasketballDetectionResult* result
) {
    // 验证输出数量
    if (!outputs || !output_attrs || !result) {
        printf("[后处理错误] 输入参数为空\n");
        return -1;
    }
    
    printf("[后处理开始] 置信度阈值=%.3f, NMS阈值=%.3f\n", conf_threshold, nms_threshold);
    
    // 验证模型输出结构 (不再是DFL格式)
    int feature_channels = output_attrs[0].dims[1];  // 特征输出的通道数
    printf("[后处理验证] 特征通道数=%d (前4通道用于bbox回归)\n", feature_channels);
    
    if (feature_channels < 4) {
        printf("❌ 错误: 特征通道数不足! 需要至少4个通道用于bbox回归\n");
        return -1;
    }
    
    std::vector<RimBasketballDetection> all_detections;
    
    // 处理3个尺度
    for (int scale_idx = 0; scale_idx < 3; scale_idx++) {
        int grid_size = ANCHOR_GRIDS[scale_idx];
        float stride = STRIDES[scale_idx];
        
        // 计算当前尺度的输出索引
        int feature_idx = scale_idx * 3 + 0;  // 64通道特征
        int class_idx = scale_idx * 3 + 1;    // 2通道分类
        int conf_idx = scale_idx * 3 + 2;     // 1通道置信度
        
        printf("[后处理尺度%d] 网格=%dx%d, 步长=%.0f, 输出索引=[%d,%d,%d]\n", 
               scale_idx, grid_size, grid_size, stride, feature_idx, class_idx, conf_idx);
        
        printf("  特征输出: [%d,%d,%d,%d], 分类输出: [%d,%d,%d,%d], 置信度输出: [%d,%d,%d,%d]\n",
               output_attrs[feature_idx].dims[0], output_attrs[feature_idx].dims[1],
               output_attrs[feature_idx].dims[2], output_attrs[feature_idx].dims[3],
               output_attrs[class_idx].dims[0], output_attrs[class_idx].dims[1],
               output_attrs[class_idx].dims[2], output_attrs[class_idx].dims[3],
               output_attrs[conf_idx].dims[0], output_attrs[conf_idx].dims[1],
               output_attrs[conf_idx].dims[2], output_attrs[conf_idx].dims[3]);
        
        size_t before_count = all_detections.size();
        
        // 处理当前尺度
        process_single_scale(
            outputs[feature_idx].buf,
            outputs[class_idx].buf,
            outputs[conf_idx].buf,
            grid_size,
            stride,
            &output_attrs[feature_idx],
            &output_attrs[class_idx],
            &output_attrs[conf_idx],
            conf_threshold,
            all_detections
        );
        
        size_t after_count = all_detections.size();
        printf("  尺度%d处理完成: 新增检测%zu个, 总计%zu个\n", 
               scale_idx, after_count - before_count, after_count);
    }
    
    printf("[后处理完成] NMS前总检测数: %zu\n", all_detections.size());
    
    // NMS后处理
    apply_nms(all_detections, nms_threshold);
    
    printf("[后处理完成] NMS后总检测数: %zu\n", all_detections.size());
    
    // 复制结果
    result->count = std::min((int)all_detections.size(), MAX_DETECTIONS);
    for (int i = 0; i < result->count; i++) {
        result->detections[i] = all_detections[i];
    }
    
    printf("[后处理结束] 最终输出检测数: %d\n", result->count);
    
    return 0;
}