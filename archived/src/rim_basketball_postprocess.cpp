/*-------------------------------------------
        篮球框和篮球检测后处理模块
        
基于modern_dual_comparator.py的C++实现:
- 支持6输出和9输出两种模型格式
- DFL (Distribution Focal Loss) 解码
- 多尺度特征处理 (80x80, 40x40, 20x20)
- NMS处理和坐标转换
- 2类检测: rim(0), basketball(1)
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <numeric>

// OpenCV
#include <opencv2/opencv.hpp>

// RKNN
#include "rknn_api.h"
#include "rim_basketball_postprocess.h"

// 类别名称
static const char* class_names[2] = {"rim", "basketball"};

// 模型参数
static const int IMG_SIZE = 640;
static const float STRIDES[3] = {8.0f, 16.0f, 32.0f};  // P3, P4, P5层的步长
static const int DFL_LEN = 16;  // DFL长度

// 辅助函数：sigmoid激活
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-fmaxf(fminf(x, 250.0f), -250.0f)));
}

// 辅助函数：计算IoU
static float calculate_iou(const float* box1, const float* box2) {
    float x1 = fmaxf(box1[0], box2[0]);
    float y1 = fmaxf(box1[1], box2[1]);
    float x2 = fminf(box1[2], box2[2]);
    float y2 = fminf(box1[3], box2[3]);
    
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

// DFL解码：将分布转换为距离值
static void decode_dfl(const float* dfl_data, int dfl_len, float* distances, int num_anchors) {
    for (int i = 0; i < num_anchors; i++) {
        for (int j = 0; j < 4; j++) { // 4个方向：left, top, right, bottom
            const float* dist_pred = dfl_data + (i * 4 + j) * dfl_len;
            
            // 稳定的softmax
            float max_val = dist_pred[0];
            for (int k = 1; k < dfl_len; k++) {
                if (dist_pred[k] > max_val) max_val = dist_pred[k];
            }
            
            float sum_exp = 0.0f;
            float softmax[DFL_LEN];
            for (int k = 0; k < dfl_len; k++) {
                softmax[k] = expf(fmaxf(fminf(dist_pred[k] - max_val, 88.0f), -88.0f));
                sum_exp += softmax[k];
            }
            
            // 加权求和得到距离
            float distance = 0.0f;
            for (int k = 0; k < dfl_len; k++) {
                distance += (softmax[k] / (sum_exp + 1e-8f)) * k;
            }
            
            distances[i * 4 + j] = distance;
        }
    }
}

// 从DFL距离和anchor生成边界框
static void decode_bboxes_from_dfl(const float* dfl_distances, const float* anchors, 
                                  float stride, int num_anchors, float* boxes) {
    for (int i = 0; i < num_anchors; i++) {
        float anchor_x = anchors[i * 2];
        float anchor_y = anchors[i * 2 + 1];
        
        float left = dfl_distances[i * 4] * stride;
        float top = dfl_distances[i * 4 + 1] * stride;
        float right = dfl_distances[i * 4 + 2] * stride;
        float bottom = dfl_distances[i * 4 + 3] * stride;
        
        boxes[i * 4] = anchor_x - left;      // x1
        boxes[i * 4 + 1] = anchor_y - top;   // y1
        boxes[i * 4 + 2] = anchor_x + right; // x2
        boxes[i * 4 + 3] = anchor_y + bottom;// y2
    }
}

// NMS处理
static std::vector<int> nms_boxes(const std::vector<float>& boxes, const std::vector<float>& scores, 
                                 const std::vector<int>& class_ids, float nms_threshold) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // 按分数降序排序
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });
    
    std::vector<bool> suppressed(scores.size(), false);
    std::vector<int> keep;
    
    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        // 动态NMS阈值：篮球使用更宽松的阈值
        float thresh = nms_threshold;
        if (class_ids[idx] == BASKETBALL_CLASS_ID) {
            thresh = 0.2f; // 篮球用更宽松的阈值
        }
        
        for (int j = i + 1; j < indices.size(); j++) {
            int other_idx = indices[j];
            if (suppressed[other_idx]) continue;
            
            // 计算IoU
            float box1[4] = {boxes[idx * 4], boxes[idx * 4 + 1], 
                           boxes[idx * 4 + 2], boxes[idx * 4 + 3]};
            float box2[4] = {boxes[other_idx * 4], boxes[other_idx * 4 + 1], 
                           boxes[other_idx * 4 + 2], boxes[other_idx * 4 + 3]};
            
            float iou = calculate_iou(box1, box2);
            if (iou > thresh) {
                suppressed[other_idx] = true;
            }
        }
    }
    
    return keep;
}

// 6输出模型的后处理 (reg1, cls1, reg2, cls2, reg3, cls3)
static int postprocess_6_outputs(rknn_output* outputs, rknn_tensor_attr* output_attrs,
                                float conf_threshold, float nms_threshold, int orig_w, int orig_h,
                                RimBasketballDetectionResult* result) {
    result->count = 0;
    std::vector<float> all_boxes, all_scores;
    std::vector<int> all_class_ids;
    
    // 处理3个尺度
    for (int scale = 0; scale < 3; scale++) {
        int reg_idx = scale * 2;     // 0, 2, 4
        int cls_idx = scale * 2 + 1; // 1, 3, 5
        
        float stride = STRIDES[scale];
        
        // 获取输出数据
        float* reg_data = (float*)outputs[reg_idx].buf;
        float* cls_data = (float*)outputs[cls_idx].buf;
        
        // 解析输出维度
        rknn_tensor_attr* cls_attr = &output_attrs[cls_idx];
        int height = cls_attr->dims[2];
        int width = cls_attr->dims[3];
        int num_anchors = height * width;
        
        printf("尺度%d: %dx%d, stride=%.0f, anchors=%d\n", scale, height, width, stride, num_anchors);
        
        // 创建anchor网格
        std::vector<float> anchors(num_anchors * 2);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = h * width + w;
                anchors[idx * 2] = (w + 0.5f) * stride;     // anchor_x
                anchors[idx * 2 + 1] = (h + 0.5f) * stride; // anchor_y
            }
        }
        
        // 处理分类输出 (应用sigmoid)
        std::vector<float> cls_scores(num_anchors * 2);
        for (int i = 0; i < num_anchors * 2; i++) {
            cls_scores[i] = sigmoid(cls_data[i]);
        }
        
        // 获取最大类别分数和索引
        std::vector<float> max_scores(num_anchors);
        std::vector<int> class_ids(num_anchors);
        for (int i = 0; i < num_anchors; i++) {
            float score0 = cls_scores[i * 2];
            float score1 = cls_scores[i * 2 + 1];
            if (score0 > score1) {
                max_scores[i] = score0;
                class_ids[i] = 0; // rim
            } else {
                max_scores[i] = score1;
                class_ids[i] = 1; // basketball
            }
        }
        
        // 过滤低置信度检测
        std::vector<int> valid_indices;
        for (int i = 0; i < num_anchors; i++) {
            if (max_scores[i] > conf_threshold) {
                valid_indices.push_back(i);
            }
        }
        
        if (valid_indices.empty()) continue;
        
        // DFL解码回归输出
        std::vector<float> dfl_distances(valid_indices.size() * 4);
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            const float* reg_anchor_data = reg_data + anchor_idx * 4 * DFL_LEN;
            decode_dfl(reg_anchor_data, DFL_LEN, &dfl_distances[i * 4], 1);
        }
        
        // 生成边界框
        std::vector<float> boxes(valid_indices.size() * 4);
        std::vector<float> valid_anchors(valid_indices.size() * 2);
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            valid_anchors[i * 2] = anchors[anchor_idx * 2];
            valid_anchors[i * 2 + 1] = anchors[anchor_idx * 2 + 1];
        }
        
        decode_bboxes_from_dfl(dfl_distances.data(), valid_anchors.data(), 
                              stride, valid_indices.size(), boxes.data());
        
        // 添加到全局结果
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            
            // 检查边界框有效性
            if (boxes[i * 4 + 2] > boxes[i * 4] && boxes[i * 4 + 3] > boxes[i * 4 + 1]) {
                all_boxes.insert(all_boxes.end(), &boxes[i * 4], &boxes[i * 4 + 4]);
                all_scores.push_back(max_scores[anchor_idx]);
                all_class_ids.push_back(class_ids[anchor_idx]);
            }
        }
    }
    
    if (all_boxes.empty()) return 0;
    
    // 坐标缩放到原始图像尺寸
    float scale_x = (float)orig_w / IMG_SIZE;
    float scale_y = (float)orig_h / IMG_SIZE;
    
    for (int i = 0; i < all_boxes.size() / 4; i++) {
        all_boxes[i * 4] *= scale_x;     // x1
        all_boxes[i * 4 + 2] *= scale_x; // x2
        all_boxes[i * 4 + 1] *= scale_y; // y1
        all_boxes[i * 4 + 3] *= scale_y; // y2
    }
    
    // NMS处理
    std::vector<int> nms_indices = nms_boxes(all_boxes, all_scores, all_class_ids, nms_threshold);
    
    // 构建最终检测结果
    result->count = std::min((int)nms_indices.size(), MAX_DETECTIONS);
    for (int i = 0; i < result->count; i++) {
        int idx = nms_indices[i];
        
        RimBasketballDetection* det = &result->detections[i];
        
        float x1 = all_boxes[idx * 4];
        float y1 = all_boxes[idx * 4 + 1];
        float x2 = all_boxes[idx * 4 + 2];
        float y2 = all_boxes[idx * 4 + 3];
        
        det->x = (x1 + x2) / 2.0f;  // center_x
        det->y = (y1 + y2) / 2.0f;  // center_y
        det->w = x2 - x1;           // width
        det->h = y2 - y1;           // height
        det->confidence = all_scores[idx];
        det->class_id = all_class_ids[idx];
        det->class_name = class_names[det->class_id];
    }
    
    printf("✓ 检测完成: %d个目标 (NMS前: %d)\n", result->count, (int)all_scores.size());
    return 0;
}

// 9输出模型的后处理 (dfl, cls, obj) * 3
static int postprocess_9_outputs(rknn_output* outputs, rknn_tensor_attr* output_attrs,
                                float conf_threshold, float nms_threshold, int orig_w, int orig_h,
                                RimBasketballDetectionResult* result) {
    result->count = 0;
    std::vector<float> all_boxes, all_scores;
    std::vector<int> all_class_ids;
    
    // 处理3个尺度
    for (int scale = 0; scale < 3; scale++) {
        int dfl_idx = scale * 3;     // 0, 3, 6
        int cls_idx = scale * 3 + 1; // 1, 4, 7
        int obj_idx = scale * 3 + 2; // 2, 5, 8
        
        float stride = STRIDES[scale];
        
        // 获取输出数据
        float* dfl_data = (float*)outputs[dfl_idx].buf;
        float* cls_data = (float*)outputs[cls_idx].buf;
        float* obj_data = (float*)outputs[obj_idx].buf;
        
        // 解析输出维度
        rknn_tensor_attr* cls_attr = &output_attrs[cls_idx];
        int height = cls_attr->dims[2];
        int width = cls_attr->dims[3];
        int num_anchors = height * width;
        
        printf("尺度%d: %dx%d, stride=%.0f, anchors=%d\n", scale, height, width, stride, num_anchors);
        
        // 处理objectness和分类分数
        std::vector<float> final_scores(num_anchors);
        std::vector<int> class_ids(num_anchors);
        
        for (int i = 0; i < num_anchors; i++) {
            float obj_score = sigmoid(obj_data[i]);
            float cls_score0 = sigmoid(cls_data[i * 2]);
            float cls_score1 = sigmoid(cls_data[i * 2 + 1]);
            
            float max_cls_score;
            int class_id;
            if (cls_score0 > cls_score1) {
                max_cls_score = cls_score0;
                class_id = 0; // rim
            } else {
                max_cls_score = cls_score1;
                class_id = 1; // basketball
            }
            
            final_scores[i] = max_cls_score * obj_score;
            class_ids[i] = class_id;
        }
        
        // 过滤低置信度检测
        std::vector<int> valid_indices;
        for (int i = 0; i < num_anchors; i++) {
            if (final_scores[i] > conf_threshold) {
                valid_indices.push_back(i);
            }
        }
        
        if (valid_indices.empty()) continue;
        
        // 创建anchor网格
        std::vector<float> anchors(num_anchors * 2);
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int idx = h * width + w;
                anchors[idx * 2] = (w + 0.5f) * stride;
                anchors[idx * 2 + 1] = (h + 0.5f) * stride;
            }
        }
        
        // DFL解码
        std::vector<float> dfl_distances(valid_indices.size() * 4);
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            const float* dfl_anchor_data = dfl_data + anchor_idx * 4 * DFL_LEN;
            decode_dfl(dfl_anchor_data, DFL_LEN, &dfl_distances[i * 4], 1);
        }
        
        // 生成边界框
        std::vector<float> boxes(valid_indices.size() * 4);
        std::vector<float> valid_anchors(valid_indices.size() * 2);
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            valid_anchors[i * 2] = anchors[anchor_idx * 2];
            valid_anchors[i * 2 + 1] = anchors[anchor_idx * 2 + 1];
        }
        
        decode_bboxes_from_dfl(dfl_distances.data(), valid_anchors.data(), 
                              stride, valid_indices.size(), boxes.data());
        
        // 添加到全局结果
        for (int i = 0; i < valid_indices.size(); i++) {
            int anchor_idx = valid_indices[i];
            
            if (boxes[i * 4 + 2] > boxes[i * 4] && boxes[i * 4 + 3] > boxes[i * 4 + 1]) {
                all_boxes.insert(all_boxes.end(), &boxes[i * 4], &boxes[i * 4 + 4]);
                all_scores.push_back(final_scores[anchor_idx]);
                all_class_ids.push_back(class_ids[anchor_idx]);
            }
        }
    }
    
    if (all_boxes.empty()) return 0;
    
    // 坐标缩放和NMS处理 (与6输出版本相同)
    float scale_x = (float)orig_w / IMG_SIZE;
    float scale_y = (float)orig_h / IMG_SIZE;
    
    for (int i = 0; i < all_boxes.size() / 4; i++) {
        all_boxes[i * 4] *= scale_x;
        all_boxes[i * 4 + 2] *= scale_x;
        all_boxes[i * 4 + 1] *= scale_y;
        all_boxes[i * 4 + 3] *= scale_y;
    }
    
    std::vector<int> nms_indices = nms_boxes(all_boxes, all_scores, all_class_ids, nms_threshold);
    
    result->count = std::min((int)nms_indices.size(), MAX_DETECTIONS);
    for (int i = 0; i < result->count; i++) {
        int idx = nms_indices[i];
        
        RimBasketballDetection* det = &result->detections[i];
        
        float x1 = all_boxes[idx * 4];
        float y1 = all_boxes[idx * 4 + 1];
        float x2 = all_boxes[idx * 4 + 2];
        float y2 = all_boxes[idx * 4 + 3];
        
        det->x = (x1 + x2) / 2.0f;
        det->y = (y1 + y2) / 2.0f;
        det->w = x2 - x1;
        det->h = y2 - y1;
        det->confidence = all_scores[idx];
        det->class_id = all_class_ids[idx];
        det->class_name = class_names[det->class_id];
    }
    
    printf("✓ 检测完成: %d个目标 (NMS前: %d)\n", result->count, (int)all_scores.size());
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
    
    printf("检测到%d个模型输出\n", num_outputs);
    
    // 打印输出维度信息
    for (int i = 0; i < num_outputs; i++) {
        printf("输出[%d]: ", i);
        for (int j = 0; j < output_attrs[i].n_dims; j++) {
            printf("%d ", output_attrs[i].dims[j]);
        }
        printf("(size: %d)\n", output_attrs[i].size);
    }
    
    // 根据输出数量选择后处理方法
    if (num_outputs == 6) {
        printf("使用6输出模型后处理\n");
        return postprocess_6_outputs(outputs, output_attrs, conf_threshold, nms_threshold, 
                                   640, 640, result);  // 暂时使用模型尺寸
    } else if (num_outputs == 9) {
        printf("使用9输出模型后处理\n");
        return postprocess_9_outputs(outputs, output_attrs, conf_threshold, nms_threshold, 
                                   640, 640, result);  // 暂时使用模型尺寸
    } else {
        printf("❌ 不支持的模型输出格式: %d个输出\n", num_outputs);
        result->count = 0;
        return -1;
    }
}