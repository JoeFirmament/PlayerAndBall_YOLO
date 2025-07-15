//
// 篮球检测后处理模块 - 专门针对RK3588平台优化
// 适配YOLOv8模型的2类别检测：player(0), basketball(1)
// 基于Rockchip官方postprocess.cc实现，针对RK3588平台特性优化
//
#include "basketball_postprocess.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <set>
#include <algorithm>

#define BASKETBALL_OBJ_CLASS_NUM 2  // player(0), basketball(1)

static char *basketball_labels[BASKETBALL_OBJ_CLASS_NUM] = {"player", "basketball"};

inline static int clamp(float val, int min, int max) { 
    return val > min ? (val < max ? val : max) : min; 
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, 
                              float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) {
                continue;
            }
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { 
    return ((float)qnt - (float)zp) * scale; 
}

static void compute_dfl(float* tensor, int dfl_len, float* box) {
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0;
        float acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = exp(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += exp_t[i] / exp_sum * i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8_basketball(int8_t *box_tensor, int32_t box_zp, float box_scale,
                                 int8_t *score_tensor, int32_t score_zp, float score_scale,
                                 int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                                 int grid_h, int grid_w, int stride, int dfl_len,
                                 std::vector<float> &boxes, 
                                 std::vector<float> &objProbs, 
                                 std::vector<int> &classId, 
                                 float threshold) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int offset = i * grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr) {
                if (score_sum_tensor[offset] < score_sum_thres_i8) {
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < BASKETBALL_OBJ_CLASS_NUM; c++) {
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score)) {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score > score_thres_i8) {
                offset = i * grid_w + j;
                float box[4];
                float before_dfl[dfl_len * 4];
                for (int k = 0; k < dfl_len * 4; k++) {
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1, y1, x2, y2, w, h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}

int process_basketball_yolov8_output(
    void** outputs,
    int* output_dims,
    int32_t* output_zps,
    float* output_scales,
    int num_outputs,
    float conf_threshold,
    BasketballDetectionResult* result)
{
    if (!outputs || !output_dims || !output_zps || !output_scales || !result) {
        return -1;
    }
    
    result->count = 0;
    
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int model_in_w = 640;  // 模型输入宽度
    int model_in_h = 640;  // 模型输入高度

    // RK3588平台: DFL长度计算 - 使用第一个输出的第二个维度除以4
    // 对于RK3588，输出格式为[N, C, H, W]，其中C维度包含DFL信息
    // output_dims数组格式: [output0_dim0, output0_dim1, output0_dim2, output0_dim3, output1_dim0, ...]
    int dfl_len = output_dims[1] / 4;  // 第一个输出的第二个维度(通道数)除以4
    int output_per_branch = num_outputs / 3;  // 每个尺度的输出数量，应该是3

    // printf("[basketball][RK3588] DFL长度=%d, 每分支输出数=%d, 总输出数=%d\n", 
    //        dfl_len, output_per_branch, num_outputs);

        // 处理3个尺度
    for (int i = 0; i < 3; i++) {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        
        if (output_per_branch == 3) {
            score_sum = outputs[i * output_per_branch + 2];
            score_sum_zp = output_zps[i * output_per_branch + 2];
            score_sum_scale = output_scales[i * output_per_branch + 2];
        }
        
        int box_idx = i * output_per_branch;
        int score_idx = i * output_per_branch + 1;
        
        // RK3588平台: 网格尺寸获取 - 输出格式为[N, C, H, W]
        // 根据原始postprocess.cc中RK3588的处理方式
        int dims_offset = box_idx * 4;  // 每个输出有4个维度
        grid_h = output_dims[dims_offset + 2];  // H维度 (dims[2])
        grid_w = output_dims[dims_offset + 3];  // W维度 (dims[3])
        stride = model_in_h / grid_h;
        
        // printf("[basketball][RK3588] 分支%d: 网格[%dx%d], 步长=%d, box_idx=%d, dims_offset=%d\n", 
        //        i, grid_h, grid_w, stride, box_idx, dims_offset);
        
        // 处理INT8量化输出
        validCount += process_i8_basketball(
            (int8_t *)outputs[box_idx], output_zps[box_idx], output_scales[box_idx],
            (int8_t *)outputs[score_idx], output_zps[score_idx], output_scales[score_idx],
            (int8_t *)score_sum, score_sum_zp, score_sum_scale,
            grid_h, grid_w, stride, dfl_len, 
            filterBoxes, objProbs, classId, conf_threshold);
    }

    // printf("[basketball][RK3588] 初步检测到 %d 个目标\n", validCount);

    // 如果没有检测到目标
    if (validCount <= 0) {
        return 0;
    }

    // 排序和NMS
    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) {
        indexArray.push_back(i);
    }
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classId), std::end(classId));
    float nms_threshold = 0.6;  // NMS阈值

    for (auto c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    // 输出最终结果
    int last_count = 0;
    for (int i = 0; i < validCount && last_count < BASKETBALL_MAX_DETECTIONS; ++i) {
        if (indexArray[i] == -1) {
            continue;
        }
        int n = indexArray[i];

        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];
        float obj_conf = objProbs[i];

        result->detections[last_count].x = clamp(x1, 0, model_in_w);
        result->detections[last_count].y = clamp(y1, 0, model_in_h);
        result->detections[last_count].w = clamp(x2, 0, model_in_w) - result->detections[last_count].x;
        result->detections[last_count].h = clamp(y2, 0, model_in_h) - result->detections[last_count].y;
        result->detections[last_count].confidence = obj_conf;
        result->detections[last_count].class_id = id;
        
        // printf("[basketball][final] %s @ (%.1f,%.1f,%.1f,%.1f) conf=%.3f\n",
        //        basketball_labels[id], 
        //        result->detections[last_count].x, result->detections[last_count].y,
        //        result->detections[last_count].w, result->detections[last_count].h,
        //        obj_conf);
        
        last_count++;
    }
    result->count = last_count;
    
    // printf("[basketball][RK3588] 最终输出 %d 个检测结果\n", result->count);
    return 0;
} 