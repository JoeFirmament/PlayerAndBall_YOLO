#ifndef _RIM_BASKETBALL_POSTPROCESS_H
#define _RIM_BASKETBALL_POSTPROCESS_H

#include <stdint.h>
#include "rknn_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BASKETBALL_CLASS_ID 0
#define RIM_CLASS_ID 1
#define MAX_DETECTIONS 100

// 检测结果结构体
typedef struct {
    float x, y, w, h;       // 边界框坐标 (center_x, center_y, width, height)
    float confidence;       // 置信度
    int class_id;           // 类别ID: 0=basketball, 1=rim
    const char* class_name; // 类别名称
} RimBasketballDetection;

typedef struct {
    RimBasketballDetection detections[MAX_DETECTIONS];
    int count;
} RimBasketballDetectionResult;

// 后处理函数
int process_rim_basketball_outputs(
    rknn_output* outputs,           // RKNN输出
    rknn_tensor_attr* output_attrs, // 输出属性
    float conf_threshold,           // 置信度阈值
    float nms_threshold,            // NMS阈值
    RimBasketballDetectionResult* result
);

#ifdef __cplusplus
}
#endif

#endif // _RIM_BASKETBALL_POSTPROCESS_H