#ifndef _BASKETBALL_POSTPROCESS_H
#define _BASKETBALL_POSTPROCESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BASKETBALL_CLASS_NUM 2  // player(0), basketball(1)
#define BASKETBALL_MAX_DETECTIONS 50

typedef struct {
    float x, y, w, h;
    float confidence;
    int class_id;
} BasketballDetection;

typedef struct {
    BasketballDetection detections[BASKETBALL_MAX_DETECTIONS];
    int count;
} BasketballDetectionResult;

// 专门的篮球检测后处理函数
int process_basketball_yolov8_output(
    void** outputs,           // RKNN输出缓冲区
    int* output_dims,         // 每个输出的维度信息 [n, c, h, w]
    int32_t* output_zps,      // 每个输出的零点
    float* output_scales,     // 每个输出的缩放因子
    int num_outputs,          // 输出数量(应该是9)
    float conf_threshold,     // 置信度阈值
    BasketballDetectionResult* result
);

#ifdef __cplusplus
}
#endif

#endif // _BASKETBALL_POSTPROCESS_H 