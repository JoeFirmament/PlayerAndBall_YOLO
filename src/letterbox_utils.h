#ifndef LETTERBOX_UTILS_H
#define LETTERBOX_UTILS_H

#include <opencv2/opencv.hpp>
#include <stdint.h>
#include "rknn_api.h"

// letterbox变换上下文结构体
// 设计原因: 保存所有letterbox相关的参数，避免重复计算
typedef struct {
    // 原始图像尺寸
    int src_width;
    int src_height;
    
    // 模型输入尺寸
    int dst_width;
    int dst_height;
    
    // letterbox缩放参数
    float scale;        // 实际使用的缩放比例
    int new_width;      // 缩放后的宽度
    int new_height;     // 缩放后的高度
    int offset_x;       // 水平偏移量
    int offset_y;       // 垂直偏移量
    
    // 用于调试的标志
    bool debug_mode;
} letterbox_context_t;

// 零拷贝letterbox上下文（专门用于NPU零拷贝）
typedef struct {
    rknn_tensor_mem* input_mem;        // NPU输入内存
    rknn_tensor_attr input_attr;       // 输入属性
    int model_width;                   // 模型输入宽度
    int model_height;                  // 模型输入高度
    int model_channels;                // 模型输入通道数
    letterbox_context_t letterbox_ctx; // letterbox参数
} zero_copy_letterbox_context_t;

#ifdef __cplusplus
extern "C" {
#endif

// 初始化letterbox上下文
// 参数: src_width, src_height - 原始图像尺寸
//       dst_width, dst_height - 目标尺寸（模型输入尺寸）
//       debug_mode - 是否启用调试模式
void init_letterbox_context(letterbox_context_t* ctx, 
                           int src_width, int src_height,
                           int dst_width, int dst_height,
                           bool debug_mode);

// 打印letterbox参数（用于调试）
void print_letterbox_params(const letterbox_context_t* ctx);

// 标准letterbox预处理（OpenCV版本）
// 参数: src_mat - 输入图像
//       dst_mat - 输出图像（已分配内存）
//       ctx - letterbox上下文
int letterbox_preprocess(const cv::Mat& src_mat, cv::Mat& dst_mat, 
                        const letterbox_context_t* ctx);

// 零拷贝letterbox预处理（直接写入NPU内存）
// 参数: src_mat - 输入图像
//       zc_ctx - 零拷贝上下文
int zero_copy_letterbox_preprocess(const cv::Mat& src_mat, 
                                  zero_copy_letterbox_context_t* zc_ctx);

// 坐标反变换：模型输出坐标 → 原始图像坐标
// 参数: model_x, model_y - 模型输出坐标
//       ctx - letterbox上下文
//       返回值: 原始图像坐标
cv::Point2f letterbox_inverse_transform(float model_x, float model_y, 
                                       const letterbox_context_t* ctx);

// 批量坐标反变换
// 参数: model_points - 模型输出坐标数组
//       original_points - 原始图像坐标数组（输出）
//       count - 坐标点数量
//       ctx - letterbox上下文
void letterbox_inverse_transform_batch(const cv::Point2f* model_points,
                                      cv::Point2f* original_points,
                                      int count,
                                      const letterbox_context_t* ctx);

// 边界框反变换：模型输出边界框 → 原始图像边界框
// 参数: model_box - 模型输出边界框 (x, y, w, h)
//       ctx - letterbox上下文
//       返回值: 原始图像边界框
cv::Rect letterbox_inverse_transform_bbox(const cv::Rect& model_box,
                                         const letterbox_context_t* ctx);

// 关键点反变换（专门用于姿态检测）
// 参数: model_keypoints - 模型输出关键点 [N][3]（包含置信度）
//       original_keypoints - 原始图像关键点 [N][2]（输出）
//       keypoint_count - 关键点数量
//       ctx - letterbox上下文
void letterbox_inverse_transform_keypoints(const float model_keypoints[][3],
                                          float original_keypoints[][2],
                                          int keypoint_count,
                                          const letterbox_context_t* ctx);

// 验证letterbox变换的准确性（调试用）
// 参数: ctx - letterbox上下文
//       返回值: 0表示验证通过，-1表示验证失败
int validate_letterbox_transform(const letterbox_context_t* ctx);

#ifdef __cplusplus
}
#endif

#endif // LETTERBOX_UTILS_H 