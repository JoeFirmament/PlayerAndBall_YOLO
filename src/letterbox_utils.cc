#include "letterbox_utils.h"
#include <stdio.h>
#include <algorithm>
#include <cmath>

// 初始化letterbox上下文
void init_letterbox_context(letterbox_context_t* ctx, 
                           int src_width, int src_height,
                           int dst_width, int dst_height,
                           bool debug_mode) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return;
    }
    
    // 保存原始尺寸和目标尺寸
    ctx->src_width = src_width;
    ctx->src_height = src_height;
    ctx->dst_width = dst_width;
    ctx->dst_height = dst_height;
    ctx->debug_mode = debug_mode;
    
    // 计算letterbox缩放参数
    float scale_w = (float)dst_width / src_width;
    float scale_h = (float)dst_height / src_height;
    ctx->scale = std::min(scale_w, scale_h);  // 保持宽高比的缩放
    
    // 计算缩放后的尺寸
    ctx->new_width = (int)(src_width * ctx->scale);
    ctx->new_height = (int)(src_height * ctx->scale);
    
    // 计算居中偏移量
    ctx->offset_x = (dst_width - ctx->new_width) / 2;
    ctx->offset_y = (dst_height - ctx->new_height) / 2;
    
    if (debug_mode) {
        print_letterbox_params(ctx);
    }
}

// 打印letterbox参数（用于调试）
void print_letterbox_params(const letterbox_context_t* ctx) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return;
    }
    
    printf("=== Letterbox参数 ===\n");
    printf("原始尺寸: %dx%d\n", ctx->src_width, ctx->src_height);
    printf("目标尺寸: %dx%d\n", ctx->dst_width, ctx->dst_height);
    printf("缩放比例: %.6f\n", ctx->scale);
    printf("缩放后尺寸: %dx%d\n", ctx->new_width, ctx->new_height);
    printf("偏移量: (%d, %d)\n", ctx->offset_x, ctx->offset_y);
    printf("==================\n");
}

// 标准letterbox预处理（OpenCV版本）
int letterbox_preprocess(const cv::Mat& src_mat, cv::Mat& dst_mat, 
                        const letterbox_context_t* ctx) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return -1;
    }
    
    if (src_mat.empty()) {
        printf("[letterbox] 错误: 输入图像为空\n");
        return -1;
    }
    
    // 验证输入图像尺寸
    if (src_mat.cols != ctx->src_width || src_mat.rows != ctx->src_height) {
        printf("[letterbox] 警告: 输入图像尺寸不匹配 实际:%dx%d 期望:%dx%d\n", 
               src_mat.cols, src_mat.rows, ctx->src_width, ctx->src_height);
    }
    
    // 创建目标图像并填充灰色背景
    dst_mat = cv::Mat::zeros(ctx->dst_height, ctx->dst_width, CV_8UC3);
    dst_mat.setTo(cv::Scalar(114, 114, 114));  // 灰色背景
    
    // 缩放原始图像
    cv::Mat resized_mat;
    cv::resize(src_mat, resized_mat, cv::Size(ctx->new_width, ctx->new_height));
    
    // 将缩放后的图像复制到目标图像的居中位置
    cv::Rect roi(ctx->offset_x, ctx->offset_y, ctx->new_width, ctx->new_height);
    resized_mat.copyTo(dst_mat(roi));
    
    if (ctx->debug_mode) {
        printf("[letterbox] 预处理完成: %dx%d -> %dx%d\n", 
               src_mat.cols, src_mat.rows, dst_mat.cols, dst_mat.rows);
    }
    
    return 0;
}

// 零拷贝letterbox预处理（直接写入NPU内存）
int zero_copy_letterbox_preprocess(const cv::Mat& src_mat, 
                                  zero_copy_letterbox_context_t* zc_ctx) {
    if (!zc_ctx || !zc_ctx->input_mem) {
        printf("[letterbox] 错误: 零拷贝上下文无效\n");
        return -1;
    }
    
    if (src_mat.empty()) {
        printf("[letterbox] 错误: 输入图像为空\n");
        return -1;
    }
    
    const letterbox_context_t* ctx = &zc_ctx->letterbox_ctx;
    
    // 验证输入图像尺寸
    if (src_mat.cols != ctx->src_width || src_mat.rows != ctx->src_height) {
        printf("[letterbox] 警告: 输入图像尺寸不匹配 实际:%dx%d 期望:%dx%d\n", 
               src_mat.cols, src_mat.rows, ctx->src_width, ctx->src_height);
    }
    
    // 直接在NPU内存上操作
    uint8_t* npu_ptr = (uint8_t*)zc_ctx->input_mem->virt_addr;
    int width_stride = zc_ctx->input_attr.w_stride;
    
    // 创建Mat包装NPU内存
    cv::Mat dst_mat(ctx->dst_height, width_stride, CV_8UC3, npu_ptr);
    dst_mat.setTo(cv::Scalar(114, 114, 114));  // 灰色背景
    
    // 缩放并复制到NPU内存的有效区域
    cv::Mat roi_mat = dst_mat(cv::Rect(ctx->offset_x, ctx->offset_y, 
                                      ctx->new_width, ctx->new_height));
    cv::resize(src_mat, roi_mat, cv::Size(ctx->new_width, ctx->new_height));
    
    if (ctx->debug_mode) {
        printf("[letterbox] 零拷贝预处理完成: %dx%d -> %dx%d (stride:%d)\n", 
               src_mat.cols, src_mat.rows, ctx->dst_width, ctx->dst_height, width_stride);
    }
    
    return 0;
}

// 坐标反变换：模型输出坐标 → 原始图像坐标
cv::Point2f letterbox_inverse_transform(float model_x, float model_y, 
                                       const letterbox_context_t* ctx) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return cv::Point2f(-1, -1);
    }
    
    // 步骤1: 减去letterbox偏移量
    float adjusted_x = model_x - ctx->offset_x;
    float adjusted_y = model_y - ctx->offset_y;
    
    // 步骤2: 除以缩放比例，得到原始图像坐标
    float original_x = adjusted_x / ctx->scale;
    float original_y = adjusted_y / ctx->scale;
    
    // 步骤3: 限制在原始图像范围内
    original_x = std::max(0.0f, std::min(original_x, (float)(ctx->src_width - 1)));
    original_y = std::max(0.0f, std::min(original_y, (float)(ctx->src_height - 1)));
    
    if (ctx->debug_mode) {
        printf("[letterbox] 坐标反变换: (%.1f,%.1f) -> (%.1f,%.1f)\n", 
               model_x, model_y, original_x, original_y);
    }
    
    return cv::Point2f(original_x, original_y);
}

// 批量坐标反变换
void letterbox_inverse_transform_batch(const cv::Point2f* model_points,
                                      cv::Point2f* original_points,
                                      int count,
                                      const letterbox_context_t* ctx) {
    if (!ctx || !model_points || !original_points) {
        printf("[letterbox] 错误: 参数指针为空\n");
        return;
    }
    
    for (int i = 0; i < count; i++) {
        original_points[i] = letterbox_inverse_transform(
            model_points[i].x, model_points[i].y, ctx);
    }
    
    if (ctx->debug_mode) {
        printf("[letterbox] 批量坐标反变换完成: %d个点\n", count);
    }
}

// 边界框反变换：模型输出边界框 → 原始图像边界框
cv::Rect letterbox_inverse_transform_bbox(const cv::Rect& model_box,
                                         const letterbox_context_t* ctx) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return cv::Rect(-1, -1, -1, -1);
    }
    
    // 转换左上角和右下角坐标
    cv::Point2f top_left = letterbox_inverse_transform(
        model_box.x, model_box.y, ctx);
    cv::Point2f bottom_right = letterbox_inverse_transform(
        model_box.x + model_box.width, model_box.y + model_box.height, ctx);
    
    // 构造原始图像边界框
    cv::Rect original_box(
        (int)top_left.x, (int)top_left.y,
        (int)(bottom_right.x - top_left.x), (int)(bottom_right.y - top_left.y));
    
    // 确保边界框有效
    original_box.x = std::max(0, original_box.x);
    original_box.y = std::max(0, original_box.y);
    original_box.width = std::max(1, std::min(original_box.width, 
                                             ctx->src_width - original_box.x));
    original_box.height = std::max(1, std::min(original_box.height, 
                                              ctx->src_height - original_box.y));
    
    if (ctx->debug_mode) {
        printf("[letterbox] 边界框反变换: (%d,%d,%d,%d) -> (%d,%d,%d,%d)\n", 
               model_box.x, model_box.y, model_box.width, model_box.height,
               original_box.x, original_box.y, original_box.width, original_box.height);
    }
    
    return original_box;
}

// 关键点反变换（专门用于姿态检测）
void letterbox_inverse_transform_keypoints(const float model_keypoints[][3],
                                          float original_keypoints[][2],
                                          int keypoint_count,
                                          const letterbox_context_t* ctx) {
    if (!ctx || !model_keypoints || !original_keypoints) {
        printf("[letterbox] 错误: 参数指针为空\n");
        return;
    }
    
    for (int i = 0; i < keypoint_count; i++) {
        // 只使用前两个坐标值，忽略置信度
        cv::Point2f original_point = letterbox_inverse_transform(
            model_keypoints[i][0], model_keypoints[i][1], ctx);
        
        original_keypoints[i][0] = original_point.x;
        original_keypoints[i][1] = original_point.y;
    }
    
    if (ctx->debug_mode) {
        printf("[letterbox] 关键点反变换完成: %d个关键点\n", keypoint_count);
    }
}

// 验证letterbox变换的准确性（调试用）
int validate_letterbox_transform(const letterbox_context_t* ctx) {
    if (!ctx) {
        printf("[letterbox] 错误: 上下文指针为空\n");
        return -1;
    }
    
    printf("=== Letterbox变换验证 ===\n");
    
    // 测试几个关键点的变换和反变换
    struct {
        float x, y;
        const char* name;
    } test_points[] = {
        {0, 0, "左上角"},
        {(float)ctx->src_width-1, 0, "右上角"},
        {0, (float)ctx->src_height-1, "左下角"},
        {(float)ctx->src_width-1, (float)ctx->src_height-1, "右下角"},
        {(float)ctx->src_width/2, (float)ctx->src_height/2, "中心点"}
    };
    
    int error_count = 0;
    for (int i = 0; i < 5; i++) {
        // 正变换：原始坐标 → 模型坐标
        float model_x = test_points[i].x * ctx->scale + ctx->offset_x;
        float model_y = test_points[i].y * ctx->scale + ctx->offset_y;
        
        // 反变换：模型坐标 → 原始坐标
        cv::Point2f recovered = letterbox_inverse_transform(model_x, model_y, ctx);
        
        // 计算误差
        float error_x = std::abs(recovered.x - test_points[i].x);
        float error_y = std::abs(recovered.y - test_points[i].y);
        float max_error = std::max(error_x, error_y);
        
        printf("%s: 原始(%.1f,%.1f) -> 模型(%.1f,%.1f) -> 恢复(%.1f,%.1f) 误差:%.3f\n",
               test_points[i].name, test_points[i].x, test_points[i].y,
               model_x, model_y, recovered.x, recovered.y, max_error);
        
        if (max_error > 1.0f) {  // 误差阈值1像素
            error_count++;
        }
    }
    
    printf("验证结果: %s (错误点数: %d/5)\n", 
           error_count == 0 ? "通过" : "失败", error_count);
    printf("===================\n");
    
    return error_count == 0 ? 0 : -1;
} 