/*-------------------------------------------
                篮球框和篮球检测器 - 更新版
                
用途: 使用Q_Rim_Basketball_724_JZ.rknn模型检测篮球框和篮球
输入: /dev/video2摄像头或指定摄像头设备
模型: 支持modern_dual_comparator.py验证的后处理逻辑
优化: NPU零拷贝，ROI位置分析，实时检测
功能: 专门负责篮筐和篮球的检测与ROI分析
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>

// RKNN headers
#include "rknn_api.h"
#include "common.h"

// RGA headers  
#include "im2d.h"
#include "im2d_type.h"
#include "im2d_single.h"
#include "RgaUtils.h"

// 后处理模块
#include "rim_basketball_postprocess.h"

// 全局变量，用于信号处理
bool g_running = true;

// 使用头文件中的定义，不需要重复定义

// 模型上下文结构体
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

// 零拷贝上下文
typedef struct {
    rknn_tensor_mem* input_mem;         // NPU输入内存
    rknn_tensor_mem* output_mems[10];   // 最多10个输出内存(兼容不同模型格式)
    rknn_tensor_attr input_attr;        // 输入属性
    rknn_tensor_attr output_attrs[10];  // 输出属性
    int model_width;                    // 640
    int model_height;                   // 640
    int model_channels;                 // 3
} rim_zero_copy_context_t;

// ROI分析结果
typedef struct {
    bool has_rim;
    bool has_basketball;
    cv::Rect rim_roi;
    cv::Rect basketball_roi;
    float rim_confidence;
    float basketball_confidence;
    cv::Point2f rim_center;
    cv::Point2f basketball_center;
    float distance_to_rim;  // 篮球到篮筐的距离
} ROIAnalysisResult;

// 类别名称 (根据模型实际训练顺序)
static const char* class_names[2] = {"basketball", "rim"};

// 信号处理函数
void sig_handler(int signo) {
    if (signo == SIGINT) {
        printf("接收到SIGINT信号，正在退出...\n");
        g_running = false;
    }
}

// 获取当前时间（微秒）
static int64_t getCurrentTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// 初始化模型
static int init_rim_basketball_model(const char* model_path, rknn_app_context_t* app_ctx) {
    int ret;
    
    printf("加载模型: %s\n", model_path);
    
    // 读取模型文件
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        printf("❌ 无法打开模型文件: %s\n", model_path);
        return -1;
    }
    
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* model_data = malloc(model_len);
    if (!model_data) {
        printf("❌ 分配内存失败\n");
        fclose(fp);
        return -1;
    }
    
    if (fread(model_data, 1, model_len, fp) != model_len) {
        printf("❌ 读取模型文件失败\n");
        free(model_data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    
    // 初始化RKNN上下文
    ret = rknn_init(&app_ctx->rknn_ctx, model_data, model_len, 0, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("❌ RKNN初始化失败! ret=%d\n", ret);
        return -1;
    }
    
    // 获取模型输入输出信息
    ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret != RKNN_SUCC) {
        printf("❌ 查询输入输出数量失败! ret=%d\n", ret);
        return -1;
    }
    
    printf("模型输入数量: %d, 输出数量: %d\n", app_ctx->io_num.n_input, app_ctx->io_num.n_output);
    
    // 获取输入属性
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(app_ctx->input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("❌ 查询输入属性失败! ret=%d\n", ret);
            return -1;
        }
    }
    
    // 获取输出属性
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(app_ctx->output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("❌ 查询输出属性失败! ret=%d\n", ret);
            return -1;
        }
    }
    
    // 调试：打印输入tensor的完整维度信息
    printf("输入tensor维度调试:\n");
    printf("- 维度数量: %d\n", app_ctx->input_attrs[0].n_dims);
    for (int i = 0; i < app_ctx->input_attrs[0].n_dims; i++) {
        printf("- dims[%d] = %d\n", i, app_ctx->input_attrs[0].dims[i]);
    }
    printf("- 数据格式: %s\n", 
           app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW ? "NCHW" : 
           (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC ? "NHWC" : "其他"));
    
    // 根据数据格式正确解析维度
    if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // 格式为 [N,C,H,W]
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
    } else if (app_ctx->input_attrs[0].fmt == RKNN_TENSOR_NHWC) {
        // 格式为 [N,H,W,C]
        app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[3];
    } else {
        // 未知格式，使用默认顺序
        printf("⚠️  未知的tensor格式，使用默认解析\n");
        app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
    }
    
    printf("✓ 模型信息: C=%d, H=%d, W=%d\n", 
           app_ctx->model_channel, app_ctx->model_height, app_ctx->model_width);
    
    return 0;
}

// 释放模型
static int release_rim_basketball_model(rknn_app_context_t* app_ctx) {
    if (app_ctx->input_attrs) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    
    if (app_ctx->output_attrs) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    
    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    
    return 0;
}

// 初始化零拷贝内存
static int init_rim_zero_copy_mem(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx) {
    int ret;
    
    // 设置输入属性
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    // 创建输入内存
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("❌ 创建输入零拷贝内存失败！\n");
        return -1;
    }
    
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("❌ 设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }

    // 创建输出内存
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->output_attrs[i].size_with_stride);
        if (!zc_ctx->output_mems[i]) {
            printf("❌ 创建输出零拷贝内存[%d]失败！\n", i);
            return -1;
        }
        
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i], &zc_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("❌ 设置输出零拷贝内存[%d]失败! ret=%d\n", i, ret);
            return -1;
        }
    }
    
    printf("✓ 零拷贝内存初始化成功\n");
    return 0;
}

// 释放零拷贝内存
static void release_rim_zero_copy_mem(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx) {
    if (zc_ctx->input_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->input_mem);
    }
    
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (zc_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i]);
        }
    }
}

// 简化的letterbox resize到NPU内存
static int letterbox_resize_to_npu(const cv::Mat& src, rim_zero_copy_context_t* zc_ctx, float* scale, int* x_pad, int* y_pad) {
    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = zc_ctx->model_width;
    int dst_h = zc_ctx->model_height;
    
    // 计算缩放比例
    *scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * (*scale));
    int new_h = (int)(src_h * (*scale));
    
    *x_pad = (dst_w - new_w) / 2;
    *y_pad = (dst_h - new_h) / 2;
    
    // 创建指向NPU内存的Mat
    cv::Mat npu_mat(dst_h, dst_w, CV_8UC3, zc_ctx->input_mem->virt_addr);
    npu_mat.setTo(cv::Scalar(114, 114, 114)); // 灰色填充 (RGB顺序)
    
    // 关键修复：BGR转RGB
    cv::Mat src_rgb;
    cv::cvtColor(src, src_rgb, cv::COLOR_BGR2RGB);
    
    // resize原图到目标尺寸
    cv::Mat resized;
    cv::resize(src_rgb, resized, cv::Size(new_w, new_h));
    
    // 拷贝到NPU内存的中心位置
    cv::Rect roi(*x_pad, *y_pad, new_w, new_h);
    resized.copyTo(npu_mat(roi));
    
    // 调试：检查预处理后的首像素
    uint8_t* data_ptr = (uint8_t*)npu_mat.data;
    static bool first_debug = true;
    if (first_debug) {
        printf("预处理后首像素RGB: [%d, %d, %d] (应该是RGB顺序)\n", 
               data_ptr[0], data_ptr[1], data_ptr[2]);
        first_debug = false;
    }
    
    return 0;
}

// 基于modern_dual_comparator.py的后处理逻辑
static int postprocess_rim_basketball(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx, 
                                    float scale, int x_pad, int y_pad, int orig_w, int orig_h,
                                    float conf_threshold, float nms_threshold,
                                    RimBasketballDetectionResult* result) {
    
    // 准备RKNN输出数据
    rknn_output outputs[10]; // 最多支持10个输出
    memset(outputs, 0, sizeof(outputs));
    
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 1;
        outputs[i].buf = zc_ctx->output_mems[i]->virt_addr;
        outputs[i].size = zc_ctx->output_attrs[i].size;
    }
    
    // 调用移植的后处理函数
    int ret = process_rim_basketball_outputs(outputs, zc_ctx->output_attrs, 
                                           conf_threshold, nms_threshold, result);
    
    if (ret != 0) {
        printf("❌ 后处理失败\n");
        return ret;
    }
    
    // 坐标映射：将模型输出坐标转换回原始图像坐标
    for (int i = 0; i < result->count; i++) {
        RimBasketballDetection* det = &result->detections[i];
        
        // 将center坐标转换为corner坐标
        float x1 = det->x - det->w / 2.0f;
        float y1 = det->y - det->h / 2.0f;
        float x2 = det->x + det->w / 2.0f;
        float y2 = det->y + det->h / 2.0f;
        
        // 逆letterbox映射：640x640 → 原始图像尺寸
        x1 = (x1 - x_pad) / scale;
        y1 = (y1 - y_pad) / scale;
        x2 = (x2 - x_pad) / scale;
        y2 = (y2 - y_pad) / scale;
        
        // 限制到图像边界内
        x1 = fmaxf(0.0f, fminf(x1, orig_w - 1));
        y1 = fmaxf(0.0f, fminf(y1, orig_h - 1));
        x2 = fmaxf(0.0f, fminf(x2, orig_w - 1));
        y2 = fmaxf(0.0f, fminf(y2, orig_h - 1));
        
        // 转换回center+size格式
        det->x = (x1 + x2) / 2.0f;
        det->y = (y1 + y2) / 2.0f;
        det->w = x2 - x1;
        det->h = y2 - y1;
    }
    
    return 0;
}

// ROI分析
static void analyze_roi(const RimBasketballDetectionResult* detections, ROIAnalysisResult* roi_result) {
    memset(roi_result, 0, sizeof(ROIAnalysisResult));
    
    float best_rim_conf = 0.0f;
    float best_basketball_conf = 0.0f;
    
    for (int i = 0; i < detections->count; i++) {
        const RimBasketballDetection* det = &detections->detections[i];
        
        if (det->class_id == RIM_CLASS_ID && det->confidence > best_rim_conf) {
            roi_result->has_rim = true;
            best_rim_conf = det->confidence;
            roi_result->rim_confidence = det->confidence;
            roi_result->rim_center = cv::Point2f(det->x, det->y);
            roi_result->rim_roi = cv::Rect(det->x - det->w/2, det->y - det->h/2, det->w, det->h);
        }
        
        if (det->class_id == BASKETBALL_CLASS_ID && det->confidence > best_basketball_conf) {
            roi_result->has_basketball = true;
            best_basketball_conf = det->confidence;
            roi_result->basketball_confidence = det->confidence;
            roi_result->basketball_center = cv::Point2f(det->x, det->y);
            roi_result->basketball_roi = cv::Rect(det->x - det->w/2, det->y - det->h/2, det->w, det->h);
        }
    }
    
    // 计算篮球到篮筐的距离
    if (roi_result->has_rim && roi_result->has_basketball) {
        float dx = roi_result->basketball_center.x - roi_result->rim_center.x;
        float dy = roi_result->basketball_center.y - roi_result->rim_center.y;
        roi_result->distance_to_rim = sqrt(dx*dx + dy*dy);
    }
}

// 绘制检测结果
static void draw_detections(cv::Mat& img, const RimBasketballDetectionResult* detections, 
                          const ROIAnalysisResult* roi_result) {
    for (int i = 0; i < detections->count; i++) {
        const RimBasketballDetection* det = &detections->detections[i];
        
        // 选择颜色
        cv::Scalar color;
        if (det->class_id == RIM_CLASS_ID) {
            color = cv::Scalar(0, 255, 0); // 绿色 - 篮筐
        } else {
            color = cv::Scalar(0, 165, 255); // 橙色 - 篮球
        }
        
        // 绘制边界框
        cv::Rect rect(det->x - det->w/2, det->y - det->h/2, det->w, det->h);
        cv::rectangle(img, rect, color, 3);
        
        // 绘制标签
        char label[100];
        snprintf(label, sizeof(label), "%s %.2f", det->class_name, det->confidence);
        
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        cv::Point label_origin(rect.x, rect.y - 10);
        
        cv::rectangle(img, cv::Point(label_origin.x, label_origin.y - label_size.height - baseline),
                     cv::Point(label_origin.x + label_size.width, label_origin.y + baseline), color, -1);
        cv::putText(img, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    
    // 绘制ROI分析信息
    if (roi_result->has_rim && roi_result->has_basketball) {
        // 绘制连线
        cv::line(img, roi_result->rim_center, roi_result->basketball_center, cv::Scalar(255, 255, 0), 2);
        
        // 显示距离
        char dist_str[100];
        snprintf(dist_str, sizeof(dist_str), "Distance: %.1f px", roi_result->distance_to_rim);
        cv::Point text_pos((roi_result->rim_center.x + roi_result->basketball_center.x) / 2,
                          (roi_result->rim_center.y + roi_result->basketball_center.y) / 2 - 10);
        cv::putText(img, dist_str, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }
}

int main(int argc, char **argv) {
    int ret;
    rknn_app_context_t rknn_app_ctx;
    rim_zero_copy_context_t zero_copy_ctx = {};
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    if (argc < 2) {
        printf("用法: %s <rknn模型路径> [输入源]\n", argv[0]);
        printf("示例: %s ../models/Q_Rim_Basketball_724_JZ.rknn  # 使用默认摄像头\n", argv[0]);
        printf("示例: %s ../models/Q_Rim_Basketball_724_JZ.rknn 0  # 使用摄像头设备0\n", argv[0]);
        printf("示例: %s ../models/Q_Rim_Basketball_724_JZ.rknn ../rim_basketball.avi  # 使用视频文件\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* input_source = (argc > 2) ? argv[2] : "2";  // 默认摄像头2
    
    // 判断输入源类型：数字为摄像头ID，文件路径为视频文件
    bool is_video_file = false;
    int camera_id = 2;
    
    if (argc > 2) {
        // 检查是否为数字（摄像头ID）
        char* endptr;
        long id = strtol(input_source, &endptr, 10);
        if (*endptr == '\0') {
            // 纯数字，表示摄像头ID
            camera_id = (int)id;
            is_video_file = false;
        } else {
            // 包含非数字字符，表示视频文件路径
            is_video_file = true;
        }
    }
    
    // 设置信号处理
    signal(SIGINT, sig_handler);
    
    printf("========================================\n");
    printf("    篮筐篮球检测系统 v2.0\n");
    printf("========================================\n");
    printf("模型文件: %s\n", model_path);
    if (is_video_file) {
        printf("视频文件: %s\n", input_source);
    } else {
        printf("摄像头设备: /dev/video%d\n", camera_id);
    }
    printf("按键说明:\n");
    printf("  [ESC] - 退出程序\n");
    printf("  [S]   - 截图保存\n");
    printf("  [空格] - 暂停/继续（仅视频文件）\n");
    printf("========================================\n");
    
    // 初始化模型
    ret = init_rim_basketball_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("❌ 初始化模型失败！\n");
        return -1;
    }
    
    // 预先声明变量，避免goto跨越初始化
    cv::Mat frame;
    int frame_count = 0;
    int screenshot_count = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    float conf_threshold = 0.25f;  // 提高置信度阈值过滤低质量检测
    float nms_threshold = 0.1f;   // 严格的NMS阈值，过滤重叠框
    cv::VideoCapture cap;
    bool paused = false;  // 视频暂停状态
    
    // 初始化零拷贝内存
    ret = init_rim_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("❌ 初始化零拷贝内存失败！\n");
        goto exit;
    }
    
    // 打开输入源（摄像头或视频文件）
    if (is_video_file) {
        cap.open(input_source);
        if (!cap.isOpened()) {
            printf("❌ 无法打开视频文件: %s\n", input_source);
            goto exit;
        }
        printf("✓ 视频文件打开成功\n");
        printf("视频分辨率: %.0fx%.0f @ %.0f FPS\n", 
               cap.get(cv::CAP_PROP_FRAME_WIDTH), 
               cap.get(cv::CAP_PROP_FRAME_HEIGHT),
               cap.get(cv::CAP_PROP_FPS));
        printf("总帧数: %.0f\n", cap.get(cv::CAP_PROP_FRAME_COUNT));
    } else {
        // 打开摄像头 - 强制使用V4L2后端
        cap.open(camera_id, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            printf("❌ 无法打开摄像头 /dev/video%d！\n", camera_id);
            goto exit;
        }
        
        // 设置摄像头参数
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        
        printf("✓ 摄像头打开成功\n");
        printf("摄像头分辨率: %.0fx%.0f @ %.0f FPS\n", 
               cap.get(cv::CAP_PROP_FRAME_WIDTH), 
               cap.get(cv::CAP_PROP_FRAME_HEIGHT),
               cap.get(cv::CAP_PROP_FPS));
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    printf("开始处理视频流...\n");
    printf("检测阈值: conf=%.2f, nms=%.2f\n", conf_threshold, nms_threshold);
    
    while (g_running) {
        // 如果暂停且是视频文件，跳过处理但继续显示
        if (paused && is_video_file) {
            int key = cv::waitKey(30) & 0xFF;
            if (key == 27) { // ESC
                break;
            } else if (key == 32) { // 空格键
                paused = !paused;
                printf("继续播放\n");
            }
            continue;
        }
        
        // 读取帧
        if (!cap.read(frame)) {
            if (is_video_file) {
                printf("视频播放完成\n");
                break;
            } else {
                printf("❌ 读取帧失败！\n");
                break;
            }
        }
        
        frame_count++;
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        
        // 推理时间统计
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // Letterbox预处理到NPU内存
        float scale;
        int x_pad, y_pad;
        ret = letterbox_resize_to_npu(frame, &zero_copy_ctx, &scale, &x_pad, &y_pad);
        if (ret != 0) {
            printf("❌ 预处理失败！\n");
            continue;
        }
        
        // 调试：验证预处理结果
        if (frame_count == 1) {
            printf("预处理调试信息:\n");
            printf("- 原始图像尺寸: %dx%d\n", frame.cols, frame.rows);
            printf("- 缩放比例: %.6f\n", scale);
            printf("- 填充: x_pad=%d, y_pad=%d\n", x_pad, y_pad);
            
            // 检查NPU输入数据的前几个像素
            uint8_t* input_data = (uint8_t*)zero_copy_ctx.input_mem->virt_addr;
            printf("NPU输入数据前16个像素: ");
            for (int i = 0; i < 16; i++) {
                printf("%d ", input_data[i]);
            }
            printf("\n");
            
            // 检查是否是BGR还是RGB
            printf("首像素RGB值: [%d, %d, %d]\n", input_data[0], input_data[1], input_data[2]);
        }
        
        // NPU推理
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0) {
            printf("❌ 推理失败! ret=%d\n", ret);
            continue;
        }
        
        // 后处理
        RimBasketballDetectionResult detections;
        ret = postprocess_rim_basketball(&rknn_app_ctx, &zero_copy_ctx, scale, x_pad, y_pad, 
                                       orig_w, orig_h, conf_threshold, nms_threshold, &detections);
        if (ret != 0) {
            printf("❌ 后处理失败！\n");
            continue;
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        float inference_time = std::chrono::duration<float, std::milli>(inference_end - inference_start).count();
        
        // ROI分析
        ROIAnalysisResult roi_result;
        analyze_roi(&detections, &roi_result);
        
        // 绘制检测结果
        draw_detections(frame, &detections, &roi_result);
        
        // 显示性能信息和ROI状态
        char status_str[200];
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(current_time - start_time).count();
        float fps = frame_count / elapsed;
        
        snprintf(status_str, sizeof(status_str), 
                "FPS: %.1f | Inference: %.1fms | Rim: %s | Ball: %s", 
                fps, inference_time, 
                roi_result.has_rim ? "YES" : "NO",
                roi_result.has_basketball ? "YES" : "NO");
        cv::putText(frame, status_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 显示ROI详细信息
        if (roi_result.has_rim || roi_result.has_basketball) {
            char roi_info[200];
            snprintf(roi_info, sizeof(roi_info), "Rim: %.2f | Ball: %.2f | Dist: %.1f", 
                    roi_result.rim_confidence, roi_result.basketball_confidence, roi_result.distance_to_rim);
            cv::putText(frame, roi_info, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }
        
        // 显示帧
        cv::imshow("Rim & Basketball Detection", frame);
        
        // 处理按键
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            break;
        } else if (key == 's' || key == 'S') {
            char filename[100];
            snprintf(filename, sizeof(filename), "screenshot_%04d.jpg", ++screenshot_count);
            cv::imwrite(filename, frame);
            printf("💾 截图保存: %s\n", filename);
        } else if (key == 32 && is_video_file) { // 空格键，仅视频文件支持
            paused = !paused;
            printf("%s\n", paused ? "暂停播放" : "继续播放");
        }
    }
    
exit:
    printf("正在清理资源...\n");
    
    // 释放零拷贝内存
    release_rim_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    
    // 释放模型
    ret = release_rim_basketball_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("❌ 释放模型失败！\n");
    }
    
    cv::destroyAllWindows();
    printf("程序退出\n");
    return 0;
}