/*-------------------------------------------
                篮球框和篮球检测器
                
用途: 使用Q_Rim_Basketball_8n_4090_500E.rknn模型检测篮球框和篮球
输入: /dev/video2摄像头，1920x1200@90fps
模型: 9输出结构，3个尺度(80x80,40x40,20x20)，2类检测(rim,basketball)
优化: NPU零拷贝，INT8量化，90fps实时性能
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

// 9输出零拷贝上下文
typedef struct {
    rknn_tensor_mem* input_mem;         // NPU输入内存
    rknn_tensor_mem* output_mems[9];    // 9个输出内存
    rknn_tensor_attr input_attr;        // 输入属性
    rknn_tensor_attr output_attrs[9];   // 9个输出属性
    int model_width;                    // 640
    int model_height;                   // 640
    int model_channels;                 // 3
} rim_zero_copy_context_t;

// 注意：检测结果结构体定义在rim_basketball_postprocess.h中

// 类别名称
const char* class_names[2] = {"rim", "basketball"};

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
        printf("❌ 内存分配失败\n");
        fclose(fp);
        return -1;
    }
    
    fread(model_data, 1, model_len, fp);
    fclose(fp);
    
    // 初始化RKNN
    ret = rknn_init(&app_ctx->rknn_ctx, model_data, model_len, 0, NULL);
    free(model_data);
    
    if (ret < 0) {
        printf("❌ rknn_init失败! ret=%d\n", ret);
        return -1;
    }
    
    // 获取输入输出信息
    ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret < 0) {
        printf("❌ rknn_query io_num失败! ret=%d\n", ret);
        return -1;
    }
    
    printf("模型输入数量: %d\n", app_ctx->io_num.n_input);
    printf("模型输出数量: %d\n", app_ctx->io_num.n_output);
    
    // 验证输出数量
    if (app_ctx->io_num.n_output != 9) {
        printf("❌ 模型输出数量错误! 期望9个，实际%d个\n", app_ctx->io_num.n_output);
        return -1;
    }
    
    // 分配属性数组
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    
    // 查询输入属性
    memset(app_ctx->input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_INPUT_ATTR, &app_ctx->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("❌ rknn_query输入属性失败! ret=%d\n", ret);
            return -1;
        }
    }
    
    // 查询输出属性
    memset(app_ctx->output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        ret = rknn_query(app_ctx->rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &app_ctx->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("❌ rknn_query输出属性失败! ret=%d\n", ret);
            return -1;
        }
    }
    
    // 设置模型参数
    app_ctx->model_channel = app_ctx->input_attrs[0].dims[1];
    app_ctx->model_height = app_ctx->input_attrs[0].dims[2];  
    app_ctx->model_width = app_ctx->input_attrs[0].dims[3];
    app_ctx->is_quant = (app_ctx->input_attrs[0].type == RKNN_TENSOR_INT8);
    
    printf("✅ 模型加载成功:\n");
    printf("   输入尺寸: %dx%dx%d\n", app_ctx->model_width, app_ctx->model_height, app_ctx->model_channel);
    printf("   量化模式: %s\n", app_ctx->is_quant ? "INT8" : "FP32");
    
    // 设置NPU核心为NPU2
    rknn_core_mask core_mask = RKNN_NPU_CORE_2;
    ret = rknn_set_core_mask(app_ctx->rknn_ctx, core_mask);
    if (ret != 0) {
        printf("⚠️ rknn_set_core_mask失败! ret=%d，将使用默认NPU核心\n", ret);
    } else {
        printf("✅ NPU核心绑定到NPU2成功\n");
    }
    
    // 打印9个输出的详细信息
    printf("   输出结构:\n");
    for (int i = 0; i < 9; i++) {
        rknn_tensor_attr* attr = &app_ctx->output_attrs[i];
        printf("     output%d: [%d,%d,%d,%d] %s\n", i,
               attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
               attr->type == RKNN_TENSOR_INT8 ? "int8" : "float");
    }
    
    // 验证9输出模型的预期结构（3个尺度 x 3个输出）
    printf("   验证模型结构:\n");
    for (int scale = 0; scale < 3; scale++) {
        int feature_idx = scale * 3 + 0;  // 特征输出 (64通道)
        int class_idx = scale * 3 + 1;    // 分类输出 (2通道)  
        int conf_idx = scale * 3 + 2;     // 置信度输出 (1通道)
        
        printf("     尺度%d: 特征[%d通道] 分类[%d通道] 置信度[%d通道]\n", 
               scale,
               app_ctx->output_attrs[feature_idx].dims[1],
               app_ctx->output_attrs[class_idx].dims[1], 
               app_ctx->output_attrs[conf_idx].dims[1]);
               
        // 检查预期通道数
        if (app_ctx->output_attrs[feature_idx].dims[1] < 4) {
            printf("❌ 错误: 尺度%d特征输出通道数为%d，至少需要4个通道\n", 
                   scale, app_ctx->output_attrs[feature_idx].dims[1]);
            return -1;
        }
        if (app_ctx->output_attrs[feature_idx].dims[1] != 64) {
            printf("⚠️ 注意: 尺度%d特征输出通道数为%d，通常期望64，但仍可处理\n", 
                   scale, app_ctx->output_attrs[feature_idx].dims[1]);
        }
        if (app_ctx->output_attrs[class_idx].dims[1] != 2) {
            printf("❌ 错误: 尺度%d分类输出通道数为%d，期望2\n", 
                   scale, app_ctx->output_attrs[class_idx].dims[1]);
            return -1;
        }
        if (app_ctx->output_attrs[conf_idx].dims[1] != 1) {
            printf("❌ 错误: 尺度%d置信度输出通道数为%d，期望1\n", 
                   scale, app_ctx->output_attrs[conf_idx].dims[1]);
            return -1;
        }
    }
    
    return 0;
}

// 释放模型
static void release_rim_basketball_model(rknn_app_context_t* app_ctx) {
    if (app_ctx->rknn_ctx) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
}

// 初始化零拷贝内存
static int init_zero_copy_mem(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx) {
    int ret;
    
    // 设置输入属性
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    // 创建NPU输入内存
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("❌ 创建输入零拷贝内存失败!\n");
        return -1;
    }
    
    // 绑定输入内存
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("❌ 设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }
    
    // 设置输出属性（暂不使用输出零拷贝）
    for (int i = 0; i < 9; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = NULL;
    }
    
    printf("✅ 零拷贝内存初始化成功\n");
    return 0;
}

// 释放零拷贝内存
static void release_zero_copy_mem(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx) {
    if (zc_ctx->input_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->input_mem);
        zc_ctx->input_mem = NULL;
    }
    
    for (int i = 0; i < 9; i++) {
        if (zc_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i]);
            zc_ctx->output_mems[i] = NULL;
        }
    }
}

// 零拷贝letterbox预处理
static int optimized_letterbox_to_npu(cv::Mat& src_mat, rim_zero_copy_context_t* zc_ctx, bool debug_save = false) {
    int dst_width = zc_ctx->model_width;
    int dst_height = zc_ctx->model_height;
    int src_width = src_mat.cols;
    int src_height = src_mat.rows;
    
    // 计算letterbox缩放参数
    float scale_w = (float)dst_width / src_width;
    float scale_h = (float)dst_height / src_height;
    float scale = std::min(scale_w, scale_h);
    int new_width = (int)(src_width * scale);
    int new_height = (int)(src_height * scale);
    int offset_x = (dst_width - new_width) / 2;
    int offset_y = (dst_height - new_height) / 2;
    
    // 直接在NPU内存上操作
    uint8_t* npu_ptr = (uint8_t*)zc_ctx->input_mem->virt_addr;
    int width_stride = zc_ctx->input_attr.w_stride;
    cv::Mat dst_mat(dst_height, width_stride, CV_8UC3, npu_ptr);
    dst_mat.setTo(cv::Scalar(114, 114, 114));  // 灰色填充
    
    // 只在有效区域进行缩放
    cv::Mat roi_mat = dst_mat(cv::Rect(offset_x, offset_y, new_width, new_height));
    cv::resize(src_mat, roi_mat, cv::Size(new_width, new_height));
    
    // 调试：保存预处理后的图像
    if (debug_save) {
        cv::Mat debug_img(dst_height, dst_width, CV_8UC3, npu_ptr);
        cv::imwrite("debug_letterbox.jpg", debug_img);
        printf("[PREPROCESS] Saved debug image: debug_letterbox.jpg\n");
        printf("[PREPROCESS] Letterbox: scale=%.3f\n", scale);
    }
    
    return 0;
}

// 零拷贝推理和后处理
static int zero_copy_inference_and_postprocess(
    rknn_app_context_t* app_ctx, 
    rim_zero_copy_context_t* zc_ctx, 
    RimBasketballDetectionResult* result,
    float conf_threshold = 0.5f,
    float nms_threshold = 0.4f
) {
    int ret;
    
    // 设置输入
    rknn_input input;
    input.index = 0;
    input.buf = zc_ctx->input_mem->virt_addr;
    input.size = zc_ctx->input_attr.size_with_stride;
    input.pass_through = 1;  // 直通模式
    input.type = zc_ctx->input_attr.type;
    input.fmt = zc_ctx->input_attr.fmt;
    
    // 同步内存到NPU
    rknn_mem_sync(app_ctx->rknn_ctx, zc_ctx->input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    
    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
    if (ret < 0) {
        printf("❌ rknn_inputs_set失败! ret=%d\n", ret);
        return -1;
    }
    
    // NPU推理
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0) {
        printf("❌ rknn_run失败! ret=%d\n", ret);
        return -1;
    }
    
    // 获取9个输出
    rknn_output outputs[9];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 9; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 0;  // 保持INT8格式
    }
    
    ret = rknn_outputs_get(app_ctx->rknn_ctx, 9, outputs, NULL);
    if (ret < 0) {
        printf("❌ rknn_outputs_get失败! ret=%d\n", ret);
        return -1;
    }
    
    // 调用后处理
    ret = process_rim_basketball_outputs(outputs, zc_ctx->output_attrs, 
                                       conf_threshold, nms_threshold, result);
    
    // 释放输出
    rknn_outputs_release(app_ctx->rknn_ctx, 9, outputs);
    
    if (ret < 0) {
        printf("❌ 后处理失败! ret=%d\n", ret);
        return -1;
    }
    
    // 添加调试信息
    if (result->count > 0) {
        printf("[推理成功] 检测到 %d 个目标\n", result->count);
        for (int i = 0; i < result->count && i < 3; i++) {  // 只打印前3个目标
            printf("  目标%d: %s, 置信度=%.3f, 位置=(%.1f,%.1f,%.1f,%.1f)\n", 
                   i, result->detections[i].class_name, result->detections[i].confidence,
                   result->detections[i].x, result->detections[i].y, 
                   result->detections[i].w, result->detections[i].h);
        }
    }
    
    return 0;
}

// 绘制检测结果
static void draw_detections(cv::Mat& image, const RimBasketballDetectionResult* result, 
                          float inv_scale, int offset_x, int offset_y) {
    // 只在有检测结果时输出简要信息
    if (result->count > 0) {
        printf("[DETECT] Found %d objects\n", result->count);
    }
    
    for (int i = 0; i < result->count; i++) {
        const RimBasketballDetection* det = &result->detections[i];
        
        // 转换坐标：模型输出(640x640) -> 原始图像坐标
        // 步骤1: 从中心坐标转换为左上角坐标 (模型坐标系)
        float model_x1 = det->x - det->w/2;
        float model_y1 = det->y - det->h/2;
        float model_x2 = det->x + det->w/2;
        float model_y2 = det->y + det->h/2;
        
        // 步骤2: letterbox逆变换到原始图像坐标
        float x1 = (model_x1 - offset_x) * inv_scale;
        float y1 = (model_y1 - offset_y) * inv_scale;
        float x2 = (model_x2 - offset_x) * inv_scale;
        float y2 = (model_y2 - offset_y) * inv_scale;
        
        // 限制在图像范围内
        x1 = std::max(0.0f, std::min(x1, (float)(image.cols - 1)));
        y1 = std::max(0.0f, std::min(y1, (float)(image.rows - 1)));
        x2 = std::max(0.0f, std::min(x2, (float)(image.cols - 1)));
        y2 = std::max(0.0f, std::min(y2, (float)(image.rows - 1)));
        
        // 选择颜色：篮球框用红色，篮球用蓝色
        cv::Scalar color;
        if (det->class_id == RIM_CLASS_ID) {
            color = cv::Scalar(0, 0, 255);  // 红色 - 篮球框
        } else {
            color = cv::Scalar(255, 0, 0);  // 蓝色 - 篮球
        }
        
        // 确保坐标有效
        if (x2 > x1 && y2 > y1 && x1 >= 0 && y1 >= 0) {
            // 绘制边界框
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), color, 3);
            printf("[DEBUG] Drew %s at (%.1f,%.1f)-(%.1f,%.1f), conf=%.3f\n", 
                   det->class_name, x1, y1, x2, y2, det->confidence);
        } else {
            printf("[DEBUG] Invalid coordinates for %s, skipping rectangle\n", det->class_name);
        }
        
        // 绘制标签
        char label[64];
        sprintf(label, "%s: %.2f", det->class_name, det->confidence);
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        
        // 绘制标签背景
        cv::rectangle(image, 
                     cv::Point(x1, y1 - text_size.height - 10),
                     cv::Point(x1 + text_size.width, y1),
                     color, -1);
        
        // 绘制标签文字
        cv::putText(image, label, cv::Point(x1, y1 - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    }
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    printf("=== 篮球框和篮球检测器 ===\n");
    printf("平台: Rock5C 8GB, CPU: aarch64, NPU: RK3588\n");
    printf("摄像头: /dev/video2, 1920x1080@90fps\n");
    printf("模型: Q_Rim_Basketball_8n_4090_500E.rknn (9输出)\n\n");
    
    if (argc != 2) {
        printf("用法: %s <model_path>\n", argv[0]);
        printf("示例: %s ./models/Q_Rim_Basketball_8n_4090_500E.rknn\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    printf("模型路径: %s\n\n", model_path);
    
    int ret;
    rknn_app_context_t rknn_app_ctx;
    rim_zero_copy_context_t zero_copy_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&zero_copy_ctx, 0, sizeof(rim_zero_copy_context_t));

    // 注册信号处理函数
    signal(SIGINT, sig_handler);

    // 系统性能优化
    printf("=== 系统性能优化 ===\n");
    
    // 1. 设置CPU亲和性到高性能核心
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(4, &cpuset);  // 绑定到大核
    CPU_SET(5, &cpuset);
    CPU_SET(6, &cpuset);
    CPU_SET(7, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    printf("✅ CPU亲和性绑定到高性能核心(4-7)\n");
    
    // 2. 设置OpenCV线程数和优化
    cv::setNumThreads(4);
    cv::setUseOptimized(true);
    printf("✅ OpenCV优化设置完成\n");
    
    // 3. NPU频率已优化（ondemand模式，1000MHz）
    printf("✅ NPU频率: ondemand模式，当前1000MHz\n");
    printf("========================\n\n");

    // 初始化模型
    ret = init_rim_basketball_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("❌ 模型初始化失败! ret=%d\n", ret);
        return -1;
    }

    // 初始化零拷贝内存
    ret = init_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("❌ 零拷贝内存初始化失败!\n");
        release_rim_basketball_model(&rknn_app_ctx);
        return -1;
    }

    // 打开摄像头 /dev/video2
    printf("=== 初始化摄像头 ===\n");
    cv::VideoCapture cap(2, cv::CAP_V4L2);  // /dev/video2
    if (!cap.isOpened()) {
        printf("❌ 摄像头/dev/video2打开失败\n");
        release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
        release_rim_basketball_model(&rknn_app_ctx);
        return -1;
    }

    // 设置摄像头参数: 1920x1080@90fps (标准1080p)
    printf("设置摄像头参数...\n");
    
    // 1. 设置格式和分辨率
    bool mjpg_ok = cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    bool width_ok = cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    bool height_ok = cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    
    printf("MJPEG设置: %s\n", mjpg_ok ? "成功" : "失败");
    printf("宽度设置: %s\n", width_ok ? "成功" : "失败");
    printf("高度设置: %s\n", height_ok ? "成功" : "失败");
    
    // 2. 尝试多种帧率设置方法
    bool fps_ok1 = cap.set(cv::CAP_PROP_FPS, 90);
    usleep(100*1000);  // 延时100ms让设置生效
    bool fps_ok2 = cap.set(cv::CAP_PROP_FPS, 90);  // 重复设置
    
    printf("FPS设置(第1次): %s\n", fps_ok1 ? "成功" : "失败");
    printf("FPS设置(第2次): %s\n", fps_ok2 ? "成功" : "失败");
    
    // 3. 尝试设置缓冲区大小
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // 最小缓冲，减少延迟
    
    usleep(200*1000);  // 延时200ms

    // 获取实际摄像头参数
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    int actual_buffer = cap.get(cv::CAP_PROP_BUFFERSIZE);
    int fourcc = cap.get(cv::CAP_PROP_FOURCC);
    
    printf("摄像头实际参数:\n");
    printf("  分辨率: %dx%d\n", actual_width, actual_height);
    printf("  帧率: %.1f FPS (理论采集时间: %.1fms)\n", actual_fps, 1000.0/actual_fps);
    printf("  缓冲区: %d\n", actual_buffer);
    printf("  格式: %c%c%c%c\n", fourcc&0xFF, (fourcc>>8)&0xFF, (fourcc>>16)&0xFF, (fourcc>>24)&0xFF);
    
    // 如果FPS设置失败，尝试替代方案
    if (actual_fps < 85.0) {
        printf("⚠️ FPS设置可能失败，尝试v4l2直接设置...\n");
        // 可以尝试使用v4l2-ctl命令
        system("v4l2-ctl -d /dev/video2 --set-parm=90 2>/dev/null || echo '无法使用v4l2-ctl'");
        usleep(100*1000);
        actual_fps = cap.get(cv::CAP_PROP_FPS);
        printf("  重新检查FPS: %.1f\n", actual_fps);
    }

    // 验证摄像头工作正常
    cv::Mat test_frame;
    if (!cap.read(test_frame)) {
        printf("❌ 无法从摄像头读取测试帧!\n");
        return -1;
    }
    printf("✅ 摄像头工作正常，采集帧尺寸: %dx%d\n", test_frame.cols, test_frame.rows);
    
    // 创建显示窗口
    const char* WINDOW_NAME = "Rim Basketball Detector";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 960, 600);  // 显示窗口尺寸
    
    // 性能统计变量
    int frame_count = 0;
    int processed_count = 0;  // 实际处理的帧数
    int64_t total_time = 0;
    int64_t start_time_overall = getCurrentTimeUs();
    
    // 跳帧策略：为了达到90fps，可能需要跳帧处理
    int frame_skip = 1;  // 每N帧处理一次，初始为1（每帧都处理）

    printf("\n=== 开始实时检测 ===\n");
    printf("按ESC键退出程序\n\n");

    // 最后一次检测结果（用于跳帧时复用）
    RimBasketballDetectionResult last_detection_result;
    memset(&last_detection_result, 0, sizeof(last_detection_result));
    
    // 主循环
    while (g_running) {
        int64_t t0 = getCurrentTimeUs();
        cv::Mat frame;
        
        // 采集帧
        if (!cap.read(frame)) {
            printf("❌ 帧采集失败\n");
            continue;
        }
        int64_t t1 = getCurrentTimeUs();
        
        RimBasketballDetectionResult detection_result;
        int64_t t2 = t1, t3 = t1;
        
        // 跳帧策略：只处理部分帧
        if (frame_count % frame_skip == 0) {
            // 预处理：letterbox到640x640
            bool debug_first_frame = (processed_count == 0);  // 第一帧保存调试信息
            ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx, debug_first_frame);
            if (ret != 0) {
                printf("❌ 预处理失败\n");
                continue;
            }
            t2 = getCurrentTimeUs();
            
            // 零拷贝推理和9输出后处理
            memset(&detection_result, 0, sizeof(detection_result));
            
            ret = zero_copy_inference_and_postprocess(&rknn_app_ctx, &zero_copy_ctx, 
                                                     &detection_result, 0.1f, 0.4f);  // 降低置信度阈值到0.1
            t3 = getCurrentTimeUs();
            
            if (ret != 0) {
                printf("❌ 推理失败\n");
                continue;
            }
            
            // 保存检测结果
            last_detection_result = detection_result;
            processed_count++;
        } else {
            // 跳帧：复用上次检测结果
            detection_result = last_detection_result;
        }
        
        // 准备显示帧
        cv::Mat display_frame = frame.clone();
        
        // 计算letterbox逆变换参数
        int src_width = frame.cols;   // 1920
        int src_height = frame.rows;  // 1200  
        int dst_width = 640;
        int dst_height = 640;
        
        float scale_w = (float)dst_width / src_width;   // 640/1920 = 0.333
        float scale_h = (float)dst_height / src_height; // 640/1200 = 0.533
        float scale = std::min(scale_w, scale_h);       // 0.333
        int new_width = (int)(src_width * scale);       // 1920 * 0.333 = 640
        int new_height = (int)(src_height * scale);     // 1200 * 0.333 = 400
        int offset_x = (dst_width - new_width) / 2;     // (640-640)/2 = 0
        int offset_y = (dst_height - new_height) / 2;   // (640-400)/2 = 120
        
        // 逆变换参数
        float inv_scale = 1.0f / scale;  // 3.0
        
        // 绘制检测结果
        draw_detections(display_frame, &detection_result, inv_scale, -offset_x, -offset_y);
        
        // 强制绘制测试框以验证绘制功能
        cv::rectangle(display_frame, cv::Point(50, 50), cv::Point(150, 100), cv::Scalar(255, 255, 0), 3);
        cv::putText(display_frame, "ALWAYS VISIBLE", cv::Point(55, 75), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // 添加测试框架，确保绘制功能正常
        if (detection_result.count == 0 && frame_count % 60 == 0) {
            // 每60帧显示一个测试框
            cv::rectangle(display_frame, cv::Point(100, 100), cv::Point(300, 200), cv::Scalar(0, 255, 255), 3);
            cv::putText(display_frame, "TEST: No detections", cv::Point(110, 130), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
        
        // 绘制性能信息
        // 动态调整跳帧策略
        int64_t frame_time = t3 - t0;
        total_time += frame_time;
        frame_count++;
        
        // 性能监控和跳帧调整
        if (frame_count % 90 == 0) {  // 每90帧(约1秒@90fps)报告一次
            float current_fps = frame_count * 1000000.0f / total_time;
            float process_fps = processed_count * 1000000.0f / total_time;
            
            printf("[PERF] Frame:%d | FPS:%.1f | Skip:1/%d | Det:%d | Cap:%.1f Pre:%.1f Inf:%.1f\n", 
                   frame_count, current_fps, frame_skip, detection_result.count,
                   (t1-t0)/1000.0f, (t2-t1)/1000.0f, (t3-t2)/1000.0f);
            
            // 动态调整跳帧策略
            if (current_fps < 85.0f && frame_skip < 3) {
                frame_skip++;
                printf("[ADAPT] Low FPS, increase skip: 1/%d\n", frame_skip);
            } else if (current_fps > 95.0f && frame_skip > 1) {
                frame_skip--;
                printf("[ADAPT] High FPS, reduce skip: 1/%d\n", frame_skip);
            }
        }
        
        // 绘制详细性能信息
        float avg_fps = frame_count * 1000000.0f / total_time;
        char perf_text[512];
        sprintf(perf_text, "FPS: %.1f | Cap: %.1fms | Pre: %.1fms | Inf: %.1fms | Tot: %.1fms", 
               avg_fps, (t1-t0)/1000.0f, (t2-t1)/1000.0f, (t3-t2)/1000.0f, frame_time/1000.0f);
        cv::putText(display_frame, perf_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // 显示跳帧和检测信息
        char detail_text[256];
        sprintf(detail_text, "Skip: 1/%d | Process: %.1f%% | Detections: %d", 
               frame_skip, (processed_count * 100.0f / frame_count), detection_result.count);
        cv::putText(display_frame, detail_text, cv::Point(10, 65), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // 显示检测详情
        if (detection_result.count > 0) {
            char detail_text[256];
            int rim_count = 0, ball_count = 0;
            for (int i = 0; i < detection_result.count; i++) {
                if (detection_result.detections[i].class_id == RIM_CLASS_ID) {
                    rim_count++;
                } else {
                    ball_count++;
                }
            }
            sprintf(detail_text, "Rim: %d, Ball: %d", rim_count, ball_count);
            cv::putText(display_frame, detail_text, cv::Point(10, 70), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        }
        
        cv::imshow(WINDOW_NAME, display_frame);
        int key = cv::waitKey(1);
        if (key == 27) {  // ESC键
            break;
        }
    }

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    release_rim_basketball_model(&rknn_app_ctx);

    // 打印最终性能统计
    int64_t total_runtime = getCurrentTimeUs() - start_time_overall;
    printf("\n=== 性能统计 ===\n");
    printf("总运行时间: %.2f秒\n", total_runtime / 1000000.0f);
    printf("总帧数: %d\n", frame_count);
    printf("平均FPS: %.1f\n", frame_count * 1000000.0f / total_time);

    return 0;
}