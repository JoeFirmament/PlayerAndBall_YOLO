/*-------------------------------------------
            双摄像头双线程检测系统
            
用途: 同时使用两个摄像头进行不同的检测任务
架构:
- 线程1: 摄像头0 → 姿态检测 (Q_yolov8_pose.rknn)
- 线程2: 摄像头2 → 篮筐篮球检测 (Q_Rim_Basketball_724_JZ.rknn)
优化: 双NPU零拷贝，真正的并行处理
显示: 可选择单独显示或拼接显示
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
#include <atomic>
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

// 姿态检测相关
#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "BYTETracker.h"
#include "letterbox_utils.h"

// 篮筐篮球检测相关
#include "rim_basketball_postprocess.h"

// 全局变量
std::atomic<bool> g_running(true);
std::atomic<bool> g_enable_tracking(true);
std::atomic<bool> g_show_combined(true); // true: 拼接显示, false: 分别显示

// 姿态检测骨架连接
int skeleton[38] = {16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
                   7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7};

// 姿态检测零拷贝上下文
typedef struct {
    rknn_tensor_mem* input_mem;
    rknn_tensor_mem* output_mems[4];
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attrs[4];
    int model_width;
    int model_height;
    int model_channels;
    letterbox_context_t letterbox_ctx;
} pose_zero_copy_context_t;

// 篮筐篮球检测零拷贝上下文
typedef struct {
    rknn_tensor_mem* input_mem;
    rknn_tensor_mem* output_mems[10];
    rknn_tensor_attr input_attr;
    rknn_tensor_attr output_attrs[10];
    int model_width;
    int model_height;
    int model_channels;
} rim_zero_copy_context_t;

// 相机标定结构体
typedef struct {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    cv::Mat homography;
    bool is_initialized;
    int calib_width;
    int calib_height;
} camera_mapping_t;

// 全局变量
camera_mapping_t g_camera_mapping = {};
BYTETracker g_byte_track;

// 线程结果共享
struct PoseResult {
    cv::Mat frame;
    int detections_count;
    float fps;
    std::mutex mutex;
    bool ready = false;
};

struct RimBasketballResult {
    cv::Mat frame;
    int detections_count;
    bool has_rim;
    bool has_basketball;
    float fps;
    std::mutex mutex;
    bool ready = false;
};

PoseResult g_pose_result;
RimBasketballResult g_rim_result;

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

// 加载相机标定参数
static int load_camera_calibration(const char* calib_file, camera_mapping_t* mapping) {
    if (!calib_file) {
        printf("⚠️ 未指定标定文件，跳过坐标映射功能\n");
        return 0;
    }
    
    cv::FileStorage fs(calib_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        printf("⚠️ 无法打开标定文件: %s，跳过坐标映射功能\n", calib_file);
        return 0;
    }
    
    try {
        fs["camera_matrix"] >> mapping->camera_matrix;
        fs["dist_coeffs"] >> mapping->dist_coeffs;
        fs["homography"] >> mapping->homography;
        fs["image_width"] >> mapping->calib_width;
        fs["image_height"] >> mapping->calib_height;
        
        mapping->is_initialized = true;
        printf("✓ 成功加载相机标定参数: %s\n", calib_file);
        return 0;
    } catch (const cv::Exception& e) {
        printf("⚠️ 标定文件格式错误: %s\n", e.what());
        return -1;
    }
}

// 姿态检测线程初始化零拷贝内存
static int init_pose_zero_copy_mem(rknn_app_context_t* app_ctx, pose_zero_copy_context_t* zc_ctx) {
    int ret;
    
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("❌ 创建姿态检测输入零拷贝内存失败！\n");
        return -1;
    }
    
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("❌ 设置姿态检测输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->output_attrs[i].size_with_stride);
        if (!zc_ctx->output_mems[i]) {
            printf("❌ 创建姿态检测输出零拷贝内存[%d]失败！\n", i);
            return -1;
        }
        
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i], &zc_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("❌ 设置姿态检测输出零拷贝内存[%d]失败! ret=%d\n", i, ret);
            return -1;
        }
    }
    
    return 0;
}

// 篮筐篮球检测线程初始化零拷贝内存
static int init_rim_zero_copy_mem(rknn_app_context_t* app_ctx, rim_zero_copy_context_t* zc_ctx) {
    int ret;
    
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("❌ 创建篮筐篮球检测输入零拷贝内存失败！\n");
        return -1;
    }
    
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("❌ 设置篮筐篮球检测输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->output_attrs[i].size_with_stride);
        if (!zc_ctx->output_mems[i]) {
            printf("❌ 创建篮筐篮球检测输出零拷贝内存[%d]失败！\n", i);
            return -1;
        }
        
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i], &zc_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("❌ 设置篮筐篮球检测输出零拷贝内存[%d]失败! ret=%d\n", i, ret);
            return -1;
        }
    }
    
    return 0;
}

// 姿态检测letterbox预处理
static int optimized_letterbox_to_npu(const cv::Mat& src, pose_zero_copy_context_t* zc_ctx) {
    cv::Mat npu_mat(zc_ctx->model_height, zc_ctx->model_width, CV_8UC3, zc_ctx->input_mem->virt_addr);
    init_letterbox_context(&zc_ctx->letterbox_ctx, src.cols, src.rows, zc_ctx->model_width, zc_ctx->model_height, false);
    return letterbox_preprocess(src, npu_mat, &zc_ctx->letterbox_ctx);
}

// 篮筐篮球检测letterbox预处理
static int letterbox_resize_to_npu(const cv::Mat& src, rim_zero_copy_context_t* zc_ctx, float* scale, int* x_pad, int* y_pad) {
    int src_w = src.cols;
    int src_h = src.rows;
    int dst_w = zc_ctx->model_width;
    int dst_h = zc_ctx->model_height;
    
    *scale = std::min((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = (int)(src_w * (*scale));
    int new_h = (int)(src_h * (*scale));
    
    *x_pad = (dst_w - new_w) / 2;
    *y_pad = (dst_h - new_h) / 2;
    
    cv::Mat npu_mat(dst_h, dst_w, CV_8UC3, zc_ctx->input_mem->virt_addr);
    npu_mat.setTo(cv::Scalar(114, 114, 114));
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    
    cv::Rect roi(*x_pad, *y_pad, new_w, new_h);
    resized.copyTo(npu_mat(roi));
    
    return 0;
}

// 绘制姿态检测结果
static void draw_pose_results(cv::Mat& img, object_detect_result_list* results, 
                            const camera_mapping_t* mapping, const letterbox_context_t* letterbox_ctx) {
    for (int i = 0; i < results->count; i++) {
        object_detect_result* result = &(results->results[i]);
        
        float x1 = (result->box.left - letterbox_ctx->offset_x) / letterbox_ctx->scale;
        float y1 = (result->box.top - letterbox_ctx->offset_y) / letterbox_ctx->scale;
        float x2 = (result->box.right - letterbox_ctx->offset_x) / letterbox_ctx->scale;
        float y2 = (result->box.bottom - letterbox_ctx->offset_y) / letterbox_ctx->scale;
        
        cv::rectangle(img, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(255, 0, 0), 2);
        
        char conf_str[50];
        snprintf(conf_str, sizeof(conf_str), "%.2f", result->prop);
        cv::putText(img, conf_str, cv::Point((int)x1, (int)y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        
        // 绘制关键点和骨架
        for (int j = 0; j < 17; j++) {
            if (result->keypoints[j][2] > 0.5) {
                float kp_x = (result->keypoints[j][0] - letterbox_ctx->offset_x) / letterbox_ctx->scale;
                float kp_y = (result->keypoints[j][1] - letterbox_ctx->offset_y) / letterbox_ctx->scale;
                cv::circle(img, cv::Point((int)kp_x, (int)kp_y), 3, cv::Scalar(0, 255, 0), -1);
            }
        }
        
        // 绘制骨架
        for (int k = 0; k < 19; k++) {
            int kpt_a = skeleton[k * 2] - 1;
            int kpt_b = skeleton[k * 2 + 1] - 1;
            
            if (result->keypoints[kpt_a][2] > 0.5 && result->keypoints[kpt_b][2] > 0.5) {
                float x_a = (result->keypoints[kpt_a][0] - letterbox_ctx->offset_x) / letterbox_ctx->scale;
                float y_a = (result->keypoints[kpt_a][1] - letterbox_ctx->offset_y) / letterbox_ctx->scale;
                float x_b = (result->keypoints[kpt_b][0] - letterbox_ctx->offset_x) / letterbox_ctx->scale;
                float y_b = (result->keypoints[kpt_b][1] - letterbox_ctx->offset_y) / letterbox_ctx->scale;
                
                cv::line(img, cv::Point((int)x_a, (int)y_a), cv::Point((int)x_b, (int)y_b), 
                        cv::Scalar(0, 255, 255), 2);
            }
        }
    }
}

// 绘制篮筐篮球检测结果
static void draw_rim_basketball_results(cv::Mat& img, const RimBasketballDetectionResult* detections) {
    for (int i = 0; i < detections->count; i++) {
        const RimBasketballDetection* det = &detections->detections[i];
        
        cv::Scalar color;
        if (det->class_id == 0) { // rim
            color = cv::Scalar(0, 255, 0); // 绿色
        } else { // basketball
            color = cv::Scalar(0, 165, 255); // 橙色
        }
        
        cv::Rect rect(det->x - det->w/2, det->y - det->h/2, det->w, det->h);
        cv::rectangle(img, rect, color, 3);
        
        char label[100];
        snprintf(label, sizeof(label), "%s %.2f", det->class_name, det->confidence);
        
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        cv::Point label_origin(rect.x, rect.y - 10);
        
        cv::rectangle(img, cv::Point(label_origin.x, label_origin.y - label_size.height - baseline),
                     cv::Point(label_origin.x + label_size.width, label_origin.y + baseline), color, -1);
        cv::putText(img, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
}

// 姿态检测线程
void pose_detection_thread(const char* model_path, const char* calib_path, int camera_id) {
    printf("🔥 启动姿态检测线程 (摄像头%d)\n", camera_id);
    
    rknn_app_context_t rknn_app_ctx;
    pose_zero_copy_context_t zero_copy_ctx = {};
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    // 初始化模型
    int ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("❌ 姿态检测模型初始化失败！\n");
        return;
    }
    
    // 初始化零拷贝内存
    ret = init_pose_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("❌ 姿态检测零拷贝内存初始化失败！\n");
        release_yolov8_pose_model(&rknn_app_ctx);
        return;
    }
    
    // 加载标定参数
    if (calib_path) {
        load_camera_calibration(calib_path, &g_camera_mapping);
    }
    
    // 打开摄像头
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        printf("❌ 无法打开姿态检测摄像头%d！\n", camera_id);
        return;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    printf("✓ 姿态检测摄像头%d打开成功\n", camera_id);
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (g_running) {
        if (!cap.read(frame)) {
            printf("❌ 姿态检测读取帧失败！\n");
            continue;
        }
        
        frame_count++;
        
        // 预处理
        ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx);
        if (ret != 0) continue;
        
        // 推理
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0) continue;
        
        // 后处理 - 创建兼容的letterbox_t结构
        letterbox_t legacy_letterbox;
        legacy_letterbox.x_pad = zero_copy_ctx.letterbox_ctx.offset_x;
        legacy_letterbox.y_pad = zero_copy_ctx.letterbox_ctx.offset_y;
        legacy_letterbox.scale = zero_copy_ctx.letterbox_ctx.scale;
        
        object_detect_result_list pose_results;
        ret = post_process(&rknn_app_ctx, zero_copy_ctx.output_mems, &legacy_letterbox, 0.25, 0.45, &pose_results);
        if (ret != 0) continue;
        
        // 绘制结果
        draw_pose_results(frame, &pose_results, &g_camera_mapping, &zero_copy_ctx.letterbox_ctx);
        
        // ByteTrack跟踪 (简化版)
        if (g_enable_tracking) {
            // 这里可以添加跟踪逻辑
        }
        
        // 计算FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(current_time - start_time).count();
        float fps = frame_count / elapsed;
        
        // 显示性能信息
        char info_str[200];
        snprintf(info_str, sizeof(info_str), "Pose: FPS=%.1f | Det=%d | Track=%s", 
                fps, pose_results.count, g_enable_tracking ? "ON" : "OFF");
        cv::putText(frame, info_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 更新共享结果
        {
            std::lock_guard<std::mutex> lock(g_pose_result.mutex);
            g_pose_result.frame = frame.clone();
            g_pose_result.detections_count = pose_results.count;
            g_pose_result.fps = fps;
            g_pose_result.ready = true;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
    }
    
    // 清理资源
    release_yolov8_pose_model(&rknn_app_ctx);
    printf("姿态检测线程退出\n");
}

// 篮筐篮球检测线程  
void rim_basketball_detection_thread(const char* model_path, int camera_id) {
    printf("🏀 启动篮筐篮球检测线程 (摄像头%d)\n", camera_id);
    
    rknn_app_context_t rknn_app_ctx;
    rim_zero_copy_context_t zero_copy_ctx = {};
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    // 初始化模型 (这里需要实现rim_basketball模型的初始化函数)
    // 暂时使用姿态检测的初始化函数作为占位符
    int ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("❌ 篮筐篮球检测模型初始化失败！\n");
        return;
    }
    
    // 初始化零拷贝内存
    ret = init_rim_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("❌ 篮筐篮球检测零拷贝内存初始化失败！\n");
        release_yolov8_pose_model(&rknn_app_ctx);
        return;
    }
    
    // 打开摄像头
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        printf("❌ 无法打开篮筐篮球检测摄像头%d！\n", camera_id);
        return;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    printf("✓ 篮筐篮球检测摄像头%d打开成功\n", camera_id);
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (g_running) {
        if (!cap.read(frame)) {
            printf("❌ 篮筐篮球检测读取帧失败！\n");
            continue;
        }
        
        frame_count++;
        int orig_w = frame.cols;
        int orig_h = frame.rows;
        
        // 预处理
        float scale;
        int x_pad, y_pad;
        ret = letterbox_resize_to_npu(frame, &zero_copy_ctx, &scale, &x_pad, &y_pad);
        if (ret != 0) continue;
        
        // 推理
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0) continue;
        
        // 后处理 (使用篮筐篮球专用后处理)
        RimBasketballDetectionResult detections;
        
        // 准备RKNN输出数据
        rknn_output outputs[10];
        memset(outputs, 0, sizeof(outputs));
        
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++) {
            outputs[i].want_float = 1;
            outputs[i].is_prealloc = 1;
            outputs[i].buf = zero_copy_ctx.output_mems[i]->virt_addr;
            outputs[i].size = zero_copy_ctx.output_attrs[i].size;
        }
        
        ret = process_rim_basketball_outputs(outputs, zero_copy_ctx.output_attrs, 0.3f, 0.3f, &detections);
        if (ret != 0) continue;
        
        // 坐标映射
        for (int i = 0; i < detections.count; i++) {
            RimBasketballDetection* det = &detections.detections[i];
            
            float x1 = det->x - det->w / 2.0f;
            float y1 = det->y - det->h / 2.0f;
            float x2 = det->x + det->w / 2.0f;
            float y2 = det->y + det->h / 2.0f;
            
            x1 = (x1 - x_pad) / scale;
            y1 = (y1 - y_pad) / scale;
            x2 = (x2 - x_pad) / scale;
            y2 = (y2 - y_pad) / scale;
            
            x1 = fmaxf(0.0f, fminf(x1, orig_w - 1));
            y1 = fmaxf(0.0f, fminf(y1, orig_h - 1));
            x2 = fmaxf(0.0f, fminf(x2, orig_w - 1));
            y2 = fmaxf(0.0f, fminf(y2, orig_h - 1));
            
            det->x = (x1 + x2) / 2.0f;
            det->y = (y1 + y2) / 2.0f;
            det->w = x2 - x1;
            det->h = y2 - y1;
        }
        
        // 绘制结果
        draw_rim_basketball_results(frame, &detections);
        
        // 分析ROI
        bool has_rim = false, has_basketball = false;
        for (int i = 0; i < detections.count; i++) {
            if (detections.detections[i].class_id == 0) has_rim = true;
            if (detections.detections[i].class_id == 1) has_basketball = true;
        }
        
        // 计算FPS
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(current_time - start_time).count();
        float fps = frame_count / elapsed;
        
        // 显示性能信息
        char info_str[200];
        snprintf(info_str, sizeof(info_str), "Rim&Ball: FPS=%.1f | Det=%d | Rim=%s | Ball=%s", 
                fps, detections.count, has_rim ? "YES" : "NO", has_basketball ? "YES" : "NO");
        cv::putText(frame, info_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 更新共享结果
        {
            std::lock_guard<std::mutex> lock(g_rim_result.mutex);
            g_rim_result.frame = frame.clone();
            g_rim_result.detections_count = detections.count;
            g_rim_result.has_rim = has_rim;
            g_rim_result.has_basketball = has_basketball;
            g_rim_result.fps = fps;
            g_rim_result.ready = true;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30fps
    }
    
    // 清理资源
    release_yolov8_pose_model(&rknn_app_ctx);
    printf("篮筐篮球检测线程退出\n");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("用法: %s <姿态模型路径> <篮筐篮球模型路径> [标定文件] [姿态摄像头ID] [篮筐摄像头ID]\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn ../models/Q_Rim_Basketball_724_JZ.rknn\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn ../models/Q_Rim_Basketball_724_JZ.rknn ../data/2025_7_11pm.json 0 2\n", argv[0]);
        return -1;
    }
    
    const char* pose_model_path = argv[1];
    const char* rim_model_path = argv[2];
    const char* calib_path = (argc > 3) ? argv[3] : nullptr;
    int pose_camera_id = (argc > 4) ? atoi(argv[4]) : 0;
    int rim_camera_id = (argc > 5) ? atoi(argv[5]) : 2;
    
    // 设置信号处理
    signal(SIGINT, sig_handler);
    
    printf("========================================\n");
    printf("      双摄像头双线程检测系统 v1.0\n");
    printf("========================================\n");
    printf("姿态模型: %s (摄像头%d)\n", pose_model_path, pose_camera_id);
    printf("篮筐模型: %s (摄像头%d)\n", rim_model_path, rim_camera_id);
    if (calib_path) printf("标定文件: %s\n", calib_path);
    printf("按键控制:\n");
    printf("  [ESC] - 退出程序\n");
    printf("  [T]   - 切换跟踪开关\n");
    printf("  [C]   - 切换显示模式 (拼接/分别)\n");
    printf("========================================\n");
    
    // 启动两个检测线程
    std::thread pose_thread(pose_detection_thread, pose_model_path, calib_path, pose_camera_id);
    std::thread rim_thread(rim_basketball_detection_thread, rim_model_path, rim_camera_id);
    
    // 主线程负责显示
    printf("等待检测线程启动...\n");
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    printf("开始显示检测结果...\n");
    
    while (g_running) {
        cv::Mat display_frame;
        
        if (g_show_combined) {
            // 拼接显示
            cv::Mat pose_frame, rim_frame;
            bool pose_ready = false, rim_ready = false;
            
            {
                std::lock_guard<std::mutex> lock(g_pose_result.mutex);
                if (g_pose_result.ready) {
                    pose_frame = g_pose_result.frame.clone();
                    pose_ready = true;
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(g_rim_result.mutex);
                if (g_rim_result.ready) {
                    rim_frame = g_rim_result.frame.clone();
                    rim_ready = true;
                }
            }
            
            if (pose_ready && rim_ready) {
                // 水平拼接
                cv::Size target_size(960, 540); // 缩小显示
                cv::Mat pose_resized, rim_resized;
                cv::resize(pose_frame, pose_resized, target_size);
                cv::resize(rim_frame, rim_resized, target_size);
                
                cv::hconcat(pose_resized, rim_resized, display_frame);
                
                // 添加分隔线
                cv::line(display_frame, cv::Point(960, 0), cv::Point(960, 540), cv::Scalar(255, 255, 255), 2);
            }
        } else {
            // 分别显示
            {
                std::lock_guard<std::mutex> lock(g_pose_result.mutex);
                if (g_pose_result.ready) {
                    cv::imshow("Pose Detection", g_pose_result.frame);
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(g_rim_result.mutex);
                if (g_rim_result.ready) {
                    cv::imshow("Rim Basketball Detection", g_rim_result.frame);
                }
            }
        }
        
        if (g_show_combined && !display_frame.empty()) {
            cv::imshow("Dual Camera Detection", display_frame);
        }
        
        // 处理按键
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            g_running = false;
            break;
        } else if (key == 't' || key == 'T') {
            g_enable_tracking = !g_enable_tracking;
            printf("跟踪模式: %s\n", g_enable_tracking ? "开启" : "关闭");
        } else if (key == 'c' || key == 'C') {
            g_show_combined = !g_show_combined;
            printf("显示模式: %s\n", g_show_combined ? "拼接显示" : "分别显示");
            if (!g_show_combined) {
                cv::destroyWindow("Dual Camera Detection");
            } else {
                cv::destroyWindow("Pose Detection");
                cv::destroyWindow("Rim Basketball Detection");
            }
        }
    }
    
    printf("等待线程退出...\n");
    pose_thread.join();
    rim_thread.join();
    
    cv::destroyAllWindows();
    printf("程序退出\n");
    return 0;
}