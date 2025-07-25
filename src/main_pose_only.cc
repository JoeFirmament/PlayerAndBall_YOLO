/*-------------------------------------------
                纯姿态检测版本 - NPU零拷贝优化
                
用途: 专注于人体姿态检测和跟踪，移除篮球检测功能
原理: 直接在NPU内存中进行预处理，避免数据传输
架构设计: 单线程架构，使用NPU1进行姿态检测
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "yolov8-pose.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "BYTETracker.h"
#include "im2d.h"
#include "im2d_type.h"
#include "im2d_single.h"
#include "RgaUtils.h"
#include "letterbox_utils.h"

int skeleton[38] ={16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}; 

// 全局变量，用于信号处理
bool g_running = true;

// 全局变量，用于控制ByteTrack功能开关
bool g_enable_tracking = true;

typedef struct {
    rknn_tensor_mem* input_mem;        // NPU输入内存，直接被OpenCV Mat使用
    rknn_tensor_mem* output_mems[4];   // NPU输出内存，YOLOv8 pose有4个输出
    rknn_tensor_attr input_attr;       // 输入属性，包含步幅信息
    rknn_tensor_attr output_attrs[4];  // 输出属性
    int model_width;                   // 模型输入宽度 (640)
    int model_height;                  // 模型输入高度 (640)
    int model_channels;                // 模型输入通道数 (3)
    letterbox_context_t letterbox_ctx; // letterbox上下文
} zero_copy_context_t;

// 相机标定和Homography相关的结构体
typedef struct {
    cv::Mat camera_matrix;    // 相机内参矩阵
    cv::Mat dist_coeffs;      // 畸变系数
    cv::Mat homography;       // 单应性矩阵
    bool is_initialized;      // 是否已初始化
    int calib_width;          // 标定分辨率宽
    int calib_height;         // 标定分辨率高
} camera_mapping_t;

// 全局变量
camera_mapping_t g_camera_mapping = {};

// ByteTrack实例
BYTETracker g_byte_track;

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

// 初始化零拷贝内存
static int init_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    int ret;
    
    // 设置输入属性
    zc_ctx->input_attr = app_ctx->input_attrs[0];
    zc_ctx->input_attr.type = RKNN_TENSOR_UINT8;
    zc_ctx->input_attr.fmt = RKNN_TENSOR_NHWC;
    zc_ctx->model_width = app_ctx->model_width;
    zc_ctx->model_height = app_ctx->model_height;
    zc_ctx->model_channels = app_ctx->model_channel;
    
    // 创建NPU直接访问的输入内存
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("创建输入零拷贝内存失败！\n");
        return -1;
    }
    
    // 将NPU内存绑定到推理上下文
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }

    // 创建输出内存
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = rknn_create_mem(app_ctx->rknn_ctx, app_ctx->output_attrs[i].size_with_stride);
        if (!zc_ctx->output_mems[i]) {
            printf("创建输出零拷贝内存[%d]失败！\n", i);
            return -1;
        }
        
        ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i], &zc_ctx->output_attrs[i]);
        if (ret < 0) {
            printf("设置输出零拷贝内存[%d]失败! ret=%d\n", i, ret);
            return -1;
        }
    }
    
    printf("✓ 零拷贝内存初始化成功\n");
    return 0;
}

// 释放零拷贝内存
static void release_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    if (zc_ctx->input_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->input_mem);
    }
    
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (zc_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i]);
        }
    }
}

// 优化的letterbox到NPU内存
static int optimized_letterbox_to_npu(const cv::Mat& src, zero_copy_context_t* zc_ctx) {
    // 创建OpenCV Mat直接指向NPU内存
    cv::Mat npu_mat(zc_ctx->model_height, zc_ctx->model_width, CV_8UC3, zc_ctx->input_mem->virt_addr);
    
    // 使用letterbox工具进行resize
    return letterbox_resize(src, npu_mat, &zc_ctx->letterbox_ctx);
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
        printf("  标定分辨率: %dx%d\n", mapping->calib_width, mapping->calib_height);
        return 0;
    } catch (const cv::Exception& e) {
        printf("⚠️ 标定文件格式错误: %s\n", e.what());
        return -1;
    }
}

// 转换图像坐标到真实世界坐标
static cv::Point2f image_to_world_coordinate(cv::Point2f image_point, const camera_mapping_t* mapping) {
    if (!mapping->is_initialized) {
        return image_point;
    }
    
    std::vector<cv::Point2f> image_points = {image_point};
    std::vector<cv::Point2f> world_points;
    
    cv::perspectiveTransform(image_points, world_points, mapping->homography);
    return world_points[0];
}

// 绘制姿态关键点和骨架
static void draw_pose_results(cv::Mat& img, object_detect_result_list* results, 
                            const camera_mapping_t* mapping, const letterbox_context_t* letterbox_ctx) {
    for (int i = 0; i < results->count; i++) {
        object_detect_result* result = &(results->results[i]);
        
        // 转换检测框坐标
        float x1 = (result->box.left - letterbox_ctx->x_pad) / letterbox_ctx->scale;
        float y1 = (result->box.top - letterbox_ctx->y_pad) / letterbox_ctx->scale;
        float x2 = (result->box.right - letterbox_ctx->x_pad) / letterbox_ctx->scale;
        float y2 = (result->box.bottom - letterbox_ctx->y_pad) / letterbox_ctx->scale;
        
        // 绘制检测框 (蓝色)
        cv::rectangle(img, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2), cv::Scalar(255, 0, 0), 2);
        
        // 绘制置信度
        char conf_str[50];
        snprintf(conf_str, sizeof(conf_str), "%.2f", result->prop);
        cv::putText(img, conf_str, cv::Point((int)x1, (int)y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        
        // 绘制关键点
        for (int j = 0; j < OBJ_NUMB_MAX_SIZE; j++) {
            if (result->point[j].score > 0.5) {
                float kp_x = (result->point[j].x - letterbox_ctx->x_pad) / letterbox_ctx->scale;
                float kp_y = (result->point[j].y - letterbox_ctx->y_pad) / letterbox_ctx->scale;
                
                cv::circle(img, cv::Point((int)kp_x, (int)kp_y), 3, cv::Scalar(0, 255, 0), -1);
                
                // 如果有坐标映射，显示真实世界坐标
                if (mapping->is_initialized && j == 15) { // 左脚踝作为参考点
                    cv::Point2f world_point = image_to_world_coordinate(cv::Point2f(kp_x, kp_y), mapping);
                    char coord_str[50];
                    snprintf(coord_str, sizeof(coord_str), "(%.1f,%.1f)", world_point.x, world_point.y);
                    cv::putText(img, coord_str, cv::Point((int)kp_x, (int)kp_y-10), 
                              cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 0), 1);
                }
            }
        }
        
        // 绘制骨架
        for (int k = 0; k < 19; k++) {
            int kpt_a = skeleton[k * 2] - 1;
            int kpt_b = skeleton[k * 2 + 1] - 1;
            
            if (result->point[kpt_a].score > 0.5 && result->point[kpt_b].score > 0.5) {
                float x_a = (result->point[kpt_a].x - letterbox_ctx->x_pad) / letterbox_ctx->scale;
                float y_a = (result->point[kpt_a].y - letterbox_ctx->y_pad) / letterbox_ctx->scale;
                float x_b = (result->point[kpt_b].x - letterbox_ctx->x_pad) / letterbox_ctx->scale;
                float y_b = (result->point[kpt_b].y - letterbox_ctx->y_pad) / letterbox_ctx->scale;
                
                cv::line(img, cv::Point((int)x_a, (int)y_a), cv::Point((int)x_b, (int)y_b), 
                        cv::Scalar(0, 255, 255), 2);
            }
        }
    }
}

// ByteTrack跟踪处理
static void process_tracking(cv::Mat& img, object_detect_result_list* results, 
                           const letterbox_context_t* letterbox_ctx) {
    if (!g_enable_tracking) {
        return;
    }
    
    // 转换为ByteTrack格式
    std::vector<BYTETracker::Object> objects;
    for (int i = 0; i < results->count; i++) {
        object_detect_result* result = &(results->results[i]);
        
        float x1 = (result->box.left - letterbox_ctx->x_pad) / letterbox_ctx->scale;
        float y1 = (result->box.top - letterbox_ctx->y_pad) / letterbox_ctx->scale;
        float x2 = (result->box.right - letterbox_ctx->x_pad) / letterbox_ctx->scale;
        float y2 = (result->box.bottom - letterbox_ctx->y_pad) / letterbox_ctx->scale;
        
        BYTETracker::Object obj;
        obj.rect = cv::Rect2f(x1, y1, x2-x1, y2-y1);
        obj.prob = result->prop;
        obj.label = 0; // person
        objects.push_back(obj);
    }
    
    // 执行跟踪
    auto tracks = g_byte_track.update(objects);
    
    // 绘制跟踪结果
    for (const auto& track : tracks) {
        cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色跟踪框
        cv::rectangle(img, track.rect, color, 2);
        
        char id_str[50];
        snprintf(id_str, sizeof(id_str), "ID:%d", track.track_id);
        cv::putText(img, id_str, cv::Point((int)track.rect.x, (int)track.rect.y-20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

int main(int argc, char **argv) {
    int ret;
    rknn_app_context_t rknn_app_ctx;
    zero_copy_context_t zero_copy_ctx = {};
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    if (argc < 2) {
        printf("用法: %s <rknn模型路径> [标定文件路径]\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* calib_path = (argc > 2) ? argv[2] : nullptr;
    
    // 设置信号处理
    signal(SIGINT, sig_handler);
    
    printf("========================================\n");
    printf("        纯姿态检测系统 v1.0\n");
    printf("========================================\n");
    printf("模型文件: %s\n", model_path);
    if (calib_path) {
        printf("标定文件: %s\n", calib_path);
    }
    printf("按键说明:\n");
    printf("  [ESC] - 退出程序\n");
    printf("  [T]   - 切换ByteTrack跟踪开关\n");
    printf("========================================\n");
    
    // 初始化模型
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("初始化YOLOv8姿态模型失败！\n");
        return -1;
    }
    printf("✓ YOLOv8姿态模型初始化成功\n");
    
    // 预先声明变量，避免goto跨越初始化
    cv::Mat frame;
    int frame_count = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    cv::VideoCapture cap;
    
    // 初始化零拷贝内存
    ret = init_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("初始化零拷贝内存失败！\n");
        goto exit;
    }
    
    // 加载相机标定参数
    if (calib_path) {
        load_camera_calibration(calib_path, &g_camera_mapping);
    }
    
    // 打开摄像头
    cap.open(0);
    if (!cap.isOpened()) {
        printf("无法打开摄像头！\n");
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
    
    start_time = std::chrono::high_resolution_clock::now();
    
    printf("开始处理视频流...\n");
    
    while (g_running) {
        // 读取帧
        if (!cap.read(frame)) {
            printf("读取帧失败！\n");
            break;
        }
        
        frame_count++;
        
        // 姿态检测推理
        auto inference_start = std::chrono::high_resolution_clock::now();
        
        // 零拷贝letterbox预处理
        ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx);
        if (ret != 0) {
            printf("letterbox预处理失败！\n");
            continue;
        }
        
        // 零拷贝推理
        ret = rknn_run(rknn_app_ctx.rknn_ctx, nullptr);
        if (ret < 0) {
            printf("推理失败! ret=%d\n", ret);
            continue;
        }
        
        // 后处理
        object_detect_result_list pose_results;
        ret = post_process(&rknn_app_ctx, zero_copy_ctx.output_mems, &zero_copy_ctx.letterbox_ctx, 0.25, 0.45, &pose_results);
        if (ret != 0) {
            printf("后处理失败！\n");
            continue;
        }
        
        auto inference_end = std::chrono::high_resolution_clock::now();
        float inference_time = std::chrono::duration<float, std::milli>(inference_end - inference_start).count();
        
        // 绘制结果
        draw_pose_results(frame, &pose_results, &g_camera_mapping, &zero_copy_ctx.letterbox_ctx);
        
        // ByteTrack跟踪
        process_tracking(frame, &pose_results, &zero_copy_ctx.letterbox_ctx);
        
        // 显示性能信息
        char fps_str[100];
        auto current_time = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(current_time - start_time).count();
        float fps = frame_count / elapsed;
        
        snprintf(fps_str, sizeof(fps_str), "FPS: %.1f | Inference: %.1fms | Detections: %d | Track: %s", 
                fps, inference_time, pose_results.count, g_enable_tracking ? "ON" : "OFF");
        cv::putText(frame, fps_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // 显示帧
        cv::imshow("YOLOv8 Pose Detection", frame);
        
        // 处理按键
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC
            break;
        } else if (key == 't' || key == 'T') {
            g_enable_tracking = !g_enable_tracking;
            printf("ByteTrack跟踪: %s\n", g_enable_tracking ? "开启" : "关闭");
        }
    }
    
exit:
    printf("正在清理资源...\n");
    
    // 释放零拷贝内存
    release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    
    // 释放模型
    ret = release_yolov8_pose_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("释放模型失败！\n");
    }
    
    cv::destroyAllWindows();
    printf("程序退出\n");
    return 0;
}