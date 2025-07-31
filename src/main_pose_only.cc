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
#include <unistd.h>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "yolov8-pose.h"
#include "postprocess.h"
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

// 优化的letterbox到NPU内存 - 使用工作版本的实现
static int optimized_letterbox_to_npu(cv::Mat& src, zero_copy_context_t* zc_ctx) {
    // 构造零拷贝letterbox上下文（使用已初始化的letterbox_ctx）
    zero_copy_letterbox_context_t letterbox_zc_ctx;
    letterbox_zc_ctx.input_mem = zc_ctx->input_mem;
    letterbox_zc_ctx.input_attr = zc_ctx->input_attr;
    letterbox_zc_ctx.model_width = zc_ctx->model_width;
    letterbox_zc_ctx.model_height = zc_ctx->model_height;
    letterbox_zc_ctx.model_channels = zc_ctx->model_channels;
    letterbox_zc_ctx.letterbox_ctx = zc_ctx->letterbox_ctx;  // 使用预初始化的上下文
    
    // 使用工作版本的letterbox函数
    return zero_copy_letterbox_preprocess(src, &letterbox_zc_ctx);
}

// 加载相机标定参数
static int load_camera_calibration(const char* calib_file, camera_mapping_t* mapping) {
    if (!calib_file) {
        printf("⚠️ 未指定标定文件，跳过坐标映射功能\n");
        return 0;
    }
    
    // 检查文件是否存在
    FILE* check_file = fopen(calib_file, "r");
    if (!check_file) {
        printf("❌ 错误: Homography JSON文件不存在: %s\n", calib_file);
        printf("请检查文件路径或创建标定文件\n");
        return -1;
    }
    fclose(check_file);
    
    // 读取JSON文件
    cv::FileStorage fs_homo(calib_file, cv::FileStorage::READ);
    if (!fs_homo.isOpened()) {
        printf("❌ 错误: 无法打开Homography JSON文件 %s\n", calib_file);
        return -1;
    }
    
    // 从JSON读取matrix数组并转换为3x3矩阵
    cv::FileNode homo_node = fs_homo["matrix"];
    if (homo_node.empty() || !homo_node.isSeq()) {
        printf("错误: JSON文件中缺少matrix数组\n");
        return -1;
    }

    if (homo_node.size() != 3) {
        printf("错误: matrix数组应为3x3矩阵，实际为%zu行\n", homo_node.size());
        return -1;
    }

    // 创建3x3矩阵并填充数据
    mapping->homography = cv::Mat::zeros(3, 3, CV_64F);
    int row = 0;
    for (cv::FileNodeIterator it = homo_node.begin(); it != homo_node.end(); ++it, ++row) {
        cv::FileNode row_node = *it;
        if (!row_node.isSeq() || row_node.size() != 3) {
            printf("错误: matrix第%d行应包含3个元素，实际为%zu个\n", row, row_node.size());
            return -1;
        }
        
        int col = 0;
        for (cv::FileNodeIterator col_it = row_node.begin(); col_it != row_node.end(); ++col_it, ++col) {
            mapping->homography.at<double>(row, col) = (double)*col_it;
        }
    }

    printf("✓ Homography矩阵加载成功\n");

    fs_homo.release();

    // 验证Homography矩阵维度
    if (mapping->homography.rows != 3 || mapping->homography.cols != 3) {
        printf("错误: Homography矩阵维度不正确\n");
        return -1;
    }

    // 不再需要相机内参和畸变系数，直接设置为空
    mapping->camera_matrix = cv::Mat();
    mapping->dist_coeffs = cv::Mat();
    mapping->calib_width = 0;
    mapping->calib_height = 0;

    mapping->is_initialized = true;
    return 0;
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
        
        // post_process已经进行了坐标反算，直接使用即可
        cv::Rect bbox(result->box.left, result->box.top,
                     result->box.right - result->box.left,
                     result->box.bottom - result->box.top);
        
        // ROI调试：输出检测框和ROI坐标
        if (i == 0) {  // 只显示第一个检测目标的信息
            printf("检测框: [%d,%d,%d,%d] -> ROI中心:(%.1f,%.1f)\n", 
                   result->box.left, result->box.top, result->box.right, result->box.bottom,
                   (bbox.x + bbox.x + bbox.width) * 0.5f, bbox.y + bbox.height);
        }
        
        // 绘制检测框 (蓝色)
        cv::rectangle(img, bbox, cv::Scalar(255, 0, 0), 2);
        
        // 绘制置信度
        char conf_str[50];
        snprintf(conf_str, sizeof(conf_str), "%.2f", result->prop);
        cv::putText(img, conf_str, cv::Point(bbox.x, bbox.y-5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
        
        // === 添加ROI地面定位点（紫色圆点） ===
        // 计算检测框下边缘中点
        cv::Point2f roi_bottom_center((bbox.x + bbox.x + bbox.width) * 0.5f, bbox.y + bbox.height);
        
        // 绘制ROI地面定位点（紫色圆点）
        cv::circle(img, roi_bottom_center, 4, cv::Scalar(255, 0, 255), -1);
        
        // 如果有坐标映射，显示ROI的真实世界坐标
        if (mapping->is_initialized) {
            cv::Point2f roi_world_point = image_to_world_coordinate(roi_bottom_center, mapping);
            char roi_coord_str[60];
            snprintf(roi_coord_str, sizeof(roi_coord_str), "ROI:(%.0f,%.0f)mm", roi_world_point.x, roi_world_point.y);
            cv::putText(img, roi_coord_str, cv::Point((int)roi_bottom_center.x - 40, (int)roi_bottom_center.y + 20), 
                      cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 255), 1);
        }
        
        // post_process已经对关键点进行了坐标反算，直接使用即可
        
        // 绘制关键点
        for (int j = 0; j < OBJ_NUMB_MAX_SIZE; j++) {
            if (result->keypoints[j][2] > 0.5) {
                cv::circle(img, cv::Point((int)result->keypoints[j][0], (int)result->keypoints[j][1]), 3, cv::Scalar(0, 255, 0), -1);
                
                // 如果有坐标映射，显示真实世界坐标
                if (mapping->is_initialized && j == 15) { // 左脚踝作为参考点
                    cv::Point2f world_point = image_to_world_coordinate(cv::Point2f(result->keypoints[j][0], result->keypoints[j][1]), mapping);
                    char coord_str[50];
                    snprintf(coord_str, sizeof(coord_str), "(%.1f,%.1f)", world_point.x, world_point.y);
                    cv::putText(img, coord_str, cv::Point((int)result->keypoints[j][0], (int)result->keypoints[j][1]-10), 
                              cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 0), 1);
                }
            }
        }
        
        // 绘制骨架（直接使用后处理输出的关键点坐标）
        for (int k = 0; k < 19; k++) {
            int kpt_a = skeleton[k * 2] - 1;
            int kpt_b = skeleton[k * 2 + 1] - 1;
            
            if (result->keypoints[kpt_a][2] > 0.5 && result->keypoints[kpt_b][2] > 0.5) {
                cv::line(img, cv::Point((int)result->keypoints[kpt_a][0], (int)result->keypoints[kpt_a][1]), 
                        cv::Point((int)result->keypoints[kpt_b][0], (int)result->keypoints[kpt_b][1]),
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
    
    // 转换为ByteTrack格式（post_process已经反算过，直接使用）
    std::vector<Object> objects;
    for (int i = 0; i < results->count; i++) {
        object_detect_result* result = &(results->results[i]);
        
        // post_process已经将坐标反算到原图空间，直接使用
        float x1 = result->box.left;
        float y1 = result->box.top;
        float x2 = result->box.right;
        float y2 = result->box.bottom;
        
        Object obj;
        obj.box = cv::Rect2f(x1, y1, x2-x1, y2-y1);
        obj.score = result->prop;
        obj.classId = 0; // person
        objects.push_back(obj);
    }
    
    // 执行跟踪
    auto tracks = g_byte_track.update(objects);
    
    // 绘制跟踪结果
    for (const auto& track : tracks) {
        cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色跟踪框
        cv::Rect2f rect(track.tlbr[0], track.tlbr[1], track.tlbr[2]-track.tlbr[0], track.tlbr[3]-track.tlbr[1]);
        cv::rectangle(img, rect, color, 2);
        
        char id_str[50];
        snprintf(id_str, sizeof(id_str), "ID:%d", track.track_id);
        cv::putText(img, id_str, cv::Point((int)track.tlbr[0], (int)track.tlbr[1]-20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
}

int main(int argc, char **argv) {
    int ret;
    rknn_app_context_t rknn_app_ctx;
    zero_copy_context_t zero_copy_ctx = {};
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    
    if (argc < 2) {
        printf("用法: %s <rknn模型路径> [标定文件路径] [摄像头路径]\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json\n", argv[0]);
        printf("示例: %s ../models/Q_yolov8_pose.rknn ../data/2025_7_11pm.json /dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0\n", argv[0]);
        return -1;
    }
    
    const char* model_path = argv[1];
    const char* calib_path = (argc > 2 && strlen(argv[2]) > 0) ? argv[2] : nullptr;
    const char* camera_path = (argc > 3 && strlen(argv[3]) > 0) ? argv[3] : "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0";
    
    // 检查设备路径是否存在，不存在则使用默认摄像头
    int camera_id = 0;  // 默认摄像头ID
    bool use_camera_path = false;
    
    if (camera_path && access(camera_path, F_OK) == 0) {
        use_camera_path = true;
    } else {
        use_camera_path = false;
    }
    
    // 设置信号处理
    signal(SIGINT, sig_handler);
    
    // 初始化后处理模块
    ret = init_post_process();
    if (ret != 0) {
        printf("后处理模块初始化失败！\n");
        return -1;
    }
    
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
    
    // 预先声明变量，避免goto跨越初始化
    cv::Mat frame;
    cv::Mat test_frame;  // 添加测试帧变量声明
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
    
    // 打开摄像头 - 强制使用V4L2后端
    if (use_camera_path) {
        cap.open(camera_path, cv::CAP_V4L2);
        if (!cap.isOpened()) {
            printf("❌ 无法打开USB摄像头: %s，尝试使用默认摄像头%d\n", camera_path, camera_id);
            cap.open(camera_id, cv::CAP_V4L2);
        }
    } else {
        cap.open(camera_id, cv::CAP_V4L2);
    }
    
    if (!cap.isOpened()) {
        printf("❌ 无法打开摄像头！\n");
        goto exit;
    }
    
    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    
    // 读取第一帧用于初始化letterbox上下文
    if (!cap.read(test_frame)) {
        printf("❌ 无法读取测试帧！\n");
        goto exit;
    }
    
    // 初始化letterbox上下文 - 关闭调试模式
    init_letterbox_context(&zero_copy_ctx.letterbox_ctx, 
                          test_frame.cols, test_frame.rows,  // 原始图像尺寸
                          rknn_app_ctx.model_width, rknn_app_ctx.model_height,  // 模型输入尺寸
                          false);  // 关闭调试模式
    
    // 验证letterbox变换准确性
    if (validate_letterbox_transform(&zero_copy_ctx.letterbox_ctx) != 0) {
        printf("警告: letterbox变换验证失败，可能影响坐标精度\n");
    }
    
    start_time = std::chrono::high_resolution_clock::now();
    
    
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
        
        // 获取输出 - 使用工作版本的方式
        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < rknn_app_ctx.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!rknn_app_ctx.is_quant);
        }
        ret = rknn_outputs_get(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("获取输出失败! ret=%d\n", ret);
            continue;
        }
        
        // 后处理 - 使用正确的letterbox参数
        object_detect_result_list pose_results;
        
        // 构建正确的letterbox参数
        letterbox_t letterbox;
        letterbox.x_pad = zero_copy_ctx.letterbox_ctx.offset_x;
        letterbox.y_pad = zero_copy_ctx.letterbox_ctx.offset_y;
        letterbox.scale = zero_copy_ctx.letterbox_ctx.scale;
        
        ret = post_process(&rknn_app_ctx, outputs, &letterbox, 0.5, 0.4, &pose_results);
        
        // 释放输出
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);
        if (ret != 0) {
            printf("后处理失败！\n");
            continue;
        }
        
        // ROI调试：每30帧显示一次ROI和坐标映射信息
        if (frame_count % 30 == 1 && pose_results.count > 0) {
            object_detect_result* first_result = &pose_results.results[0];
            cv::Rect bbox(first_result->box.left, first_result->box.top,
                         first_result->box.right - first_result->box.left,
                         first_result->box.bottom - first_result->box.top);
            cv::Point2f roi_center((bbox.x + bbox.x + bbox.width) * 0.5f, bbox.y + bbox.height);
            
            printf("ROI调试 - 检测目标:%d, ROI中心:(%.1f,%.1f)", pose_results.count, roi_center.x, roi_center.y);
            
            if (g_camera_mapping.is_initialized) {
                cv::Point2f world_coord = image_to_world_coordinate(roi_center, &g_camera_mapping);
                printf(", 世界坐标:(%.0f,%.0f)mm", world_coord.x, world_coord.y);
            }
            printf("\n");
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
    
    // 清理后处理模块
    deinit_post_process();
    
    cv::destroyAllWindows();
    printf("程序退出\n");
    return 0;
}