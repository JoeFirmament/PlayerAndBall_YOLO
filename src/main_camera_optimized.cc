/*-------------------------------------------
                性能优化版本 - NPU零拷贝 Bquill 2025—5-16
                
用途: 消除CPU↔NPU内存拷贝开销，提升推理性能
原理: 直接在NPU内存中进行预处理，避免数据传输
优化效果: 相比基础版本性能提升100% (比较main.camera.cc)
架构设计:
- 主线程: 使用NPU1进行姿态检测，支持零拷贝优化
- 副线程: 使用NPU2进行篮球检测，独立运行避免资源竞争
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <signal.h>
#include <memory>
#include <chrono>
#include <thread>              // 用于多线程
#include <queue>               // 用于线程安全队列
#include <mutex>               // 用于互斥锁
#include <condition_variable>  // 用于条件变量
#include <opencv2/opencv.hpp>
#include "pose_yolov8.h"
#include "rim_basketball_postprocess.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "BYTETracker.h"
#include "im2d.h"
#include "im2d_type.h"
#include "im2d_single.h"
#include "RgaUtils.h"
#include "pose_letterbox_utils.h"

int skeleton[38] ={16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}; 

// 全局变量，用于信号处理
bool g_running = true;

// 全局变量，用于控制ByteTrack功能开关
bool g_enable_tracking = true;

// 全局变量，用于控制篮球检测功能开关
bool g_enable_basketball_detection = true;

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

// 添加相机标定和Homography相关的结构体
// 设计原因: 将图像坐标转换为真实世界坐标，用于测量球员在场地上的实际位置
// 应用场景: 篮球战术分析需要知道球员的真实位置，而不是像素坐标
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

// 在全局变量部分添加ByteTrack实例
//多目标跟踪需要在连续帧之间保持目标身份
BYTETracker g_byte_track;

// ----------------------
// 篮球检测结果结构体及队列
// 设计原因: 篮球检测在独立线程中运行，需要线程安全的队列传递结果到主线程
// 异步处理: 避免篮球检测阻塞主线程的姿态检测，提升整体性能
// ----------------------
struct BasketballDetectResult {
    float x, y, w, h;      // 框坐标（模型输入尺寸下）
    float conf;            // 置信度
    int64_t timestamp;     // 检测时间戳
};
// 线程安全队列，用于传递篮球检测结果
std::queue<BasketballDetectResult> basketball_result_queue;
std::mutex basketball_result_mutex;
std::condition_variable basketball_result_cv;

// ----------------------
// 篮球检测线程相关全局变量
// 设计原因: 双线程架构 - 主线程处理姿态检测，副线程处理篮球检测
// 性能优化: 利用RK3588的双NPU核心，实现真正的并行处理
// ----------------------
// 线程安全队列，用于传递resize后的图像
std::queue<cv::Mat> basketball_frame_queue;
std::mutex basketball_queue_mutex;
std::condition_variable basketball_queue_cv;
bool basketball_thread_running = true;

// 篮球检测线程句柄
std::thread basketball_detect_thread;

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
// 设计原因: 传统推理需要CPU内存→NPU内存的数据拷贝，这是性能瓶颈
// 零拷贝方案: NPU和CPU共享内存区域，OpenCV直接在NPU内存上操作，消除拷贝开销
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
    // 设计原因: NPU内存有对齐要求，确保正确访问
    zc_ctx->input_mem = rknn_create_mem(app_ctx->rknn_ctx, zc_ctx->input_attr.size_with_stride);
    if (!zc_ctx->input_mem) {
        printf("创建输入零拷贝内存失败！\n");
        return -1;
    }
    
    // 将NPU内存绑定到推理上下文
    // 设计原因: 告诉NPU直接从这块内存读取数据，无需拷贝
    ret = rknn_set_io_mem(app_ctx->rknn_ctx, zc_ctx->input_mem, &zc_ctx->input_attr);
    if (ret < 0) {
        printf("设置输入零拷贝内存失败! ret=%d\n", ret);
        return -1;
    }
    
    // 创建输出零拷贝内存
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        zc_ctx->output_attrs[i] = app_ctx->output_attrs[i];
        zc_ctx->output_mems[i] = NULL;
    }
    
    return 0;
}

// 释放零拷贝内存
static void release_zero_copy_mem(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx) {
    if (zc_ctx->input_mem) {
        rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->input_mem);
        zc_ctx->input_mem = NULL;
    }
    
    // 暂时不使用输出零拷贝，所以不需要释放
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (zc_ctx->output_mems[i]) {
            rknn_destroy_mem(app_ctx->rknn_ctx, zc_ctx->output_mems[i]);
            zc_ctx->output_mems[i] = NULL;
        }
    }
}

// 零拷贝优化的letterbox预处理，直接写入NPU内存
static int optimized_letterbox_to_npu(cv::Mat& src_mat, zero_copy_context_t* zc_ctx) {
    // 构造零拷贝letterbox上下文
    zero_copy_letterbox_context_t letterbox_zc_ctx;
    letterbox_zc_ctx.input_mem = zc_ctx->input_mem;
    letterbox_zc_ctx.input_attr = zc_ctx->input_attr;
    letterbox_zc_ctx.model_width = zc_ctx->model_width;
    letterbox_zc_ctx.model_height = zc_ctx->model_height;
    letterbox_zc_ctx.model_channels = zc_ctx->model_channels;
    letterbox_zc_ctx.letterbox_ctx = zc_ctx->letterbox_ctx;
    
    // 使用letterbox工具函数进行预处理
    return zero_copy_letterbox_preprocess(src_mat, &letterbox_zc_ctx);
}

// 零拷贝推理和后处理
// 整合推理和后处理流程，减少函数调用开销
static int zero_copy_inference_and_postprocess(rknn_app_context_t* app_ctx, zero_copy_context_t* zc_ctx, 
        object_detect_result_list* od_results, std::vector<Object>& objects) {
    int ret;
    
    // 设置输入
    rknn_input input;
    input.index = 0;
    input.buf = zc_ctx->input_mem->virt_addr;
    input.size = zc_ctx->input_attr.size_with_stride;
    input.pass_through = 1; // 直通模式，所有预处理都在CPU侧完成
    input.type = zc_ctx->input_attr.type; // 与模型输入类型一致
    input.fmt = zc_ctx->input_attr.fmt;   // 与模型输入格式一致
    rknn_mem_sync(app_ctx->rknn_ctx, zc_ctx->input_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, &input);
    if (ret < 0) {
        printf("rknn_inputs_set 失败! ret=%d\n", ret);
        return -1;
    }
    
    // 推理
    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0) {
        printf("rknn_run 失败! ret=%d\n", ret);
        return -1;
    }
    
    // 获取输出
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = (!app_ctx->is_quant);
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get 失败! ret=%d\n", ret);
        return -1;
    }
    
    // 后处理
    letterbox_t letter_box = {0, 0, 1.0f}; // 零拷贝模式下的默认letterbox
    ret = post_process(app_ctx, outputs, &letter_box, 0.5, 0.4, od_results);
    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    
    // 设计原因: 生成BYTETracker所需objects（使用letterbox工具函数进行反变换）
    objects.clear();
    for (int i = 0; i < od_results->count; i++) {
        object_detect_result* det_result = &od_results->results[i];
        Object obj;
        
        // 使用letterbox工具函数进行边界框反变换
        cv::Rect model_box(det_result->box.left, det_result->box.top,
                          det_result->box.right - det_result->box.left,
                          det_result->box.bottom - det_result->box.top);
        cv::Rect original_box = letterbox_inverse_transform_bbox(model_box, &zc_ctx->letterbox_ctx);
        
        obj.box.x = original_box.x;
        obj.box.y = original_box.y;
        obj.box.width = original_box.width;
        obj.box.height = original_box.height;
        obj.score = det_result->prop;
        obj.classId = det_result->cls_id;
        objects.push_back(obj);
    }
    
    return ret;
}

// 初始化Homography（使用指定的JSON文件路径）
static int init_camera_mapping(const char* homo_json_path) {
    // printf("加载指定的Homography JSON文件: %s\n", homo_json_path);

    // 设计原因: 检查文件是否存在
    FILE* check_file = fopen(homo_json_path, "r");
    if (!check_file) {
        printf("❌ 错误: Homography JSON文件不存在: %s\n", homo_json_path);
        printf("请检查文件路径或创建标定文件\n");
        return -1;
    }
    fclose(check_file);
    
    //读取JSON文件
    cv::FileStorage fs_homo(homo_json_path, cv::FileStorage::READ);
    if (!fs_homo.isOpened()) {
        printf("❌ 错误: 无法打开Homography JSON文件 %s\n", homo_json_path);
        return -1;
    }

    // 设计原因: 从JSON读取matrix数组并转换为3x3矩阵
    cv::FileNode homo_node = fs_homo["matrix"];
    if (homo_node.empty() || !homo_node.isSeq()) {
        printf("错误: JSON文件中缺少matrix数组\n");
        return -1;
    }

    if (homo_node.size() != 3) {
        printf("错误: matrix数组应为3x3矩阵，实际为%zu行\n", homo_node.size());
        return -1;
    }

    // 设计原因: 创建3x3矩阵并填充数据
    g_camera_mapping.homography = cv::Mat::zeros(3, 3, CV_64F);
    int row = 0;
    for (cv::FileNodeIterator it = homo_node.begin(); it != homo_node.end(); ++it, ++row) {
        cv::FileNode row_node = *it;
        if (!row_node.isSeq() || row_node.size() != 3) {
            printf("错误: matrix第%d行应包含3个元素，实际为%zu个\n", row, row_node.size());
            return -1;
        }
        
        int col = 0;
        for (cv::FileNodeIterator col_it = row_node.begin(); col_it != row_node.end(); ++col_it, ++col) {
            g_camera_mapping.homography.at<double>(row, col) = (double)*col_it;
        }
    }

    std::cout << "[DEBUG] homography_matrix (从JSON读取): " << g_camera_mapping.homography << std::endl;
    std::cout << "[DEBUG] homography_matrix size: " << g_camera_mapping.homography.rows << "x" << g_camera_mapping.homography.cols << std::endl;

    fs_homo.release();

    // 设计原因: 验证Homography矩阵维度
    if (g_camera_mapping.homography.rows != 3 || g_camera_mapping.homography.cols != 3) {
        printf("错误: Homography矩阵维度不正确\n");
        return -1;
    }

    // 设计原因: 不再需要相机内参和畸变系数，直接设置为空
    g_camera_mapping.camera_matrix = cv::Mat();
    g_camera_mapping.dist_coeffs = cv::Mat();
    g_camera_mapping.calib_width = 0;
    g_camera_mapping.calib_height = 0;

    g_camera_mapping.is_initialized = true;
    // printf("Homography初始化成功（仅使用JSON文件，无相机标定）\n");
    return 0;
}

// 根据实际分辨率缩放相机内参矩阵
static void scale_camera_matrix(cv::Mat& camera_matrix, 
                              int calib_width, int calib_height,
                              int actual_width, int actual_height) {
    double scale_x = (double)actual_width / (double)calib_width;
    double scale_y = (double)actual_height / (double)calib_height;
    // 设计原因: 只缩放fx, fy, cx, cy，不重复缩放
    camera_matrix.at<double>(0,0) = camera_matrix.at<double>(0,0) * scale_x;  // fx
    camera_matrix.at<double>(1,1) = camera_matrix.at<double>(1,1) * scale_y;  // fy
    camera_matrix.at<double>(0,2) = camera_matrix.at<double>(0,2) * scale_x;  // cx
    camera_matrix.at<double>(1,2) = camera_matrix.at<double>(1,2) * scale_y;  // cy
}

// 结构体定义：保存两种方法的结果
struct FootPositionResult {
    cv::Point2f roi_method;        // ROI下边缘中点方法
    cv::Point2f ankle_method;      // 脚踝+偏移量方法
    bool use_keypoints;            // 是否使用关键点
    float vertical_offset;         // 垂直偏移量
};

// 平滑算法：使用滑动平均窗口过滤抖动
static std::map<int, std::deque<cv::Point2f>> roi_position_history;     // ROI方法历史记录
static std::map<int, std::deque<cv::Point2f>> ankle_position_history;   // 脚踝方法历史记录
static const int SMOOTH_WINDOW_SIZE = 8;  // 平滑窗口大小 - 增加以减少抖动

// 计算俯视角度下的脚部地面接触点 - 返回两种方法的结果
static FootPositionResult calculate_foot_position_comparison(const float keypoints[17][2], cv::Mat& debug_frame, const object_detect_result* det_result, const letterbox_context_t* letterbox_ctx, int track_id) {
    FootPositionResult result;
    
    // 设计原因: 计算ROI包围框（使用letterbox工具函数进行反变换）
    cv::Rect model_box(det_result->box.left, det_result->box.top,
                      det_result->box.right - det_result->box.left,
                      det_result->box.bottom - det_result->box.top);
    cv::Rect original_box = letterbox_inverse_transform_bbox(model_box, letterbox_ctx);
    cv::Point2f roi_bottom_center((original_box.x + original_box.x + original_box.width) * 0.5f, 
                                 original_box.y + original_box.height);
    
    // 设计原因: 获取关键点（COCO格式）
    cv::Point2f nose(keypoints[0][0], keypoints[0][1]);           // 鼻子
    cv::Point2f left_ankle(keypoints[15][0], keypoints[15][1]);   // 左脚踝
    cv::Point2f right_ankle(keypoints[16][0], keypoints[16][1]);  // 右脚踝
    
    // 检查关键点置信度
    float left_conf = det_result->keypoints[15][2];
    float right_conf = det_result->keypoints[16][2];
    float nose_conf = det_result->keypoints[0][2];
    
    cv::Point2f ankle_center;
    bool use_keypoints = false;
    
    if (left_conf > 0.3 && right_conf > 0.3) {
        // 两个脚踝都可见，使用中点
        ankle_center = cv::Point2f((left_ankle.x + right_ankle.x) * 0.5f, (left_ankle.y + right_ankle.y) * 0.5f);
        use_keypoints = true;
    } else if (left_conf > 0.3) {
        ankle_center = left_ankle;
        use_keypoints = true;
    } else if (right_conf > 0.3) {
        ankle_center = right_ankle;
        use_keypoints = true;
    } else {
        // 脚踝不可见，使用ROI下边缘
        ankle_center = roi_bottom_center;
        use_keypoints = false;
    }
    
    // 方法1：ROI下边缘中点方法（原始方法）
    cv::Point2f roi_method_result = roi_bottom_center;
    
    // 方法2：脚踝+偏移量方法
    cv::Point2f ankle_method_result = ankle_center;
    float vertical_offset = 0.0f;
    
    // 设计原因: 俯视角度下的动态垂直偏移量计算
    if (use_keypoints && nose_conf > 0.3) {
        // 计算人物高度（鼻子到脚踝的距离）
        float person_height_pixels = std::abs(nose.y - ankle_center.y);
        
        // 根据人物高度动态计算偏移量
        // 俯视角度：人越高，脚部偏移量越大
        float height_ratio = person_height_pixels / 640.0f;  // 标准化到模型输入尺寸
        vertical_offset = height_ratio * 25.0f;  // 基础偏移量25像素
        
        // 限制偏移量范围（防止过度修正）
        vertical_offset = std::max(5.0f, std::min(vertical_offset, 40.0f));
        
        // 应用垂直偏移
        ankle_method_result.y += vertical_offset;
    }
    
    // 设计原因: 使用track_id进行平滑
    
    // 方法1的平滑
    roi_position_history[track_id].push_back(roi_method_result);
    if (roi_position_history[track_id].size() > SMOOTH_WINDOW_SIZE) {
        roi_position_history[track_id].pop_front();
    }
    
    cv::Point2f smoothed_roi(0, 0);
    int roi_count = roi_position_history[track_id].size();
    for (const auto& pos : roi_position_history[track_id]) {
        smoothed_roi.x += pos.x;
        smoothed_roi.y += pos.y;
    }
    smoothed_roi.x /= roi_count;
    smoothed_roi.y /= roi_count;
    
    // 方法2的平滑
    ankle_position_history[track_id].push_back(ankle_method_result);
    if (ankle_position_history[track_id].size() > SMOOTH_WINDOW_SIZE) {
        ankle_position_history[track_id].pop_front();
    }
    
    cv::Point2f smoothed_ankle(0, 0);
    int ankle_count = ankle_position_history[track_id].size();
    for (const auto& pos : ankle_position_history[track_id]) {
        smoothed_ankle.x += pos.x;
        smoothed_ankle.y += pos.y;
    }
    smoothed_ankle.x /= ankle_count;
    smoothed_ankle.y /= ankle_count;
    
    // 设置返回结果
    result.roi_method = smoothed_roi;
    result.ankle_method = smoothed_ankle;
    result.use_keypoints = use_keypoints;
    result.vertical_offset = vertical_offset;
    
    // 设计原因: 简化调试信息，只显示关键点
    // cv::rectangle(debug_frame, cv::Point(bbox_left, bbox_top), cv::Point(bbox_right, bbox_bottom), cv::Scalar(128, 128, 128), 1);
    
    // 只绘制脚踝关键点
    if (left_conf > 0.3) {
        cv::circle(debug_frame, left_ankle, 2, cv::Scalar(0, 0, 255), -1);  // 红色：左脚踝
    }
    if (right_conf > 0.3) {
        cv::circle(debug_frame, right_ankle, 2, cv::Scalar(255, 0, 0), -1); // 蓝色：右脚踝
    }
    
    // 绘制两种方法的结果比较（缩小圆圈）
    cv::circle(debug_frame, smoothed_roi, 4, cv::Scalar(255, 0, 255), 1);        // 紫色：ROI方法
    cv::circle(debug_frame, smoothed_ankle, 4, cv::Scalar(0, 255, 0), 1);        // 绿色：脚踝方法
    
    // 简化连接线
    cv::line(debug_frame, smoothed_roi, smoothed_ankle, cv::Scalar(128, 128, 128), 1);
    
    return result;
}

// 将图像坐标映射到地面坐标
static cv::Point2f map_to_ground(const cv::Point2f& image_point, cv::Mat& debug_frame) {
    if (!g_camera_mapping.is_initialized) {
        printf("[map_to_ground] ❌ 警告: Homography未初始化，返回(-1,-1)\n");
        return cv::Point2f(-1, -1);
    }

    std::vector<cv::Point2f> src_points = {image_point};
    std::vector<cv::Point2f> dst_points;
    cv::perspectiveTransform(src_points, dst_points, g_camera_mapping.homography);
    
    // 设计原因: 在调试图像上绘制映射关系
    cv::Point2f ground_point = dst_points[0];
    if (ground_point.x >= 0 && ground_point.y >= 0) {
        // 设计原因: 绘制从图像点到地面点的映射线
        cv::line(debug_frame, image_point, 
                cv::Point(image_point.x, image_point.y + 50), 
                cv::Scalar(0, 255, 255), 2);
                
        // 设计原因: 显示地面坐标（毫米）
        char coord_text[128];
        sprintf(coord_text, "Ground: (%.0f,%.0f)mm", ground_point.x, ground_point.y);
        cv::putText(debug_frame, coord_text,
                   cv::Point(image_point.x + 10, image_point.y + 70),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    }
    
    return ground_point;
}

// 计算两点之间的欧氏距离（单位：毫米）
static float calculate_distance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return std::sqrt(dx*dx + dy*dy);
}

// ----------------------
// 篮球检测线程函数   后添加用来测试
// 设计原因: 双线程架构 - 主线程处理姿态检测，副线程处理篮球检测
// 性能优化: 利用RK3588的多NPU核心，实现真正的并行处理，以后再添加进球模型检测
// ----------------------
void basketball_detect_thread_func() {
    // 设计原因: 1. 加载我们优化的篮球+球员检测模型
    const char* basketball_model_path = "/home/radxa/Qworkspace/Qrknn/rknpu2/examples/yolov8/model/Q_Player_Ball_8n_4090_Drun_500E.rknn";
    rknn_app_context_t basketball_ctx;
    memset(&basketball_ctx, 0, sizeof(rknn_app_context_t));
    
    int ret = 0;
    // 设计原因: 先初始化模型（使用姿态检测初始化函数，但模型是我们的2类别模型）
    ret = init_yolov8_pose_model(basketball_model_path, &basketball_ctx);
    if (ret != 0) {
        printf("[basketball][error] 模型加载失败! 路径: %s\n", basketball_model_path);
        return;
    }
    
    // 设计原因: 设置NPU核心为NPU2，确保与主线程分开运行
    rknn_core_mask core_mask = RKNN_NPU_CORE_2;
    ret = rknn_set_core_mask(basketball_ctx.rknn_ctx, core_mask);
    if (ret != 0) {
        // printf("[basketball][error] rknn_set_core_mask 失败! ret=%d\n", ret);
    }

    // 推理主循环
    while (basketball_thread_running) {
        cv::Mat input_img;
        {
            std::unique_lock<std::mutex> lock(basketball_queue_mutex);
            basketball_queue_cv.wait(lock, []{ return !basketball_frame_queue.empty() || !basketball_thread_running; });
            if (!basketball_thread_running) break;
            input_img = basketball_frame_queue.front();
            basketball_frame_queue.pop();
        }
        
        // 输入图像已经是letterbox处理后的640x640，只需BGR->RGB转换
        cv::Mat rgb_img;
        cv::cvtColor(input_img, rgb_img, cv::COLOR_BGR2RGB);
        
        //直接构造输入（无需再次letterbox）
        rknn_input input;
        input.index = 0;
        input.buf = rgb_img.data;
        input.size = basketball_ctx.model_width * basketball_ctx.model_height * 3;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.pass_through = 0;
        ret = rknn_inputs_set(basketball_ctx.rknn_ctx, 1, &input);
        if (ret < 0) {
            // printf("[basketball][error] rknn_inputs_set 失败! ret=%d\n", ret);
            continue;
        }
        
        // 设计原因: 2.4 推理
        ret = rknn_run(basketball_ctx.rknn_ctx, NULL);
        if (ret < 0) {
            // printf("[basketball][error] rknn_run 失败! ret=%d\n", ret);
            continue;
        }
        
        // 设计原因: 2.5 使用与成功程序相同的后处理方式
        rknn_output outputs[basketball_ctx.io_num.n_output];
        memset(outputs, 0, sizeof(outputs));
        for (int i = 0; i < basketball_ctx.io_num.n_output; i++) {
            outputs[i].index = i;
            outputs[i].want_float = (!basketball_ctx.is_quant);
        }
        ret = rknn_outputs_get(basketball_ctx.rknn_ctx, basketball_ctx.io_num.n_output, outputs, NULL);
        if (ret < 0) {
            // printf("[basketball][error] rknn_outputs_get 失败! ret=%d\n", ret);
            continue;
        }
        
        // 使用专门为篮球检测适配的后处理方式  
        float conf_threshold = 0.5;  // 设计原因: 降低置信度阈值，提高检测率
        BasketballDetectionResult basketball_result;
        
        void* output_buffers[9];
        int output_dims[36];  // 9 outputs * 4 dimensions each
        int32_t output_zps[9];
        float output_scales[9];
        
        for (int i = 0; i < basketball_ctx.io_num.n_output; i++) {
            output_buffers[i] = outputs[i].buf;
            output_dims[i*4 + 0] = basketball_ctx.output_attrs[i].dims[0];
            output_dims[i*4 + 1] = basketball_ctx.output_attrs[i].dims[1];
            output_dims[i*4 + 2] = basketball_ctx.output_attrs[i].dims[2];
            output_dims[i*4 + 3] = basketball_ctx.output_attrs[i].dims[3];
            output_zps[i] = basketball_ctx.output_attrs[i].zp;
            output_scales[i] = basketball_ctx.output_attrs[i].scale;
        }
        
        // 设计原因: 修正DFL长度：第一个输出(回归分支)的通道数/4
        int dfl_len = basketball_ctx.output_attrs[0].dims[1] / 4;
        output_dims[0] = dfl_len;  // 将DFL长度放在output_dims[0]位置
        
        // 设计原因: 调用专门的篮球检测后处理函数
        ret = process_basketball_yolov8_output(
            output_buffers, output_dims, output_zps, output_scales,
            basketball_ctx.io_num.n_output, conf_threshold, &basketball_result
        );
        
        if (ret == 0) {
            // 设计原因: 将检测结果推送到主线程队列
            for (int i = 0; i < basketball_result.count; i++) {
                BasketballDetection* det = &basketball_result.detections[i];
                
                // 设计原因: 只推送篮球检测结果（类别ID为1）
                if (det->class_id == 1) {  // basketball
                    BasketballDetectResult queue_result;
                    queue_result.x = det->x;
                    queue_result.y = det->y;
                    queue_result.w = det->w;
                    queue_result.h = det->h;
                    queue_result.conf = det->confidence;
                    queue_result.timestamp = getCurrentTimeUs();
                    
                    {
                        std::lock_guard<std::mutex> lock(basketball_result_mutex);
                        basketball_result_queue.push(queue_result);
                        while (basketball_result_queue.size() > 10) basketball_result_queue.pop();
                    }
                    basketball_result_cv.notify_one();
                }
            }
        } else {
            // printf("[basketball][error] 篮球检测后处理失败! ret=%d\n", ret);
        }
        
        rknn_outputs_release(basketball_ctx.rknn_ctx, basketball_ctx.io_num.n_output, outputs);
    }
    release_yolov8_pose_model(&basketball_ctx);
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    printf("=== YOLOv8 Pose Detection===\n");
    printf("平台: Rock5C 8GB, CPU: aarch64, NPU: RK3588\n");
    printf("NPU零拷贝 + 直接内存访问\n\n");
    
    if (argc < 2 || argc > 3)
    {
        printf("用法: %s <model_path> [homography_json_path]\n", argv[0]);
        printf("参数说明:\n");
        printf("  model_path: YOLOv8模型文件路径\n");
        printf("  homography_json_path: Homography标定文件路径（可选，默认为./data/2025_7_11pm.json）\n");
        printf("示例:\n");
        printf("  %s model.rknn\n", argv[0]);
        printf("  %s model.rknn ./data/my_calibration.json\n", argv[0]);
        return -1;
    }

    const char *model_path = argv[1];
    const char *homo_json_path = (argc >= 3) ? argv[2] : "./data/2025_7_11pm.json";
    
    printf("模型路径: %s\n", model_path);
    printf("Homography文件路径: %s\n", homo_json_path);
    
    int ret;
    rknn_app_context_t rknn_app_ctx;
    zero_copy_context_t zero_copy_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    memset(&zero_copy_ctx, 0, sizeof(zero_copy_context_t));

    // 设计原因: 注册信号处理函数
    signal(SIGINT, sig_handler);

    // 设计原因: 初始化后处理
    init_post_process();

    // 设计原因: 初始化模型
    ret = init_yolov8_pose_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("模型初始化失败! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    printf("模型初始化成功, 输入尺寸: %dx%d\n", 
           rknn_app_ctx.model_width, rknn_app_ctx.model_height);

    // 设计原因: 初始化零拷贝内存
    ret = init_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    if (ret != 0) {
        printf("零拷贝内存初始化失败!\n");
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 设计原因: 初始化Homography
    printf("\n=== 初始化Homography ===\n");
    ret = init_camera_mapping(homo_json_path);
    if (ret != 0) {
        printf("错误: Homography初始化失败!\n");
        printf("请确保JSON文件位于正确位置，且格式正确\n");
        return -1;
    }
    printf("=== Homography初始化完成 ===\n\n");

    // 设计原因: 打开摄像头
//注意⚠️：这里需要使用cv::CAP_V4L2，否则会默认调用 GStreamer，导致无法正常采集帧率。
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        printf("摄像头打开失败\n");
        release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
        release_yolov8_pose_model(&rknn_app_ctx);
        deinit_post_process();
        return -1;
    }

    // 设计原因: 严格设置MJPG格式和1920x1080分辨率
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    usleep(200*1000); // 设计原因: 延时200ms，部分驱动需要

    // 设计原因: 获取实际的摄像头参数
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    printf("摄像头参数: %dx%d @ %.1fFPS, 格式: MJPEG\n", actual_width, actual_height, actual_fps);

    // 设计原因: 采集100帧并统计FPS（与fps_test.cc一致）
    cv::Mat fps_test_frame;
    int fps_count = 0;
    int fps_total = 100;
    auto fps_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < fps_total; ++i) {
        if (!cap.read(fps_test_frame)) {
            printf("采集失败\n");
            break;
        }
        fps_count++;
    }
    auto fps_end = std::chrono::high_resolution_clock::now();
    double fps_seconds = std::chrono::duration<double>(fps_end - fps_start).count();
    printf("主程序采集帧率: %.2f FPS (共%d帧, %.2f秒)\n", fps_count / fps_seconds, fps_count, fps_seconds);
    printf("最后一帧尺寸: %dx%d\n", fps_test_frame.cols, fps_test_frame.rows);

    // 设计原因: 验证摄像头是否正常工作
    cv::Mat test_frame;
    if (!cap.read(test_frame)) {
        printf("错误: 无法从摄像头读取测试帧!\n");
        return -1;
    }
    printf("采集帧尺寸: %dx%d\n", test_frame.cols, test_frame.rows);
    printf("cap.get尺寸: %dx%d\n", (int)cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // 设计原因: 初始化letterbox上下文
    init_letterbox_context(&zero_copy_ctx.letterbox_ctx, 
                          test_frame.cols, test_frame.rows,  // 原始图像尺寸
                          rknn_app_ctx.model_width, rknn_app_ctx.model_height,  // 模型输入尺寸
                          true);  // 启用调试模式
    
    // 验证letterbox变换准确性
    if (validate_letterbox_transform(&zero_copy_ctx.letterbox_ctx) != 0) {
        printf("警告: letterbox变换验证失败，可能影响坐标精度\n");
    }
    
    // 设计原因: 创建窗口并验证
    const char* WINDOW_NAME = "YOLOv8 Pose Basketball";
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 640, 480);
    
    // 设计原因: 显示测试图像
    cv::imshow(WINDOW_NAME, test_frame);
    printf("显示测试帧...\n");
    cv::waitKey(1000); // 设计原因: 等待1秒
    printf("测试帧显示完成\n");

    // 设计原因: 性能统计变量
    int frame_count = 0;
    int64_t total_time = 0;
    int64_t start_time_overall = getCurrentTimeUs();

    // 设计原因: 移除相关的静态变量，直接使用原始图像

    // 设计原因: 启动篮球检测线程
    basketball_thread_running = true;
    basketball_detect_thread = std::thread(basketball_detect_thread_func);

    // 设计原因: 主循环
    while (g_running) {
        int64_t t0 = getCurrentTimeUs();
        cv::Mat frame;
        // 设计原因: 采集
        cap.read(frame);
        int64_t t1 = getCurrentTimeUs();
        
        // 设计原因: 直接使用原始图像
        cv::Mat yolo_input_cv;
        int yolo_width = zero_copy_ctx.model_width;
        int yolo_height = zero_copy_ctx.model_height;
        
        // 设计原因: 使用letterbox工具函数进行预处理
        int64_t t2_start = getCurrentTimeUs();
        ret = optimized_letterbox_to_npu(frame, &zero_copy_ctx);
        int64_t t2 = getCurrentTimeUs();
        if (ret != 0) {
            printf("预处理失败\n");
            continue;
        }
        
        // 设计原因: 零拷贝推理
        object_detect_result_list od_results;
        std::vector<Object> objects;
        int64_t t3 = getCurrentTimeUs();
        
        ret = zero_copy_inference_and_postprocess(&rknn_app_ctx, &zero_copy_ctx, &od_results, objects);
        int64_t t4 = getCurrentTimeUs();
        if (ret != 0) {
            printf("推理失败\n");
            continue;
        }
        // 设计原因: 跟踪（可通过t键开关）
        int64_t t_track0 = getCurrentTimeUs();
        std::vector<STrack> tracks;
        if (g_enable_tracking) {
            tracks = g_byte_track.update(objects);
        }
        int64_t t_track1 = getCurrentTimeUs();
        
        // 设计原因: 绘制结果 - 结合姿态检测与跟踪
        cv::Mat result_frame = frame.clone();
        
        if (g_enable_tracking && !tracks.empty()) {
            // 跟踪模式：建立track与检测结果的对应关系
            std::map<int, int> track_to_detection;  // track_index -> detection_index
            for (int track_idx = 0; track_idx < tracks.size(); track_idx++) {
            const STrack& track = tracks[track_idx];
            cv::Rect_<float> track_rect(track.tlbr[0], track.tlbr[1], 
                                       track.tlbr[2] - track.tlbr[0], 
                                       track.tlbr[3] - track.tlbr[1]);
            
            // 寻找最匹配的检测结果
            float best_iou = 0.0f;
            int best_detection_idx = -1;
            
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result *det_result = &(od_results.results[i]);
                
                // 使用letterbox工具函数进行边界框反变换
                cv::Rect model_box(det_result->box.left, det_result->box.top,
                                  det_result->box.right - det_result->box.left,
                                  det_result->box.bottom - det_result->box.top);
                cv::Rect original_box = letterbox_inverse_transform_bbox(model_box, &zero_copy_ctx.letterbox_ctx);
                cv::Rect_<float> det_rect(original_box.x, original_box.y, original_box.width, original_box.height);
                
                // 计算IoU
                float intersection_area = (track_rect & det_rect).area();
                float union_area = track_rect.area() + det_rect.area() - intersection_area;
                float iou = (union_area > 0) ? intersection_area / union_area : 0.0f;
                
                if (iou > best_iou) {
                    best_iou = iou;
                    best_detection_idx = i;
                }
            }
            
            if (best_iou > 0.3) {  // IoU阈值
                track_to_detection[track_idx] = best_detection_idx;
            }
        }
        
        // 绘制跟踪结果和姿态检测结果
        for (int track_idx = 0; track_idx < tracks.size(); track_idx++) {
            const STrack& track = tracks[track_idx];
            int track_id = track.track_id;
            
            cv::Rect_<float> rect(track.tlbr[0], track.tlbr[1],
                                 track.tlbr[2] - track.tlbr[0],
                                 track.tlbr[3] - track.tlbr[1]);
            
            // 绘制跟踪框和ID
            cv::rectangle(result_frame, rect, cv::Scalar(0,255,0), 2);
            char id_text[32];
            sprintf(id_text, "ID:%d", track_id);
            cv::putText(result_frame, id_text, cv::Point(rect.x, rect.y-5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
            
            // 如果有对应的姿态检测结果，绘制姿态和计算地面坐标
            if (track_to_detection.find(track_idx) != track_to_detection.end()) {
                int det_idx = track_to_detection[track_idx];
                object_detect_result *det_result = &(od_results.results[det_idx]);
                
                // 转换关键点坐标（使用letterbox工具函数进行反变换）
                float keypoints[17][2];
                letterbox_inverse_transform_keypoints(det_result->keypoints, keypoints, 17, &zero_copy_ctx.letterbox_ctx);
                
                // 绘制骨架
                for (int j = 0; j < 38/2; ++j) {
                    cv::line(result_frame, 
                        cv::Point((int)(keypoints[skeleton[2*j]-1][0]), (int)(keypoints[skeleton[2*j]-1][1])),
                        cv::Point((int)(keypoints[skeleton[2*j+1]-1][0]), (int)(keypoints[skeleton[2*j+1]-1][1])),
                        cv::Scalar(0, 128, 255), 2);
                }
                
                // 绘制关键点
                for (int j = 0; j < 17; ++j) {
                    cv::circle(result_frame, 
                        cv::Point((int)(keypoints[j][0]), (int)(keypoints[j][1])),
                        3, cv::Scalar(255, 255, 0), -1);
                }
                
                // 使用新的函数计算两种方法的脚部位置
                FootPositionResult foot_result = calculate_foot_position_comparison(
                    keypoints, result_frame, det_result, &zero_copy_ctx.letterbox_ctx, track_id);
                
                // 计算两种方法的地面坐标
                cv::Point2f ground_point_roi = map_to_ground(foot_result.roi_method, result_frame);
                cv::Point2f ground_point_ankle = map_to_ground(foot_result.ankle_method, result_frame);
                
                // 显示两种方法的比较结果
                int text_start_y = rect.y + rect.height + 20;
                
                // ROI下沿中点地面坐标（紫色）
                if (ground_point_roi.x >= 0 && ground_point_roi.y >= 0) {
                    char roi_text[256];
                    sprintf(roi_text, "ROI底部: (%.0f,%.0f)mm", ground_point_roi.x, ground_point_roi.y);
                    cv::putText(result_frame, roi_text, cv::Point(rect.x, text_start_y), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
                }
                
                // 脚踝+垂直拉伸量地面坐标（绿色）
                if (ground_point_ankle.x >= 0 && ground_point_ankle.y >= 0) {
                    char ankle_text[256];
                    sprintf(ankle_text, "脚踝+偏移: (%.0f,%.0f)mm", ground_point_ankle.x, ground_point_ankle.y);
                    cv::putText(result_frame, ankle_text, cv::Point(rect.x, text_start_y + 25), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
                
                // 显示动态垂直拉伸量
                if (foot_result.use_keypoints) {
                    char offset_text[128];
                    sprintf(offset_text, "垂直拉伸: %.1fpx", foot_result.vertical_offset);
                    cv::putText(result_frame, offset_text, cv::Point(rect.x, text_start_y + 50), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                }
                
                // 显示两种方法的地面距离差异
                float distance_diff = cv::norm(ground_point_roi - ground_point_ankle);
                char diff_text[128];
                sprintf(diff_text, "地面差距: %.1fmm", distance_diff);
                cv::putText(result_frame, diff_text, cv::Point(rect.x, text_start_y + 75), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }
        }
        } else {
            // 非跟踪模式：直接绘制检测结果
            for (int i = 0; i < od_results.count; i++) {
                object_detect_result *det_result = &(od_results.results[i]);
                
                // 绘制检测框（使用letterbox工具函数进行反变换）
                cv::Rect model_box(det_result->box.left, det_result->box.top,
                                  det_result->box.right - det_result->box.left,
                                  det_result->box.bottom - det_result->box.top);
                cv::Rect bbox = letterbox_inverse_transform_bbox(model_box, &zero_copy_ctx.letterbox_ctx);
                cv::rectangle(result_frame, bbox, cv::Scalar(255, 0, 0), 2);  // 蓝色框表示检测模式
                
                // 显示检测框标签
                char det_text[32];
                sprintf(det_text, "Det:%.2f", det_result->prop);
                cv::putText(result_frame, det_text, cv::Point(bbox.x, bbox.y-5), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
                
                // 转换关键点坐标（使用letterbox工具函数进行反变换）
                float keypoints[17][2];
                letterbox_inverse_transform_keypoints(det_result->keypoints, keypoints, 17, &zero_copy_ctx.letterbox_ctx);
                
                // 绘制骨架
                for (int j = 0; j < 38/2; ++j) {
                    cv::line(result_frame, 
                        cv::Point((int)(keypoints[skeleton[2*j]-1][0]), (int)(keypoints[skeleton[2*j]-1][1])),
                        cv::Point((int)(keypoints[skeleton[2*j+1]-1][0]), (int)(keypoints[skeleton[2*j+1]-1][1])),
                        cv::Scalar(0, 128, 255), 2);
                }
                
                // 绘制关键点
                for (int j = 0; j < 17; ++j) {
                    cv::circle(result_frame, 
                        cv::Point((int)(keypoints[j][0]), (int)(keypoints[j][1])),
                        3, cv::Scalar(255, 255, 0), -1);
                }
                
                // 计算并显示地面坐标（使用临时ID）
                FootPositionResult foot_result = calculate_foot_position_comparison(
                    keypoints, result_frame, det_result, &zero_copy_ctx.letterbox_ctx, i);  // 使用检测索引作为临时ID
                
                cv::Point2f ground_point_roi = map_to_ground(foot_result.roi_method, result_frame);
                cv::Point2f ground_point_ankle = map_to_ground(foot_result.ankle_method, result_frame);
                
                // 显示地面坐标和拉伸量信息
                int text_start_y = bbox.y + bbox.height + 20;
                
                // ROI下沿中点地面坐标（紫色）
                if (ground_point_roi.x >= 0 && ground_point_roi.y >= 0) {
                    char roi_text[256];
                    sprintf(roi_text, "ROI底部: (%.0f,%.0f)mm", ground_point_roi.x, ground_point_roi.y);
                    cv::putText(result_frame, roi_text, cv::Point(bbox.x, text_start_y), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
                }
                
                // 脚踝+垂直拉伸量地面坐标（绿色）
                if (ground_point_ankle.x >= 0 && ground_point_ankle.y >= 0) {
                    char ankle_text[256];
                    sprintf(ankle_text, "脚踝+偏移: (%.0f,%.0f)mm", ground_point_ankle.x, ground_point_ankle.y);
                    cv::putText(result_frame, ankle_text, cv::Point(bbox.x, text_start_y + 25), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
                
                // 显示动态垂直拉伸量
                if (foot_result.use_keypoints) {
                    char offset_text[128];
                    sprintf(offset_text, "垂直拉伸: %.1fpx", foot_result.vertical_offset);
                    cv::putText(result_frame, offset_text, cv::Point(bbox.x, text_start_y + 50), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
        
        // --- 取出篮球检测结果并画框（可通过b键开关）---
        if (g_enable_basketball_detection) {
            std::lock_guard<std::mutex> lock(basketball_result_mutex);
            if (!basketball_result_queue.empty()) {
                BasketballDetectResult result = basketball_result_queue.back();
                
                // 设计原因: 清空队列，避免显示过时的检测结果
                while (!basketball_result_queue.empty()) {
                    basketball_result_queue.pop();
                }
                
                // 检查结果时间戳，避免显示过时的结果（超过100ms的结果丢弃）
                int64_t current_time = getCurrentTimeUs();
                if (current_time - result.timestamp > 100000) { // 100ms
                    // 丢弃过时结果，不显示
                } else {
                    // 使用letterbox工具函数进行篮球检测结果的坐标转换
                    cv::Rect model_bbox((int)result.x, (int)result.y, (int)result.w, (int)result.h);
                    cv::Rect bbox = letterbox_inverse_transform_bbox(model_bbox, &zero_copy_ctx.letterbox_ctx);
                    
                    // 设计原因: 只绘制有效的检测框
                    if (bbox.width > 0 && bbox.height > 0) {
                        cv::rectangle(result_frame, bbox, cv::Scalar(0,0,255), 3);
                        char conf_text[64];
                        sprintf(conf_text, "Basketball: %.2f", result.conf);
                        cv::putText(result_frame, conf_text, cv::Point(bbox.x, bbox.y-10), 
                                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
                    }
                }
            }
        }
        
        int64_t t5 = getCurrentTimeUs();
        char perf_text[256];
        sprintf(perf_text, "FPS: %.1f", frame_count * 1000000.0f / total_time);
        cv::putText(result_frame, perf_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // 显示ByteTrack状态
        char track_status[128];
        sprintf(track_status, "ByteTrack: %s (Press 'T' to toggle)", g_enable_tracking ? "ON" : "OFF");
        cv::Scalar track_color = g_enable_tracking ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(result_frame, track_status, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, track_color, 2);
        
        // 显示篮球检测状态
        char basketball_status[128];
        sprintf(basketball_status, "Basketball Detection: %s (Press 'B' to toggle)", g_enable_basketball_detection ? "ON" : "OFF");
        cv::Scalar basketball_color = g_enable_basketball_detection ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(result_frame, basketball_status, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, basketball_color, 2);
        cv::imshow(WINDOW_NAME, result_frame);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC键
            break;
        } else if (key == 't' || key == 'T') { // T键切换ByteTrack
            g_enable_tracking = !g_enable_tracking;
            printf("ByteTrack: %s\n", g_enable_tracking ? "ON" : "OFF");
        } else if (key == 'b' || key == 'B') { // B键切换篮球检测
            g_enable_basketball_detection = !g_enable_basketball_detection;
            printf("Basketball Detection: %s\n", g_enable_basketball_detection ? "ON" : "OFF");
        }
        int64_t frame_time = t5 - t0;
        total_time += frame_time;
        frame_count++;

        // --- 在此处将 letterbox处理后的图像 送入篮球检测队列（可通过b键开关）---
        if (g_enable_basketball_detection) {
            std::lock_guard<std::mutex> lock(basketball_queue_mutex);
            // 设计原因: 增加队列长度到10，提高处理能力
            if (basketball_frame_queue.size() < 10) {
                // 设计原因: 对原始图像进行letterbox处理后传递给篮球检测线程
                cv::Mat basketball_input;
                letterbox_preprocess(frame, basketball_input, &zero_copy_ctx.letterbox_ctx);
                basketball_frame_queue.push(basketball_input);
                basketball_queue_cv.notify_one();
            }
        }
    } // 主循环结束

    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    release_zero_copy_mem(&rknn_app_ctx, &zero_copy_ctx);
    release_yolov8_pose_model(&rknn_app_ctx);
    deinit_post_process();

    // 退出前安全关闭篮球检测线程
    basketball_thread_running = false;
    basketball_queue_cv.notify_all();
    if (basketball_detect_thread.joinable()) {
        basketball_detect_thread.join();
    }

    return 0;
} 