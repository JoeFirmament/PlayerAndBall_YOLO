#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <chrono>

// RGA headers
#include "im2d.h"
#include "im2d_type.h"
#include "im2d_single.h"
#include "RgaUtils.h"

static int64_t getCurrentTimeUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// OpenCV resize实现
int opencv_resize_test(cv::Mat& src, cv::Mat& dst, int iterations) {
    int64_t total_time = 0;
    
    for (int i = 0; i < iterations; i++) {
        int64_t start = getCurrentTimeUs();
        cv::resize(src, dst, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
        int64_t end = getCurrentTimeUs();
        total_time += (end - start);
    }
    
    printf("OpenCV resize: %.2fms/次 (共%d次)\n", 
           total_time / 1000.0f / iterations, iterations);
    return 0;
}

// RGA resize实现
int rga_resize_test(cv::Mat& src, cv::Mat& dst, int iterations) {
    int64_t total_time = 0;
    
    for (int i = 0; i < iterations; i++) {
        int64_t start = getCurrentTimeUs();
        
        // 使用importbuffer方式
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
        if (src_handle == 0) {
            printf("❌ RGA源buffer导入失败\n");
            return -1;
        }
        
        rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
        if (dst_handle == 0) {
            printf("❌ RGA目标buffer导入失败\n");
            releasebuffer_handle(src_handle);
            return -1;
        }
        
        // 创建rga_buffer_t
        rga_buffer_t src_buf, dst_buf;
        src_buf = wrapbuffer_handle(src_handle, src.cols, src.rows, RK_FORMAT_BGR_888);
        dst_buf = wrapbuffer_handle(dst_handle, dst.cols, dst.rows, RK_FORMAT_BGR_888);
        
        // RGA resize
        int ret = imresize(src_buf, dst_buf);
        if (ret != IM_STATUS_SUCCESS) {
            printf("❌ RGA resize失败: %s\n", imStrError((IM_STATUS)ret));
            releasebuffer_handle(src_handle);
            releasebuffer_handle(dst_handle);
            return -1;
        }
        
        // 释放handle
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
        
        int64_t end = getCurrentTimeUs();
        total_time += (end - start);
    }
    
    printf("RGA resize: %.2fms/次 (共%d次)\n", 
           total_time / 1000.0f / iterations, iterations);
    return 0;
}

// 混合方案：RGA + letterbox
int rga_letterbox_test(cv::Mat& src, cv::Mat& dst, int iterations) {
    int64_t total_time = 0;
    
    // 计算letterbox参数
    int src_width = src.cols;
    int src_height = src.rows;
    int dst_width = 640;
    int dst_height = 640;
    
    float scale_w = (float)dst_width / src_width;
    float scale_h = (float)dst_height / src_height;
    float scale = std::min(scale_w, scale_h);
    int new_width = (int)(src_width * scale);
    int new_height = (int)(src_height * scale);
    int offset_x = (dst_width - new_width) / 2;
    int offset_y = (dst_height - new_height) / 2;
    
    printf("Letterbox参数: %dx%d -> %dx%d, offset(%d,%d)\n", 
           src_width, src_height, new_width, new_height, offset_x, offset_y);
    
    for (int i = 0; i < iterations; i++) {
        int64_t start = getCurrentTimeUs();
        
        // 1. 用RGA resize到letterbox尺寸
        cv::Mat resized(new_height, new_width, CV_8UC3);
        
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
        rga_buffer_handle_t resized_handle = importbuffer_virtualaddr(resized.data, resized.cols, resized.rows, RK_FORMAT_BGR_888);
        
        if (src_handle == 0 || resized_handle == 0) {
            printf("❌ RGA buffer导入失败\n");
            if (src_handle) releasebuffer_handle(src_handle);
            if (resized_handle) releasebuffer_handle(resized_handle);
            return -1;
        }
        
        rga_buffer_t src_buf = wrapbuffer_handle(src_handle, src.cols, src.rows, RK_FORMAT_BGR_888);
        rga_buffer_t resized_buf = wrapbuffer_handle(resized_handle, resized.cols, resized.rows, RK_FORMAT_BGR_888);
        
        int ret = imresize(src_buf, resized_buf);
        releasebuffer_handle(src_handle);
        releasebuffer_handle(resized_handle);
        
        if (ret != IM_STATUS_SUCCESS) {
            printf("❌ RGA resize失败\n");
            return -1;
        }
        
        // 2. 填充灰色背景并复制到中央
        dst.setTo(cv::Scalar(114, 114, 114));
        cv::Mat roi = dst(cv::Rect(offset_x, offset_y, new_width, new_height));
        resized.copyTo(roi);
        
        int64_t end = getCurrentTimeUs();
        total_time += (end - start);
    }
    
    printf("RGA+Letterbox: %.2fms/次 (共%d次)\n", 
           total_time / 1000.0f / iterations, iterations);
    return 0;
}

int main(int argc, char** argv) {
    printf("=== RGA硬件加速resize性能测试 ===\n");
    
    // 模拟摄像头数据：1920x1080 -> 640x640
    cv::Mat src_frame(1080, 1920, CV_8UC3);
    cv::randu(src_frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    
    cv::Mat dst_opencv(640, 640, CV_8UC3);
    cv::Mat dst_rga(640, 640, CV_8UC3);
    cv::Mat dst_letterbox(640, 640, CV_8UC3);
    
    printf("输入图像: %dx%d\n", src_frame.cols, src_frame.rows);
    printf("输出图像: %dx%d\n", dst_opencv.cols, dst_opencv.rows);
    printf("测试次数: 100次\n\n");
    
    // 检查RGA是否可用
    const char* rga_version = querystring(RGA_VERSION);
    if (!rga_version) {
        printf("❌ RGA不可用，请检查驱动\n");
        return -1;
    }
    printf("RGA版本: %s\n", rga_version);
    printf("---\n");
    
    // 性能测试
    int iterations = 100;
    
    printf("1️⃣ OpenCV resize测试:\n");
    opencv_resize_test(src_frame, dst_opencv, iterations);
    
    printf("\n2️⃣ RGA硬件加速测试:\n");
    rga_resize_test(src_frame, dst_rga, iterations);
    
    printf("\n3️⃣ RGA+Letterbox混合方案测试:\n");
    rga_letterbox_test(src_frame, dst_letterbox, iterations);
    
    // 验证结果正确性
    printf("\n=== 结果验证 ===\n");
    
    // 保存测试图像
    cv::imwrite("opencv_result.jpg", dst_opencv);
    cv::imwrite("rga_result.jpg", dst_rga);
    cv::imwrite("letterbox_result.jpg", dst_letterbox);
    printf("✅ 结果图像已保存: opencv_result.jpg, rga_result.jpg, letterbox_result.jpg\n");
    
    // 简单的像素差异检查
    cv::Mat diff;
    cv::absdiff(dst_opencv, dst_rga, diff);
    cv::Scalar mean_diff = cv::mean(diff);
    printf("OpenCV vs RGA 平均像素差异: R=%.1f G=%.1f B=%.1f\n", 
           mean_diff[2], mean_diff[1], mean_diff[0]);
    
    return 0;
}