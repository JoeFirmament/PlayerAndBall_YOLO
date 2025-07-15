#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>

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

// 高精度计时
class PrecisionTimer {
public:
    static double getCurrentTimeMs() {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = now.time_since_epoch();
        return std::chrono::duration<double, std::milli>(duration).count();
    }
};

// OpenCV resize基准测试
void opencv_resize_benchmark(cv::Mat& src, cv::Mat& dst, int iterations) {
    printf("🔍 OpenCV resize基准测试 (%d次)\n", iterations);
    
    std::vector<double> times;
    times.reserve(iterations);
    
    // 预热
    for (int i = 0; i < 10; i++) {
        cv::resize(src, dst, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    }
    
    // 正式测试
    for (int i = 0; i < iterations; i++) {
        double start = PrecisionTimer::getCurrentTimeMs();
        cv::resize(src, dst, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
        double end = PrecisionTimer::getCurrentTimeMs();
        times.push_back(end - start);
    }
    
    // 统计分析
    double total = 0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        total += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    
    double avg = total / iterations;
    
    // 计算标准差
    double variance = 0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double stddev = sqrt(variance / iterations);
    
    printf("  平均耗时: %.3fms\n", avg);
    printf("  最短耗时: %.3fms\n", min_time);
    printf("  最长耗时: %.3fms\n", max_time);
    printf("  标准差  : %.3fms\n", stddev);
    printf("  抖动    : %.1f%%\n", (stddev / avg) * 100);
}

// RGA resize基准测试
void rga_resize_benchmark(cv::Mat& src, cv::Mat& dst, int iterations) {
    printf("🔍 RGA resize基准测试 (%d次)\n", iterations);
    
    std::vector<double> times;
    times.reserve(iterations);
    
    // 预热
    for (int i = 0; i < 10; i++) {
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
        rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
        if (src_handle && dst_handle) {
            rga_buffer_t src_buf = wrapbuffer_handle(src_handle, src.cols, src.rows, RK_FORMAT_BGR_888);
            rga_buffer_t dst_buf = wrapbuffer_handle(dst_handle, dst.cols, dst.rows, RK_FORMAT_BGR_888);
            imresize(src_buf, dst_buf);
        }
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
    }
    
    // 正式测试
    for (int i = 0; i < iterations; i++) {
        double start = PrecisionTimer::getCurrentTimeMs();
        
        rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
        rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
        
        if (src_handle && dst_handle) {
            rga_buffer_t src_buf = wrapbuffer_handle(src_handle, src.cols, src.rows, RK_FORMAT_BGR_888);
            rga_buffer_t dst_buf = wrapbuffer_handle(dst_handle, dst.cols, dst.rows, RK_FORMAT_BGR_888);
            imresize(src_buf, dst_buf);
        }
        
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
        
        double end = PrecisionTimer::getCurrentTimeMs();
        times.push_back(end - start);
    }
    
    // 统计分析
    double total = 0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        total += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    
    double avg = total / iterations;
    
    // 计算标准差
    double variance = 0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double stddev = sqrt(variance / iterations);
    
    printf("  平均耗时: %.3fms\n", avg);
    printf("  最短耗时: %.3fms\n", min_time);
    printf("  最长耗时: %.3fms\n", max_time);
    printf("  标准差  : %.3fms\n", stddev);
    printf("  抖动    : %.1f%%\n", (stddev / avg) * 100);
}

// 分离RGA操作测试：仅测试resize，不包含buffer管理
void rga_resize_only_benchmark(cv::Mat& src, cv::Mat& dst, int iterations) {
    printf("🔍 RGA纯resize测试 (不含buffer管理开销，%d次)\n", iterations);
    
    // 预先创建buffer handles，避免重复创建
    rga_buffer_handle_t src_handle = importbuffer_virtualaddr(src.data, src.cols, src.rows, RK_FORMAT_BGR_888);
    rga_buffer_handle_t dst_handle = importbuffer_virtualaddr(dst.data, dst.cols, dst.rows, RK_FORMAT_BGR_888);
    
    if (!src_handle || !dst_handle) {
        printf("❌ RGA buffer创建失败\n");
        return;
    }
    
    rga_buffer_t src_buf = wrapbuffer_handle(src_handle, src.cols, src.rows, RK_FORMAT_BGR_888);
    rga_buffer_t dst_buf = wrapbuffer_handle(dst_handle, dst.cols, dst.rows, RK_FORMAT_BGR_888);
    
    std::vector<double> times;
    times.reserve(iterations);
    
    // 预热
    for (int i = 0; i < 10; i++) {
        imresize(src_buf, dst_buf);
    }
    
    // 正式测试 - 只测试resize操作
    for (int i = 0; i < iterations; i++) {
        double start = PrecisionTimer::getCurrentTimeMs();
        int ret = imresize(src_buf, dst_buf);
        double end = PrecisionTimer::getCurrentTimeMs();
        
        if (ret == IM_STATUS_SUCCESS) {
            times.push_back(end - start);
        }
    }
    
    // 清理
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    
    if (times.empty()) {
        printf("❌ 没有成功的RGA resize操作\n");
        return;
    }
    
    // 统计分析
    double total = 0, min_time = times[0], max_time = times[0];
    for (double t : times) {
        total += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }
    
    double avg = total / times.size();
    
    // 计算标准差
    double variance = 0;
    for (double t : times) {
        variance += (t - avg) * (t - avg);
    }
    double stddev = sqrt(variance / times.size());
    
    printf("  平均耗时: %.3fms\n", avg);
    printf("  最短耗时: %.3fms\n", min_time);
    printf("  最长耗时: %.3fms\n", max_time);
    printf("  标准差  : %.3fms\n", stddev);
    printf("  抖动    : %.1f%%\n", (stddev / avg) * 100);
}

int main() {
    printf("=== OpenCV vs RGA Resize 精确性能对比 ===\n");
    
    // 检查RGA可用性
    const char* rga_version = querystring(RGA_VERSION);
    if (!rga_version) {
        printf("❌ RGA不可用\n");
        return -1;
    }
    printf("🔧 RGA版本: %s\n\n", rga_version);
    
    // 测试不同尺寸
    struct TestCase {
        int src_w, src_h, dst_w, dst_h;
        const char* name;
    };
    
    TestCase cases[] = {
        {1920, 1080, 640, 640, "1080p->640x640 (实际使用场景)"},
        {1280, 720, 640, 640, "720p->640x640"},
        {3840, 2160, 640, 640, "4K->640x640 (大尺寸)"},
        {640, 480, 320, 320, "VGA->320x320 (小尺寸)"}
    };
    
    int iterations = 200;  // 增加测试次数提高精度
    
    for (auto& test_case : cases) {
        printf("📊 测试场景: %s\n", test_case.name);
        printf("    输入: %dx%d -> 输出: %dx%d\n", 
               test_case.src_w, test_case.src_h, 
               test_case.dst_w, test_case.dst_h);
        
        // 创建测试图像
        cv::Mat src(test_case.src_h, test_case.src_w, CV_8UC3);
        cv::Mat dst_opencv(test_case.dst_h, test_case.dst_w, CV_8UC3);
        cv::Mat dst_rga(test_case.dst_h, test_case.dst_w, CV_8UC3);
        
        // 填充随机数据
        cv::randu(src, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        
        // OpenCV测试
        opencv_resize_benchmark(src, dst_opencv, iterations);
        
        // RGA完整测试
        rga_resize_benchmark(src, dst_rga, iterations);
        
        // RGA纯resize测试
        rga_resize_only_benchmark(src, dst_rga, iterations);
        
        // 质量对比
        cv::Mat diff;
        cv::absdiff(dst_opencv, dst_rga, diff);
        cv::Scalar mean_diff = cv::mean(diff);
        printf("  质量差异: R=%.1f G=%.1f B=%.1f (越小越好)\n", 
               mean_diff[2], mean_diff[1], mean_diff[0]);
        
        printf("\n" + std::string(60, '=') + "\n\n");
    }
    
    printf("📋 结论建议:\n");
    printf("1. 如果OpenCV更快 -> 继续使用OpenCV\n");
    printf("2. 如果RGA纯resize更快但总时间更慢 -> buffer管理是瓶颈\n");
    printf("3. 如果RGA在大尺寸图像上更快 -> 考虑场景优化\n");
    printf("4. 质量差异 < 5.0 认为可接受\n");
    
    return 0;
}