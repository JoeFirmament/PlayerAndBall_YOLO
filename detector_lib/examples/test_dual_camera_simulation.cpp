/*-------------------------------------------
       双摄像头模拟测试程序
         
模拟真实用户的双摄像头使用场景：
- 持续运行的双线程检测
- 性能统计和对比
- NPU资源分配效果验证
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"
#include "npu_utils.h"

using namespace detector;

// 全局控制变量
std::atomic<bool> g_running(true);
std::mutex g_print_mutex;

// 性能统计结构
struct PerformanceStats {
    std::atomic<int> frame_count{0};
    std::atomic<int> total_inference_time_ms{0};
    std::atomic<int> max_inference_time_ms{0};
    std::atomic<int> min_inference_time_ms{9999};
    std::chrono::steady_clock::time_point start_time;
    
    void reset() {
        frame_count = 0;
        total_inference_time_ms = 0;
        max_inference_time_ms = 0;
        min_inference_time_ms = 9999;
        start_time = std::chrono::steady_clock::now();
    }
    
    void update(int inference_time_ms) {
        frame_count++;
        total_inference_time_ms += inference_time_ms;
        
        int current_max = max_inference_time_ms.load();
        while (inference_time_ms > current_max && 
               !max_inference_time_ms.compare_exchange_weak(current_max, inference_time_ms));
        
        int current_min = min_inference_time_ms.load();
        while (inference_time_ms < current_min && 
               !min_inference_time_ms.compare_exchange_weak(current_min, inference_time_ms));
    }
    
    void print(const std::string& name) {
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        if (duration == 0) duration = 1;
        
        int frames = frame_count.load();
        int total_time = total_inference_time_ms.load();
        float avg_time = frames > 0 ? (float)total_time / frames : 0;
        float fps = (float)frames / duration;
        
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "[" << name << "] "
                  << "FPS: " << fps 
                  << " | 平均推理: " << avg_time << "ms"
                  << " | 最小: " << min_inference_time_ms.load() << "ms"
                  << " | 最大: " << max_inference_time_ms.load() << "ms"
                  << " | 总帧数: " << frames
                  << std::endl;
    }
};

// 模拟摄像头线程 - 姿态检测
void pose_camera_thread(const std::string& test_image_path, int npu_core, 
                       PerformanceStats& stats, int camera_fps = 30) {
    try {
        std::string thread_name = "姿态检测-NPU" + std::to_string(npu_core);
        std::cout << "[" << thread_name << "] 线程启动" << std::endl;
        
        // 加载测试图像
        cv::Mat test_image = cv::imread(test_image_path);
        if (test_image.empty()) {
            throw std::runtime_error("无法加载测试图像: " + test_image_path);
        }
        
        // 创建检测器
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        PoseDetectorLib detector(model_path, npu_core);
        detector.enable_tracking(true);
        
        // 模拟摄像头帧率
        auto frame_duration = std::chrono::milliseconds(1000 / camera_fps);
        auto next_frame_time = std::chrono::steady_clock::now();
        
        while (g_running) {
            // 模拟获取新帧（实际使用同一张图像）
            cv::Mat frame = test_image.clone();
            
            // 执行检测
            auto results = detector.detect(frame);
            
            // 更新统计
            int inference_time = detector.get_last_inference_time_ms();
            stats.update(inference_time);
            
            // 每秒输出一次统计
            if (stats.frame_count % camera_fps == 0) {
                stats.print(thread_name);
            }
            
            // 控制帧率
            next_frame_time += frame_duration;
            std::this_thread::sleep_until(next_frame_time);
        }
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cerr << "[姿态检测] 错误: " << e.what() << std::endl;
    }
}

// 模拟摄像头线程 - 篮筐检测
void rim_camera_thread(const std::string& test_image_path, int npu_core, 
                      PerformanceStats& stats, int camera_fps = 30) {
    try {
        std::string thread_name = "篮筐检测-NPU" + std::to_string(npu_core);
        std::cout << "[" << thread_name << "] 线程启动" << std::endl;
        
        // 加载测试图像
        cv::Mat test_image = cv::imread(test_image_path);
        if (test_image.empty()) {
            throw std::runtime_error("无法加载测试图像: " + test_image_path);
        }
        
        // 创建检测器
        std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        RimBasketballDetectorLib detector(model_path, npu_core);
        
        // 模拟摄像头帧率
        auto frame_duration = std::chrono::milliseconds(1000 / camera_fps);
        auto next_frame_time = std::chrono::steady_clock::now();
        
        while (g_running) {
            // 模拟获取新帧
            cv::Mat frame = test_image.clone();
            
            // 执行检测
            auto results = detector.detect(frame);
            
            // 更新统计
            int inference_time = detector.get_last_inference_time_ms();
            stats.update(inference_time);
            
            // 每秒输出一次统计
            if (stats.frame_count % camera_fps == 0) {
                stats.print(thread_name);
            }
            
            // 控制帧率
            next_frame_time += frame_duration;
            std::this_thread::sleep_until(next_frame_time);
        }
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(g_print_mutex);
        std::cerr << "[篮筐检测] 错误: " << e.what() << std::endl;
    }
}

// 运行测试场景
void run_test_scenario(const std::string& scenario_name, 
                      int pose_npu, int rim_npu, 
                      int duration_seconds = 10) {
    std::cout << "\n=========================================" << std::endl;
    std::cout << "场景: " << scenario_name << std::endl;
    std::cout << "配置: 姿态检测->NPU" << pose_npu 
              << ", 篮筐检测->NPU" << rim_npu << std::endl;
    std::cout << "测试时长: " << duration_seconds << "秒" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // 重置运行标志
    g_running = true;
    
    // 性能统计
    PerformanceStats pose_stats, rim_stats;
    pose_stats.reset();
    rim_stats.reset();
    
    // 启动检测线程
    std::thread pose_thread(pose_camera_thread, "../imgs/pose.jpg", 
                           pose_npu, std::ref(pose_stats), 30);
    std::thread rim_thread(rim_camera_thread, "../imgs/rim.jpg", 
                          rim_npu, std::ref(rim_stats), 30);
    
    // 运行指定时长
    std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
    
    // 停止线程
    g_running = false;
    pose_thread.join();
    rim_thread.join();
    
    // 输出最终统计
    std::cout << "\n--- 最终统计 ---" << std::endl;
    pose_stats.print("姿态检测");
    rim_stats.print("篮筐检测");
    
    // 计算总体性能
    float total_fps = pose_stats.frame_count / (float)duration_seconds + 
                     rim_stats.frame_count / (float)duration_seconds;
    std::cout << "系统总吞吐量: " << total_fps << " FPS" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "    双摄像头模拟测试程序" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // 获取NPU信息
    std::cout << "\n📊 系统信息" << std::endl;
    NPUInfo npu_info = NPUUtils::get_npu_info();
    std::cout << "NPU核心数: " << npu_info.total_cores << std::endl;
    std::cout << "当前频率: " << npu_info.current_freq_mhz << " MHz" << std::endl;
    std::cout << "温度: " << npu_info.temperature_celsius << "°C" << std::endl;
    
    // 测试时长（秒）
    int test_duration = 15;
    if (argc > 1) {
        test_duration = std::atoi(argv[1]);
    }
    
    // 场景1：自动分配（可能造成冲突）
    run_test_scenario("自动NPU分配", -1, -1, test_duration);
    
    // 等待系统稳定
    std::cout << "\n等待3秒..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // 场景2：相同NPU核心（最差情况）
    run_test_scenario("相同NPU核心（NPU0）", 0, 0, test_duration);
    
    // 等待系统稳定
    std::cout << "\n等待3秒..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // 场景3：不同NPU核心（推荐配置）
    run_test_scenario("不同NPU核心（NPU0+NPU1）", 0, 1, test_duration);
    
    // 等待系统稳定
    std::cout << "\n等待3秒..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // 场景4：使用NPU0和NPU2
    run_test_scenario("不同NPU核心（NPU0+NPU2）", 0, 2, test_duration);
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << "测试完成！" << std::endl;
    std::cout << "建议：使用不同的NPU核心可获得最佳性能" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    return 0;
}