/*-------------------------------------------
       NPU调度机制验证测试
         
验证RKNN Runtime是否有智能调度功能
对比自动分配 vs 手动分配的详细性能
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

std::atomic<bool> g_running{true};
std::atomic<int> g_pose_frames{0};
std::atomic<int> g_rim_frames{0};

// 姿态检测线程
void pose_detection_worker(int npu_core, int duration_sec) {
    try {
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        PoseDetectorLib detector(model_path, npu_core);
        
        cv::Mat test_image = cv::imread("../imgs/pose.jpg");
        if (test_image.empty()) {
            test_image = cv::Mat::zeros(640, 480, CV_8UC3);
            cv::rectangle(test_image, cv::Rect(100, 100, 200, 300), cv::Scalar(128, 128, 128), -1);
        }
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (g_running) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            if (elapsed.count() >= duration_sec) {
                break;
            }
            
            auto results = detector.detect(test_image);
            g_pose_frames++;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[姿态检测] 异常: " << e.what() << std::endl;
    }
}

// 篮筐检测线程
void rim_detection_worker(int npu_core, int duration_sec) {
    try {
        std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        RimBasketballDetectorLib detector(model_path, npu_core);
        
        cv::Mat test_image = cv::imread("../imgs/rim.jpg");
        if (test_image.empty()) {
            test_image = cv::Mat::zeros(640, 480, CV_8UC3);
            cv::rectangle(test_image, cv::Rect(200, 200, 100, 100), cv::Scalar(255, 128, 0), -1);
        }
        
        auto start_time = std::chrono::steady_clock::now();
        
        while (g_running) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            if (elapsed.count() >= duration_sec) {
                break;
            }
            
            auto results = detector.detect(test_image);
            g_rim_frames++;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[篮筐检测] 异常: " << e.what() << std::endl;
    }
}

// 运行测试场景
void run_scenario(const std::string& name, int pose_npu, int rim_npu, int duration) {
    std::cout << "\n=========================================" << std::endl;
    std::cout << "场景: " << name << std::endl;
    std::cout << "姿态检测NPU: " << pose_npu << " | 篮筐检测NPU: " << rim_npu << std::endl;
    std::cout << "测试时长: " << duration << "秒" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // 重置计数器
    g_running = true;
    g_pose_frames = 0;
    g_rim_frames = 0;
    
    // 启动线程
    std::thread pose_thread(pose_detection_worker, pose_npu, duration);
    std::thread rim_thread(rim_detection_worker, rim_npu, duration);
    
    // 等待完成
    pose_thread.join();
    rim_thread.join();
    
    g_running = false;
    
    // 统计结果
    int total_pose = g_pose_frames.load();
    int total_rim = g_rim_frames.load();
    float pose_fps = (float)total_pose / duration;
    float rim_fps = (float)total_rim / duration;
    float total_fps = pose_fps + rim_fps;
    
    std::cout << "结果:" << std::endl;
    std::cout << "  姿态检测: " << total_pose << "帧, " << pose_fps << " FPS" << std::endl;
    std::cout << "  篮筐检测: " << total_rim << "帧, " << rim_fps << " FPS" << std::endl;
    std::cout << "  系统总吞吐量: " << total_fps << " FPS" << std::endl;
    std::cout << "=========================================" << std::endl;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    NPU调度机制验证测试" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    const int test_duration = 10; // 每个测试10秒
    
    // 测试1：自动分配（-1）
    std::cout << "\n🔄 测试RKNN Runtime自动调度能力" << std::endl;
    run_scenario("自动NPU分配", -1, -1, test_duration);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 测试2：相同NPU（最差情况）
    std::cout << "\n⚠️ 测试相同NPU性能（基线对比）" << std::endl;
    run_scenario("相同NPU核心（NPU0）", 0, 0, test_duration);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 测试3：不同NPU（理想情况）
    std::cout << "\n✅ 测试不同NPU性能（理想情况）" << std::endl;
    run_scenario("不同NPU核心（NPU0+NPU1）", 0, 1, test_duration);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 测试4：另一种不同NPU组合
    run_scenario("不同NPU核心（NPU0+NPU2）", 0, 2, test_duration);
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // 测试5：第三种不同NPU组合
    run_scenario("不同NPU核心（NPU1+NPU2）", 1, 2, test_duration);
    
    std::cout << "\n📊 分析结论:" << std::endl;
    std::cout << "1. 如果自动分配性能接近不同NPU，说明Runtime有智能调度" << std::endl;
    std::cout << "2. 如果自动分配性能接近相同NPU，说明Runtime调度有限" << std::endl;
    std::cout << "3. 不同NPU组合性能应该相近，验证并行能力" << std::endl;
    
    return 0;
}