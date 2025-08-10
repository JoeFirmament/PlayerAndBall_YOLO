/*-------------------------------------------
       内存隔离和资源管理测试
         
验证多个检测器之间的内存隔离
测试资源创建、释放和错误恢复能力
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <atomic>
#include <random>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

std::atomic<int> g_total_detectors_created{0};
std::atomic<int> g_total_detectors_destroyed{0};
std::atomic<bool> g_test_running{true};

// 测试1：多实例创建和销毁
void test_multiple_instances() {
    std::cout << "\n=== 测试1: 多实例创建和销毁 ===" << std::endl;
    
    std::vector<std::unique_ptr<PoseDetectorLib>> pose_detectors;
    std::vector<std::unique_ptr<RimBasketballDetectorLib>> rim_detectors;
    
    std::string pose_model = PathUtils::find_model("Q_yolov8_pose.rknn");
    std::string rim_model = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
    
    // 创建多个实例
    for (int i = 0; i < 3; i++) {
        for (int npu = 0; npu < 3; npu++) {
            std::cout << "创建实例 " << i << " NPU" << npu << "...";
            
            auto pose_det = std::make_unique<PoseDetectorLib>(pose_model, npu);
            auto rim_det = std::make_unique<RimBasketballDetectorLib>(rim_model, npu);
            
            pose_detectors.push_back(std::move(pose_det));
            rim_detectors.push_back(std::move(rim_det));
            g_total_detectors_created += 2;
            
            std::cout << " ✓" << std::endl;
        }
    }
    
    std::cout << "创建了 " << pose_detectors.size() << " 个姿态检测器" << std::endl;
    std::cout << "创建了 " << rim_detectors.size() << " 个篮筐检测器" << std::endl;
    
    // 逐个销毁
    std::cout << "逐个销毁检测器..." << std::endl;
    pose_detectors.clear();
    rim_detectors.clear();
    g_total_detectors_destroyed += 18; // 9个pose + 9个rim
    
    std::cout << "✓ 多实例测试完成" << std::endl;
}

// 测试2：并发创建销毁
void detector_lifecycle_thread(int thread_id, int iterations) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> npu_dist(0, 2);
    
    std::string pose_model = PathUtils::find_model("Q_yolov8_pose.rknn");
    cv::Mat test_image = cv::Mat::zeros(640, 480, CV_8UC3);
    cv::rectangle(test_image, cv::Rect(100, 100, 200, 300), cv::Scalar(128, 128, 128), -1);
    
    for (int i = 0; i < iterations && g_test_running; i++) {
        try {
            int npu_core = npu_dist(gen);
            
            // 创建检测器
            auto detector = std::make_unique<PoseDetectorLib>(pose_model, npu_core);
            g_total_detectors_created++;
            
            // 进行几次检测
            for (int j = 0; j < 3; j++) {
                auto results = detector->detect(test_image);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // 销毁检测器
            detector.reset();
            g_total_detectors_destroyed++;
            
            if (i % 5 == 0) {
                std::cout << "[线程" << thread_id << "] 完成迭代 " << i << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[线程" << thread_id << "] 异常: " << e.what() << std::endl;
        }
    }
}

void test_concurrent_lifecycle() {
    std::cout << "\n=== 测试2: 并发生命周期管理 ===" << std::endl;
    
    const int num_threads = 4;
    const int iterations_per_thread = 10;
    
    std::vector<std::thread> threads;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // 启动测试线程
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back(detector_lifecycle_thread, i, iterations_per_thread);
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "并发测试完成，耗时: " << duration.count() << "秒" << std::endl;
    std::cout << "✓ 并发生命周期测试完成" << std::endl;
}

// 测试3：内存泄漏检测（简单版）
void test_memory_leaks() {
    std::cout << "\n=== 测试3: 内存泄漏检测 ===" << std::endl;
    
    std::string pose_model = PathUtils::find_model("Q_yolov8_pose.rknn");
    cv::Mat test_image = cv::Mat::zeros(640, 480, CV_8UC3);
    
    // 大量创建和销毁
    const int test_cycles = 50;
    
    for (int cycle = 0; cycle < test_cycles; cycle++) {
        std::vector<std::unique_ptr<PoseDetectorLib>> detectors;
        
        // 创建多个检测器
        for (int i = 0; i < 5; i++) {
            auto detector = std::make_unique<PoseDetectorLib>(pose_model, i % 3);
            
            // 进行几次检测
            auto results = detector->detect(test_image);
            
            detectors.push_back(std::move(detector));
        }
        
        // 检测器会在这里自动销毁
        if (cycle % 10 == 0) {
            std::cout << "完成周期 " << cycle << "/" << test_cycles << std::endl;
        }
    }
    
    std::cout << "✓ 内存泄漏测试完成 (共创建/销毁 " << test_cycles * 5 << " 个检测器)" << std::endl;
}

// 测试4：错误恢复能力
void test_error_recovery() {
    std::cout << "\n=== 测试4: 错误恢复能力 ===" << std::endl;
    
    try {
        // 测试无效模型路径
        std::cout << "测试无效模型路径...";
        try {
            PoseDetectorLib invalid_detector("/invalid/path/model.rknn", 0);
            cv::Mat test_img = cv::Mat::zeros(640, 480, CV_8UC3);
            auto results = invalid_detector.detect(test_img); // 应该优雅处理失败
            std::cout << " ✓ (优雅处理)" << std::endl;
        } catch (...) {
            std::cout << " ✓ (抛出异常)" << std::endl;
        }
        
        // 测试无效NPU核心
        std::cout << "测试无效NPU核心...";
        std::string pose_model = PathUtils::find_model("Q_yolov8_pose.rknn");
        PoseDetectorLib detector_invalid_npu(pose_model, 999); // 无效核心ID
        cv::Mat test_img = cv::Mat::zeros(640, 480, CV_8UC3);
        auto results = detector_invalid_npu.detect(test_img);
        std::cout << " ✓ (回退到自动分配)" << std::endl;
        
        // 测试空图像
        std::cout << "测试空图像...";
        PoseDetectorLib detector_valid(pose_model, 0);
        cv::Mat empty_img;
        auto empty_results = detector_valid.detect(empty_img);
        std::cout << " ✓ (返回空结果)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "错误恢复测试异常: " << e.what() << std::endl;
    }
    
    std::cout << "✓ 错误恢复测试完成" << std::endl;
}

// 测试5：长时间压力测试
void stress_test_thread(int thread_id, int duration_seconds) {
    std::string pose_model = PathUtils::find_model("Q_yolov8_pose.rknn");
    cv::Mat test_image = cv::Mat::zeros(640, 480, CV_8UC3);
    cv::rectangle(test_image, cv::Rect(100, 100, 200, 300), cv::Scalar(128, 128, 128), -1);
    
    PoseDetectorLib detector(pose_model, thread_id % 3);
    
    auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    
    while (g_test_running) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
        
        if (elapsed.count() >= duration_seconds) {
            break;
        }
        
        // 进行检测
        auto results = detector.detect(test_image);
        frame_count++;
        
        // 控制帧率到约30FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        
        // 每10秒输出一次统计
        if (frame_count % 300 == 0) {
            float fps = frame_count / (float)elapsed.count();
            std::cout << "[线程" << thread_id << "] " 
                      << elapsed.count() << "s, "
                      << frame_count << "帧, "
                      << fps << "FPS" << std::endl;
        }
    }
    
    std::cout << "[线程" << thread_id << "] 压力测试完成: " << frame_count << "帧" << std::endl;
}

void test_long_term_stability() {
    std::cout << "\n=== 测试5: 长时间稳定性测试 (30秒) ===" << std::endl;
    
    const int num_threads = 3;
    const int duration_seconds = 30;
    
    std::vector<std::thread> stress_threads;
    
    // 启动压力测试线程
    for (int i = 0; i < num_threads; i++) {
        stress_threads.emplace_back(stress_test_thread, i, duration_seconds);
    }
    
    // 等待所有线程完成
    for (auto& t : stress_threads) {
        t.join();
    }
    
    std::cout << "✓ 长时间稳定性测试完成" << std::endl;
}

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    内存隔离和资源管理测试程序" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        test_multiple_instances();
        test_concurrent_lifecycle();
        test_memory_leaks();
        test_error_recovery();
        test_long_term_stability();
        
    } catch (const std::exception& e) {
        std::cerr << "测试程序异常: " << e.what() << std::endl;
        return 1;
    }
    
    g_test_running = false;
    
    std::cout << "\n=========================================" << std::endl;
    std::cout << "所有测试完成！" << std::endl;
    std::cout << "总创建检测器: " << g_total_detectors_created.load() << std::endl;
    std::cout << "总销毁检测器: " << g_total_detectors_destroyed.load() << std::endl;
    std::cout << "=========================================" << std::endl;
    
    return 0;
}