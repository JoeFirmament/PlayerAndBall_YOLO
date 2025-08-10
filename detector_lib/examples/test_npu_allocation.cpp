/*-------------------------------------------
         NPU核心分配测试程序
         
用于验证detector_lib的NPU核心分配功能
测试两个detector分别使用不同的NPU核心
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

void test_pose_detector(int npu_core) {
    std::cout << "[PoseDetector] 线程启动，使用NPU核心: " << npu_core << std::endl;
    
    try {
        // 创建姿态检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        PoseDetectorLib detector(model_path, npu_core);
        
        // 创建测试图像
        cv::Mat test_image(640, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        
        // 进行10次检测
        for (int i = 0; i < 10; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto results = detector.detect(test_image);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "[PoseDetector] 检测 #" << i+1 
                      << " 完成，耗时: " << duration.count() << "ms" 
                      << " 推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[PoseDetector] 测试完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[PoseDetector] 错误: " << e.what() << std::endl;
    }
}

void test_rim_detector(int npu_core) {
    std::cout << "[RimDetector] 线程启动，使用NPU核心: " << npu_core << std::endl;
    
    try {
        // 创建篮筐检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        RimBasketballDetectorLib detector(model_path, npu_core);
        
        // 创建测试图像
        cv::Mat test_image(640, 640, CV_8UC3, cv::Scalar(100, 100, 100));
        
        // 进行10次检测
        for (int i = 0; i < 10; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto results = detector.detect(test_image);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            
            std::cout << "[RimDetector] 检测 #" << i+1 
                      << " 完成，耗时: " << duration.count() << "ms"
                      << " 推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "[RimDetector] 测试完成" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[RimDetector] 错误: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== NPU核心分配测试程序 ===" << std::endl;
    std::cout << "RK3588S有3个NPU核心 (0, 1, 2)" << std::endl;
    
    std::cout << "\n--- 测试1: 自动分配NPU核心 ---" << std::endl;
    {
        std::thread t1(test_pose_detector, -1);  // 自动分配
        std::thread t2(test_rim_detector, -1);   // 自动分配
        
        t1.join();
        t2.join();
    }
    
    std::cout << "\n--- 测试2: 手动分配不同NPU核心 ---" << std::endl;
    {
        std::thread t1(test_pose_detector, 0);   // 使用NPU核心0
        std::thread t2(test_rim_detector, 1);    // 使用NPU核心1
        
        t1.join();
        t2.join();
    }
    
    std::cout << "\n--- 测试3: 错开初始化时间 ---" << std::endl;
    {
        std::thread t1(test_pose_detector, 0);   // 先启动pose
        std::this_thread::sleep_for(std::chrono::seconds(2));  // 延迟2秒
        std::thread t2(test_rim_detector, 1);    // 再启动rim
        
        t1.join();
        t2.join();
    }
    
    std::cout << "\n=== 所有测试完成 ===" << std::endl;
    
    return 0;
}