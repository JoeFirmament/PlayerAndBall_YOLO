/*-------------------------------------------
       双图像NPU分配测试程序
         
使用两张静态图像模拟双摄像头场景
测试NPU核心分配功能是否正常工作
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"
#include "npu_utils.h"

using namespace detector;

// 测试单张图像的姿态检测
void test_pose_detection(const std::string& image_path, int npu_core) {
    std::cout << "\n[姿态检测测试] NPU核心: " << npu_core << std::endl;
    std::cout << "图像: " << image_path << std::endl;
    
    try {
        // 读取图像
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("无法读取图像: " + image_path);
        }
        std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
        
        // 创建姿态检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        std::cout << "模型路径: " << model_path << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        PoseDetectorLib detector(model_path, npu_core);
        auto init_end = std::chrono::high_resolution_clock::now();
        
        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - start).count();
        std::cout << "检测器初始化时间: " << init_time << "ms" << std::endl;
        
        // 执行检测
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto results = detector.detect(image);
        auto detect_end = std::chrono::high_resolution_clock::now();
        
        auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();
        std::cout << "检测耗时: " << detect_time << "ms" << std::endl;
        std::cout << "推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        std::cout << "检测到人数: " << results.size() << std::endl;
        
        // 绘制结果
        cv::Mat display = image.clone();
        for (const auto& pose : results) {
            // 绘制边界框
            cv::rectangle(display, pose.bbox, cv::Scalar(0, 255, 0), 2);
            
            // 绘制关键点
            for (size_t i = 0; i < pose.keypoints.size() && i < pose.keypoint_scores.size(); i++) {
                if (pose.keypoint_scores[i] > 0.5) {
                    cv::circle(display, pose.keypoints[i], 
                              3, cv::Scalar(0, 255, 255), -1);
                }
            }
            
            // 显示置信度
            std::string info = "conf: " + std::to_string(pose.confidence).substr(0, 4);
            cv::putText(display, info, cv::Point(pose.bbox.x, pose.bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        
        // 保存结果
        std::string output_path = "pose_result_npu" + std::to_string(npu_core) + ".jpg";
        cv::imwrite(output_path, display);
        std::cout << "结果已保存到: " << output_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[姿态检测] 错误: " << e.what() << std::endl;
    }
}

// 测试单张图像的篮筐检测
void test_rim_detection(const std::string& image_path, int npu_core) {
    std::cout << "\n[篮筐检测测试] NPU核心: " << npu_core << std::endl;
    std::cout << "图像: " << image_path << std::endl;
    
    try {
        // 读取图像
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("无法读取图像: " + image_path);
        }
        std::cout << "图像尺寸: " << image.cols << "x" << image.rows << std::endl;
        
        // 创建篮筐检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        std::cout << "模型路径: " << model_path << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        RimBasketballDetectorLib detector(model_path, npu_core);
        auto init_end = std::chrono::high_resolution_clock::now();
        
        auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - start).count();
        std::cout << "检测器初始化时间: " << init_time << "ms" << std::endl;
        
        // 执行检测
        auto detect_start = std::chrono::high_resolution_clock::now();
        auto results = detector.detect(image);
        auto detect_end = std::chrono::high_resolution_clock::now();
        
        auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();
        std::cout << "检测耗时: " << detect_time << "ms" << std::endl;
        std::cout << "推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        
        // 统计结果
        int rim_count = 0, basketball_count = 0;
        for (const auto& obj : results) {
            if (obj.class_id == 1) rim_count++;
            else basketball_count++;
        }
        std::cout << "检测结果: 篮筐=" << rim_count << ", 篮球=" << basketball_count << std::endl;
        
        // 绘制结果
        cv::Mat display = image.clone();
        for (const auto& obj : results) {
            cv::Scalar color = (obj.class_id == 0) ? 
                cv::Scalar(0, 165, 255) :  // 篮球-橙色
                cv::Scalar(0, 255, 0);      // 篮筐-绿色
                
            cv::rectangle(display, obj.bbox, color, 2);
            
            std::string label = obj.class_name + " " + 
                std::to_string(obj.confidence).substr(0, 4);
            cv::putText(display, label, cv::Point(obj.bbox.x, obj.bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        }
        
        // 保存结果
        std::string output_path = "rim_result_npu" + std::to_string(npu_core) + ".jpg";
        cv::imwrite(output_path, display);
        std::cout << "结果已保存到: " << output_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[篮筐检测] 错误: " << e.what() << std::endl;
    }
}

// 并行测试函数
void parallel_test_thread(bool is_pose, const std::string& image_path, int npu_core) {
    if (is_pose) {
        test_pose_detection(image_path, npu_core);
    } else {
        test_rim_detection(image_path, npu_core);
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "    双图像NPU分配测试程序" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // 获取NPU信息
    std::cout << "\n📊 检测NPU状态..." << std::endl;
    NPUInfo npu_info = NPUUtils::get_npu_info();
    std::cout << "NPU核心数: " << npu_info.total_cores << std::endl;
    std::cout << "当前频率: " << npu_info.current_freq_mhz << " MHz" << std::endl;
    std::cout << "温度: " << npu_info.temperature_celsius << "°C" << std::endl;
    
    // 图像路径
    std::string pose_image = "../imgs/pose.jpg";
    std::string rim_image = "../imgs/rim.jpg";
    
    std::cout << "\n=== 测试1: 串行测试（相同NPU核心） ===" << std::endl;
    test_pose_detection(pose_image, -1);  // 自动分配
    test_rim_detection(rim_image, -1);    // 自动分配
    
    std::cout << "\n=== 测试2: 串行测试（不同NPU核心） ===" << std::endl;
    test_pose_detection(pose_image, 0);   // NPU核心0
    test_rim_detection(rim_image, 1);     // NPU核心1
    
    std::cout << "\n=== 测试3: 并行测试（不同NPU核心） ===" << std::endl;
    std::cout << "启动并行检测线程..." << std::endl;
    
    std::thread pose_thread(parallel_test_thread, true, pose_image, 0);
    std::thread rim_thread(parallel_test_thread, false, rim_image, 1);
    
    pose_thread.join();
    rim_thread.join();
    
    std::cout << "\n=== 测试4: 并行测试（相同NPU核心） ===" << std::endl;
    std::cout << "警告：使用相同NPU核心可能导致性能下降" << std::endl;
    
    std::thread pose_thread2(parallel_test_thread, true, pose_image, 0);
    std::thread rim_thread2(parallel_test_thread, false, rim_image, 0);
    
    pose_thread2.join();
    rim_thread2.join();
    
    std::cout << "\n=== 所有测试完成 ===" << std::endl;
    std::cout << "请查看生成的结果图像：" << std::endl;
    std::cout << "- pose_result_npu*.jpg" << std::endl;
    std::cout << "- rim_result_npu*.jpg" << std::endl;
    
    return 0;
}