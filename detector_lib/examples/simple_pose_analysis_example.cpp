/**
 * @file simple_pose_analysis_example.cpp  
 * @brief 简化的姿态分析系统使用示例
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

// 姿态分析系统头文件
#include "pose_analyzer.h"

using namespace pose_analysis;

// 创建符合要球条件的关键点数据
std::vector<PoseResult> simulate_bytetrack_output(int frame_id) {
    std::vector<PoseResult> results;
    
    // === Person 1: 要球动作 ===
    PoseResult pose1;
    pose1.person_id = 1;
    pose1.detection_confidence = 0.85f;
    pose1.timestamp = std::chrono::steady_clock::now();
    pose1.bbox = cv::Rect2f(100.0f, 50.0f, 120.0f, 450.0f);
    
    // 初始化17个关键点
    pose1.keypoints.resize(17);
    pose1.keypoint_confidences.resize(17, 0.8f);
    
    float center_x1 = pose1.bbox.x + pose1.bbox.width * 0.5f;  // 160
    
    // 头部关键点
    pose1.keypoints[0] = cv::Point2f(center_x1, pose1.bbox.y + 20);      // NOSE
    pose1.keypoints[1] = cv::Point2f(center_x1 - 10, pose1.bbox.y + 15); // LEFT_EYE
    pose1.keypoints[2] = cv::Point2f(center_x1 + 10, pose1.bbox.y + 15); // RIGHT_EYE
    pose1.keypoints[3] = cv::Point2f(center_x1 - 15, pose1.bbox.y + 18); // LEFT_EAR
    pose1.keypoints[4] = cv::Point2f(center_x1 + 15, pose1.bbox.y + 18); // RIGHT_EAR
    
    // 上身关键点
    pose1.keypoints[5] = cv::Point2f(center_x1 - 30, pose1.bbox.y + 80);  // LEFT_SHOULDER
    pose1.keypoints[6] = cv::Point2f(center_x1 + 30, pose1.bbox.y + 80);  // RIGHT_SHOULDER
    pose1.keypoints[7] = cv::Point2f(center_x1 - 40, pose1.bbox.y + 140); // LEFT_ELBOW
    pose1.keypoints[8] = cv::Point2f(center_x1 + 40, pose1.bbox.y + 140); // RIGHT_ELBOW
    
    // 要球手势：双手在胸前，距离较近
    float chest_y = pose1.bbox.y + 120;
    pose1.keypoints[9]  = cv::Point2f(center_x1 - 25, chest_y);  // LEFT_WRIST
    pose1.keypoints[10] = cv::Point2f(center_x1 + 25, chest_y);  // RIGHT_WRIST
    
    // 下身关键点
    pose1.keypoints[11] = cv::Point2f(center_x1 - 20, pose1.bbox.y + 200); // LEFT_HIP
    pose1.keypoints[12] = cv::Point2f(center_x1 + 20, pose1.bbox.y + 200); // RIGHT_HIP
    pose1.keypoints[13] = cv::Point2f(center_x1 - 25, pose1.bbox.y + 300); // LEFT_KNEE
    pose1.keypoints[14] = cv::Point2f(center_x1 + 25, pose1.bbox.y + 300); // RIGHT_KNEE
    pose1.keypoints[15] = cv::Point2f(center_x1 - 20, pose1.bbox.y + 430); // LEFT_ANKLE
    pose1.keypoints[16] = cv::Point2f(center_x1 + 20, pose1.bbox.y + 430); // RIGHT_ANKLE
    
    results.push_back(pose1);
    
    // === Person 2: 身高测量 ===
    PoseResult pose2;
    pose2.person_id = 2;
    pose2.detection_confidence = 0.85f;
    pose2.timestamp = std::chrono::steady_clock::now();
    pose2.bbox = cv::Rect2f(300.0f, 50.0f, 120.0f, 450.0f);
    
    pose2.keypoints.resize(17);
    pose2.keypoint_confidences.resize(17, 0.8f);
    
    float center_x2 = pose2.bbox.x + pose2.bbox.width * 0.5f;  // 360
    
    // 头部关键点
    pose2.keypoints[0] = cv::Point2f(center_x2, pose2.bbox.y + 20);      // NOSE
    pose2.keypoints[1] = cv::Point2f(center_x2 - 10, pose2.bbox.y + 15); // LEFT_EYE
    pose2.keypoints[2] = cv::Point2f(center_x2 + 10, pose2.bbox.y + 15); // RIGHT_EYE
    pose2.keypoints[3] = cv::Point2f(center_x2 - 15, pose2.bbox.y + 18); // LEFT_EAR
    pose2.keypoints[4] = cv::Point2f(center_x2 + 15, pose2.bbox.y + 18); // RIGHT_EAR
    
    // 上身关键点
    pose2.keypoints[5] = cv::Point2f(center_x2 - 30, pose2.bbox.y + 80);  // LEFT_SHOULDER
    pose2.keypoints[6] = cv::Point2f(center_x2 + 30, pose2.bbox.y + 80);  // RIGHT_SHOULDER
    pose2.keypoints[7] = cv::Point2f(center_x2 - 35, pose2.bbox.y + 140); // LEFT_ELBOW
    pose2.keypoints[8] = cv::Point2f(center_x2 + 35, pose2.bbox.y + 140); // RIGHT_ELBOW
    
    // 身高测量：双手在身体两侧，低于头部
    pose2.keypoints[9]  = cv::Point2f(center_x2 - 50, pose2.bbox.y + 180); // LEFT_WRIST
    pose2.keypoints[10] = cv::Point2f(center_x2 + 50, pose2.bbox.y + 180); // RIGHT_WRIST
    
    // 下身关键点
    pose2.keypoints[11] = cv::Point2f(center_x2 - 20, pose2.bbox.y + 200); // LEFT_HIP
    pose2.keypoints[12] = cv::Point2f(center_x2 + 20, pose2.bbox.y + 200); // RIGHT_HIP
    pose2.keypoints[13] = cv::Point2f(center_x2 - 25, pose2.bbox.y + 300); // LEFT_KNEE
    pose2.keypoints[14] = cv::Point2f(center_x2 + 25, pose2.bbox.y + 300); // RIGHT_KNEE
    pose2.keypoints[15] = cv::Point2f(center_x2 - 20, pose2.bbox.y + 430); // LEFT_ANKLE
    pose2.keypoints[16] = cv::Point2f(center_x2 + 20, pose2.bbox.y + 430); // RIGHT_ANKLE
    
    results.push_back(pose2);
    return results;
}

int main(int argc, char* argv[]) {
    std::cout << "=== 姿态分析系统简单示例 ===" << std::endl;
    
    try {
        // 创建默认配置的分析器
        auto config = PoseAnalyzer::create_default_config();
        auto analyzer = std::make_unique<PoseAnalyzer>(config);
        
        if (!analyzer->initialize()) {
            std::cerr << "分析器初始化失败" << std::endl;
            return -1;
        }
        
        std::cout << "✓ 姿态分析器初始化成功" << std::endl;
        
        // 模拟处理多帧数据
        for (int frame_id = 0; frame_id < 10; ++frame_id) {
            // 获取模拟的ByteTrack输出
            auto pose_results = simulate_bytetrack_output(frame_id);
            
            // 运行姿态分析
            auto analysis_results = analyzer->analyze(pose_results);
            
            // 显示结果
            std::cout << "\n=== 帧 " << frame_id << " ===" << std::endl;
            for (const auto& result : analysis_results) {
                std::cout << "Person " << result.id_priority_result.priority_id << ":" << std::endl;
                
                if (result.height_result.is_stable) {
                    std::cout << "  身高: " << result.height_result.estimated_height_mm << "mm" << std::endl;
                }
                
                if (result.ball_request_result.is_confirmed) {
                    std::cout << "  要球动作: 已确认" << std::endl;
                }
            }
            
            // 模拟帧间隔
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "\n=== 测试完成 ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}