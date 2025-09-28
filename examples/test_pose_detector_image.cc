#include <iostream>
#include <opencv2/opencv.hpp>
#include "PoseDetector.h"

// COCO 17关键点连接关系 (骨架)
const std::vector<std::pair<int, int>> skeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
    {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, 
    {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
    {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

// 绘制姿态检测结果
void draw_pose_results(cv::Mat& frame, const std::vector<PoseResult>& results) {
    for (const auto& pose : results) {
        // 绘制边界框
        cv::rectangle(frame, pose.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 显示人员ID和置信度
        std::string label = "ID:" + std::to_string(pose.person_id) + 
                           " (" + std::to_string((int)(pose.confidence * 100)) + "%)";
        cv::putText(frame, label, cv::Point(pose.bbox.x, pose.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // 绘制关键点
        std::cout << "🔍 关键点详细信息:" << std::endl;
        for (size_t i = 0; i < pose.keypoints.size(); i++) {
            const auto& kpt = pose.keypoints[i];
            float score = pose.keypoint_scores[i];
            
            // 检查关键点是否在图像范围内
            bool in_image = (kpt.x >= 0 && kpt.x < frame.cols && 
                           kpt.y >= 0 && kpt.y < frame.rows);
                           
            std::cout << "    关键点[" << i << "]: 坐标(" << kpt.x << ", " << kpt.y 
                     << "), 置信度=" << score << ", 在图像内=" << (in_image ? "是" : "否") << std::endl;
            
            if (score > 0.3f && in_image) {  // 只显示置信度较高且在图像内的关键点
                cv::circle(frame, kpt, 6, cv::Scalar(0, 0, 255), -1);
                // 显示关键点索引
                cv::putText(frame, std::to_string(i), 
                           cv::Point(kpt.x + 8, kpt.y - 8),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // 绘制骨架连接
        for (const auto& connection : skeleton) {
            int idx1 = connection.first - 1;   // COCO索引从1开始，转换为0开始
            int idx2 = connection.second - 1;
            
            if (idx1 >= 0 && idx1 < pose.keypoints.size() && 
                idx2 >= 0 && idx2 < pose.keypoints.size() &&
                pose.keypoint_scores[idx1] > 0.3f && 
                pose.keypoint_scores[idx2] > 0.3f) {
                
                cv::line(frame, pose.keypoints[idx1], pose.keypoints[idx2], 
                        cv::Scalar(255, 0, 0), 2);
            }
        }
        
        // 显示地面坐标 (如果有)
        if (pose.has_ground_position) {
            std::string ground_pos = "Ground: (" + 
                std::to_string((int)pose.ground_position.x) + ", " +
                std::to_string((int)pose.ground_position.y) + ")";
            cv::putText(frame, ground_pos, 
                       cv::Point(pose.bbox.x, pose.bbox.y + pose.bbox.height + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
    }
}

int main() {
    std::cout << "=== PoseDetector 图片测试程序 ===" << std::endl;
    
    // 1. 创建PoseDetector
    std::cout << "正在创建PoseDetector..." << std::endl;
    PoseDetector detector("models/Q_yolov8_pose.rknn");
    
    // 2. 配置
    detector.set_confidence_threshold(0.3f);
    detector.enable_tracking(false);  // 单图无需跟踪
    std::cout << "检测器配置完成" << std::endl;
    
    // 3. 预热检测器
    std::cout << "正在预热检测器..." << std::endl;
    cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    detector.detect(dummy_frame);
    std::cout << "预热完成！" << std::endl;
    
    // 4. 测试图片
    std::vector<std::string> image_paths = {
        "../pose_analysis/imgs/pose.jpg",
        "../pose_analysis/imgs/pose1.jpg",
        "../pose_analysis/imgs/pose2.jpg",
        "../pose_analysis/imgs/pose3.jpg"
    };
    
    for (size_t img_idx = 0; img_idx < image_paths.size(); ++img_idx) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "测试图片 " << (img_idx + 1) << ": " << image_paths[img_idx] << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // 加载图片
        cv::Mat image = cv::imread(image_paths[img_idx]);
        if (image.empty()) {
            std::cerr << "❌ 无法加载图片: " << image_paths[img_idx] << std::endl;
            continue;
        }
        
        std::cout << "✓ 图片加载成功 (" << image.cols << "x" << image.rows << ")" << std::endl;
        
        // 进行检测
        auto detect_start = std::chrono::high_resolution_clock::now();
        std::vector<PoseResult> results = detector.detect(image);
        auto detect_end = std::chrono::high_resolution_clock::now();
        
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start);
        
        std::cout << "✓ 检测完成，推理时间: " << inference_time.count() << "ms" << std::endl;
        std::cout << "✓ 检测到 " << results.size() << " 个人" << std::endl;
        
        if (results.empty()) {
            std::cout << "  无检测结果，跳过" << std::endl;
            continue;
        }
        
        // 显示详细信息
        for (size_t i = 0; i < results.size(); i++) {
            const auto& pose = results[i];
            std::cout << "\n--- Person " << i << " ---" << std::endl;
            std::cout << "ID: " << pose.person_id << ", 置信度: " << pose.confidence << std::endl;
            std::cout << "ROI框: (" << pose.bbox.x << ", " << pose.bbox.y << ", " 
                     << pose.bbox.width << ", " << pose.bbox.height << ")" << std::endl;
            std::cout << "关键点数量: " << pose.keypoints.size() << std::endl;
        }
        
        // 绘制结果
        cv::Mat result_image = image.clone();
        draw_pose_results(result_image, results);
        
        // 保存结果
        std::string output_filename = "old_pose_detector_result_" + std::to_string(img_idx + 1) + ".jpg";
        bool saved = cv::imwrite(output_filename, result_image);
        
        if (saved) {
            std::cout << "\n✅ 结果图片已保存到: " << output_filename << std::endl;
        } else {
            std::cout << "\n❌ 保存结果图片失败！" << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ 所有图片测试完成" << std::endl;
    
    return 0;
}