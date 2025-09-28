#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"

using namespace detector;

int main() {
    std::cout << "=== 关键点调试测试 ===\n";
    
    // 创建检测器
    const std::string model_path = "../../models/Q_yolov8_pose.rknn";
    PoseDetectorLib detector(model_path);
    
    // 关闭跟踪
    detector.enable_tracking(false);
    detector.set_confidence_threshold(0.25f);
    
    // 加载图片
    const std::string image_path = "../../imgs/pose.jpg";
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        std::cerr << "❌ 无法读取图片: " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "✓ 图片加载成功: " << image.cols << "x" << image.rows << std::endl;
    
    // 检测
    auto results = detector.detect(image);
    std::cout << "✓ 检测完成，发现 " << results.size() << " 个人\n";
    
    // 详细分析每个检测结果
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        std::cout << "\n========== 人员 [" << i << "] ==========\n";
        std::cout << "ID: " << result.person_id << ", 置信度: " << result.confidence << "\n";
        std::cout << "ROI框: (" << result.bbox.x << ", " << result.bbox.y 
                  << ", " << result.bbox.width << ", " << result.bbox.height << ")\n";
        std::cout << "关键点数组大小: " << result.keypoints.size() << "\n";
        std::cout << "关键点置信度数组大小: " << result.keypoint_scores.size() << "\n";
        
        // 检查每个关键点
        for (size_t j = 0; j < result.keypoints.size(); j++) {
            const cv::Point2f& kpt = result.keypoints[j];
            float score = (j < result.keypoint_scores.size()) ? result.keypoint_scores[j] : -1.0f;
            
            bool in_bounds = (kpt.x >= 0 && kpt.x < image.cols && kpt.y >= 0 && kpt.y < image.rows);
            bool valid_score = (score >= 0.0f && score <= 1.0f);
            
            std::cout << "  关键点[" << std::setw(2) << j << "]: "
                     << "坐标(" << std::setw(7) << std::fixed << std::setprecision(1) << kpt.x 
                     << "," << std::setw(7) << kpt.y << ") "
                     << "置信度=" << std::setw(5) << score 
                     << " [" << (in_bounds ? "✓边界内" : "❌边界外") 
                     << ", " << (valid_score ? "✓有效分数" : "❌无效分数") << "]\n";
        }
        
        // 统计有效关键点
        int valid_keypoints = 0;
        int high_conf_keypoints = 0;
        for (size_t j = 0; j < result.keypoints.size(); j++) {
            const cv::Point2f& kpt = result.keypoints[j];
            float score = (j < result.keypoint_scores.size()) ? result.keypoint_scores[j] : 0.0f;
            
            if (kpt.x >= 0 && kpt.x < image.cols && kpt.y >= 0 && kpt.y < image.rows && score > 0.0f) {
                valid_keypoints++;
                if (score > 0.3f) {
                    high_conf_keypoints++;
                }
            }
        }
        
        std::cout << "\n📊 关键点统计:\n";
        std::cout << "  总关键点: " << result.keypoints.size() << "\n";
        std::cout << "  有效关键点: " << valid_keypoints << "\n";  
        std::cout << "  高置信度关键点(>0.3): " << high_conf_keypoints << "\n";
        
        if (valid_keypoints == 0) {
            std::cout << "🚨 警告：此人员没有任何有效的关键点！\n";
        } else {
            std::cout << "✅ 此人员有 " << valid_keypoints << " 个有效关键点\n";
        }
    }
    
    // 绘制所有关键点到图片上（包括无效的，用于调试）
    cv::Mat debug_image = image.clone();
    
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        // 绘制ROI框
        cv::rectangle(debug_image, result.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 绘制ID标签
        std::string label = "ID:" + std::to_string(result.person_id) + 
                           " (" + std::to_string((int)(result.confidence * 100)) + "%)";
        cv::putText(debug_image, label, cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // 绘制所有关键点（无论是否有效）
        for (size_t j = 0; j < result.keypoints.size(); j++) {
            const cv::Point2f& kpt = result.keypoints[j];
            float score = (j < result.keypoint_scores.size()) ? result.keypoint_scores[j] : 0.0f;
            
            // 根据关键点状态选择颜色
            cv::Scalar color;
            if (kpt.x >= 0 && kpt.x < image.cols && kpt.y >= 0 && kpt.y < image.rows) {
                if (score > 0.3f) {
                    color = cv::Scalar(0, 255, 0);  // 绿色：有效高置信度
                } else if (score > 0.0f) {
                    color = cv::Scalar(0, 255, 255);  // 黄色：有效低置信度
                } else {
                    color = cv::Scalar(255, 0, 0);  // 蓝色：坐标有效但置信度无效
                }
            } else {
                color = cv::Scalar(0, 0, 255);  // 红色：坐标无效
            }
            
            // 绘制关键点
            cv::circle(debug_image, kpt, 4, color, -1);
            
            // 添加关键点索引标签
            cv::putText(debug_image, std::to_string(j),
                       cv::Point(kpt.x + 6, kpt.y - 6),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // 添加图例说明
    int legend_y = 30;
    cv::putText(debug_image, "Legend:", cv::Point(10, legend_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    legend_y += 25;
    cv::circle(debug_image, cv::Point(30, legend_y), 4, cv::Scalar(0, 255, 0), -1);
    cv::putText(debug_image, "Valid High Conf (>0.3)", cv::Point(40, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    legend_y += 20;
    cv::circle(debug_image, cv::Point(30, legend_y), 4, cv::Scalar(0, 255, 255), -1);
    cv::putText(debug_image, "Valid Low Conf (0-0.3)", cv::Point(40, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    legend_y += 20;
    cv::circle(debug_image, cv::Point(30, legend_y), 4, cv::Scalar(255, 0, 0), -1);
    cv::putText(debug_image, "Invalid Coord", cv::Point(40, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    legend_y += 20;
    cv::circle(debug_image, cv::Point(30, legend_y), 4, cv::Scalar(0, 0, 255), -1);
    cv::putText(debug_image, "Out of Bounds", cv::Point(40, legend_y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    
    // 保存调试图片
    std::string output_filename = "keypoints_debug_result.jpg";
    bool saved = cv::imwrite(output_filename, debug_image);
    
    if (saved) {
        std::cout << "\n✅ 关键点调试图片已保存: " << output_filename << std::endl;
    } else {
        std::cout << "\n❌ 保存调试图片失败" << std::endl;
    }
    
    return 0;
}