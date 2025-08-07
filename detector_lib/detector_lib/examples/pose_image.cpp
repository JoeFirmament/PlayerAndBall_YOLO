/*-------------------------------------------
          纯姿态检测图片程序
          
只测试姿态检测功能，输入图片
运行: ./pose_image [图片路径]
-------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "detector_lib.h"

int main(int argc, char* argv[]) {
    std::cout << "=== 姿态检测图片程序 ===" << std::endl;
    
    std::string image_path = "../../imgs/pose.jpg";  // 默认图片
    std::string model_path = "../../models/Q_yolov8_pose.rknn";
    
    if (argc >= 2) {
        image_path = argv[1];
    }
    
    std::cout << "图片: " << image_path << std::endl;
    std::cout << "模型: " << model_path << std::endl;
    
    try {
        // 读取图片
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "❌ 无法读取图片: " << image_path << std::endl;
            return -1;
        }
        
        std::cout << "✅ 图片加载成功: " << image.cols << "x" << image.rows << std::endl;
        
        // 创建检测器
        std::cout << "正在创建姿态检测器..." << std::endl;
        detector::PoseDetectorLib detector(model_path);
        
        // 配置参数 - 使用工作版本的阈值
        detector.set_confidence_threshold(0.5f);
        detector.enable_tracking(true);
        
        // 检测
        std::cout << "正在检测..." << std::endl;
        auto results = detector.detect(image);
        
        std::cout << "✅ 检测完成!" << std::endl;
        std::cout << "推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        std::cout << "检测器状态: " << (int)detector.get_status() << std::endl;
        std::cout << "检测到人员数量: " << results.size() << std::endl;
        
        // 分析结果并绘制
        cv::Mat result_image = image.clone();
        if (results.empty()) {
            std::cout << "未检测到人员姿态" << std::endl;
        } else {
            std::cout << "\n📊 检测统计:" << std::endl;
            std::cout << "  👤 人员: " << results.size() << "个" << std::endl;
            
            std::cout << "\n📋 详细结果:" << std::endl;
            for (size_t i = 0; i < results.size(); i++) {
                const auto& pose = results[i];
                std::cout << "  人员" << (i+1) << ": ID=" << pose.person_id 
                         << ", 置信度=" << std::fixed << std::setprecision(2) << pose.confidence;
                
                // 统计有效关键点
                int valid_keypoints = 0;
                for (const auto& score : pose.keypoint_scores) {
                    if (score > 0.5f) valid_keypoints++;
                }
                std::cout << ", 关键点=" << valid_keypoints << "/17";
                
                if (pose.has_ground_position) {
                    std::cout << ", 地面坐标=(" << std::fixed << std::setprecision(1) 
                             << pose.ground_position.x << "," << pose.ground_position.y << ")";
                }
                std::cout << std::endl;
                
                // 绘制检测框 (红色)
                cv::rectangle(result_image, pose.bbox, cv::Scalar(0, 0, 255), 3);
                
                // 绘制置信度
                std::string conf_text = "Person " + std::to_string(i+1) + ": " + std::to_string(pose.confidence).substr(0, 4);
                cv::putText(result_image, conf_text, cv::Point(pose.bbox.x, pose.bbox.y - 10), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
                
                // 绘制17个关键点 (绿色圆点)
                for (size_t j = 0; j < pose.keypoints.size() && j < 17; j++) {
                    if (pose.keypoint_scores[j] > 0.5f) {
                        cv::circle(result_image, pose.keypoints[j], 4, cv::Scalar(0, 255, 0), -1);
                        
                        // 在关键点旁边显示编号
                        cv::putText(result_image, std::to_string(j), 
                                   cv::Point(pose.keypoints[j].x + 5, pose.keypoints[j].y - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
                    }
                }
                
                // 绘制骨架连线 (黄色线条)
                // COCO 17关键点的骨架连接定义
                int skeleton[][2] = {
                    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
                    {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
                    {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
                    {2, 4}, {3, 5}, {4, 6}, {5, 7}
                };
                
                for (auto& bone : skeleton) {
                    int kpt_a = bone[0] - 1; // 转换为0索引
                    int kpt_b = bone[1] - 1;
                    
                    if (kpt_a >= 0 && kpt_a < 17 && kpt_b >= 0 && kpt_b < 17 &&
                        pose.keypoint_scores[kpt_a] > 0.5f && pose.keypoint_scores[kpt_b] > 0.5f) {
                        cv::line(result_image, pose.keypoints[kpt_a], pose.keypoints[kpt_b], 
                                cv::Scalar(0, 255, 255), 2);
                    }
                }
            }
        }
        
        // 保存结果图片
        std::string output_filename = "pose_detection_result.jpg";
        bool save_success = cv::imwrite(output_filename, result_image);
        if (save_success) {
            std::cout << "\n✅ 结果图片已保存: " << output_filename << std::endl;
        } else {
            std::cout << "\n❌ 保存结果图片失败!" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 程序异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}