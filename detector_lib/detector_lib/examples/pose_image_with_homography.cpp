#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"

using namespace detector;

int main() {
    std::cout << "=== 姿态检测+Homography坐标映射测试 ===" << std::endl;
    
    // 1. 创建检测器实例
    const std::string model_path = "../../models/Q_yolov8_pose.rknn";
    PoseDetectorLib detector(model_path);
    
    std::cout << "✓ 检测器创建成功" << std::endl;
    
    // 2. 启用跟踪功能（简化版ID分配）
    detector.enable_tracking(true);
    std::cout << "✓ 启用跟踪功能" << std::endl;
    
    // 3. 尝试加载Homography标定文件
    bool calib_loaded = detector.load_calibration("../../data/2025_8_6_1280_720.json");
    if (calib_loaded) {
        std::cout << "✓ Homography标定加载成功" << std::endl;
    } else {
        std::cout << "⚠ Homography标定文件未找到，将跳过地面坐标计算" << std::endl;
    }
    
    // 4. 设置置信度阈值
    detector.set_confidence_threshold(0.25f);
    
    // 5. 加载测试图片
    const std::string image_path = "../../imgs/pose.jpg";
    cv::Mat image = cv::imread(image_path);
    
    if (image.empty()) {
        std::cerr << "❌ 无法加载图片: " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "✓ 成功加载图片: " << image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    
    // 6. 进行姿态检测
    std::cout << "\n开始姿态检测..." << std::endl;
    auto results = detector.detect(image);
    
    std::cout << "检测完成，推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
    std::cout << "检测到 " << results.size() << " 个人" << std::endl;
    
    // 7. 显示检测结果（简化输出）
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        // ROI框信息
        std::cout << "\n人员[" << i << "] ROI框: (" 
                  << result.bbox.x << ", " << result.bbox.y 
                  << ", " << result.bbox.width << ", " << result.bbox.height << ")" << std::endl;
        
        // 计算ROI框底部中点
        float roi_bottom_center_x = result.bbox.x + result.bbox.width / 2.0f;
        float roi_bottom_center_y = result.bbox.y + result.bbox.height;
        
        std::cout << "ROI底部中点: (" << std::fixed << std::setprecision(1) 
                  << roi_bottom_center_x << ", " << roi_bottom_center_y << ")" << std::endl;
        
        // 如果有地面坐标映射，显示世界坐标
        if (result.has_ground_position) {
            std::cout << "世界坐标: (" << std::fixed << std::setprecision(1) 
                      << result.ground_position.x << ", " << result.ground_position.y << ")mm" << std::endl;
        } else {
            std::cout << "世界坐标: 未计算" << std::endl;
        }
    }
    
    // 8. 绘制检测结果
    cv::Mat result_image = image.clone();
    
    for (const auto& result : results) {
        // 绘制ROI框
        cv::Scalar box_color = cv::Scalar(0, 255, 0);  // 绿色框
        cv::rectangle(result_image, result.bbox, box_color, 3);
        
        // 计算ROI框底部中点
        float roi_bottom_center_x = result.bbox.x + result.bbox.width / 2.0f;
        float roi_bottom_center_y = result.bbox.y + result.bbox.height;
        cv::Point2f bottom_center(roi_bottom_center_x, roi_bottom_center_y);
        
        // 绘制底部中点（大红色圆点）
        cv::circle(result_image, bottom_center, 8, cv::Scalar(0, 0, 255), -1);
        cv::circle(result_image, bottom_center, 12, cv::Scalar(0, 0, 255), 2);
        
        // 绘制ID和置信度标签
        std::string label = "ID:" + std::to_string(result.person_id) + 
                           " (" + std::to_string(int(result.confidence * 100)) + "%)";
        cv::putText(result_image, label, 
                   cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
        
        // 显示ROI信息
        std::string roi_info = "ROI: " + std::to_string(result.bbox.width) + "x" + std::to_string(result.bbox.height);
        cv::putText(result_image, roi_info,
                   cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 25),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 显示底部中点坐标
        std::string center_info = "Center: (" + 
            std::to_string(int(roi_bottom_center_x)) + "," +
            std::to_string(int(roi_bottom_center_y)) + ")";
        cv::putText(result_image, center_info,
                   cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 45),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // 如果有世界坐标，显示世界坐标信息
        if (result.has_ground_position) {
            std::string world_info = "World: (" + 
                std::to_string(int(result.ground_position.x)) + "," +
                std::to_string(int(result.ground_position.y)) + ")mm";
            cv::putText(result_image, world_info,
                       cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 65),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }
        
        // 绘制从底部中点到图像边界的垂直线（可选，用于可视化）
        cv::line(result_image, bottom_center, 
                cv::Point(roi_bottom_center_x, image.rows), 
                cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }
    
    // 9. 保存结果图片
    const std::string output_path = "pose_with_homography_result.jpg";
    bool saved = cv::imwrite(output_path, result_image);
    
    if (saved) {
        std::cout << "\n✅ 检测结果已保存到: " << output_path << std::endl;
    } else {
        std::cout << "\n⚠ 保存结果图片失败" << std::endl;
    }
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}