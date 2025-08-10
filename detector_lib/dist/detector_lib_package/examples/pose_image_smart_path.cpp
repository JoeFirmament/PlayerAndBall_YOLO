/**
 * @file pose_image_smart_path.cpp
 * @brief 智能路径解析的姿态检测示例
 * @author YOLOv8 Detector Library
 * @version 1.0.3
 * @date 2025-08-07
 * 
 * 功能说明：
 * - 自动查找模型文件和标定文件
 * - 支持环境变量配置
 * - 无需硬编码路径，适配各种用户环境
 * 
 * 支持的环境变量：
 * DETECTOR_MODEL_PATH - 模型文件目录
 * DETECTOR_DATA_PATH  - 标定数据目录
 * 
 * 使用方法：
 * ./pose_image_smart_path
 * 
 * 或设置环境变量：
 * export DETECTOR_MODEL_PATH=/my/models
 * export DETECTOR_DATA_PATH=/my/data
 * ./pose_image_smart_path
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

int main() {
    std::cout << "=== 智能路径解析姿态检测测试 ===" << std::endl;
    
    // 1. 自动查找模型文件
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    if (model_path.empty()) {
        std::cerr << "❌ 无法找到姿态检测模型文件" << std::endl;
        std::cerr << "请设置环境变量 DETECTOR_MODEL_PATH 或将模型文件放在以下位置之一：" << std::endl;
        auto search_paths = PathUtils::get_model_search_paths();
        for (const auto& path : search_paths) {
            std::cerr << "  - " << path << std::endl;
        }
        return -1;
    }
    
    // 2. 创建检测器
    PoseDetectorLib detector(model_path);
    std::cout << "✓ 检测器创建成功" << std::endl;
    
    // 3. 尝试加载标定文件（可选）
    std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
    if (!calib_path.empty()) {
        if (detector.load_calibration(calib_path)) {
            std::cout << "✓ 标定文件加载成功，启用坐标映射功能" << std::endl;
        } else {
            std::cout << "⚠️ 标定文件加载失败，使用纯检测模式" << std::endl;
        }
    } else {
        std::cout << "⚠️ 未找到标定文件，使用纯检测模式" << std::endl;
    }
    
    // 4. 查找测试图片（多种可能的位置）
    std::vector<std::string> image_search_paths = {
        "./pose.jpg",
        "../imgs/pose.jpg", 
        "../imgs/pose.jpg",
        "./test_image.jpg",
        "../test_image.jpg"
    };
    
    cv::Mat image;
    std::string used_image_path;
    
    for (const auto& img_path : image_search_paths) {
        image = cv::imread(img_path);
        if (!image.empty()) {
            used_image_path = img_path;
            break;
        }
    }
    
    if (image.empty()) {
        std::cout << "⚠️ 未找到测试图片，创建模拟图像进行测试" << std::endl;
        // 创建一个简单的测试图像
        image = cv::Mat::zeros(640, 640, CV_8UC3);
        cv::rectangle(image, cv::Rect(200, 200, 200, 300), cv::Scalar(100, 150, 200), -1);
        used_image_path = "generated";
    } else {
        std::cout << "✓ 成功加载图片: " << used_image_path << " (" << image.cols << "x" << image.rows << ")" << std::endl;
    }
    
    // 5. 进行检测
    std::cout << "\n开始姿态检测..." << std::endl;
    auto results = detector.detect(image);
    
    std::cout << "检测完成，推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
    std::cout << "检测到 " << results.size() << " 个人" << std::endl;
    
    // 6. 显示检测结果
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        std::cout << "\n=== 人员[" << i << "] ===" << std::endl;
        std::cout << "跟踪ID: " << result.person_id << std::endl;
        std::cout << "置信度: " << std::fixed << std::setprecision(2) << result.confidence << std::endl;
        
        if (result.has_ground_position) {
            std::cout << "世界坐标: (" << result.ground_position.x << ", " << result.ground_position.y << ")mm" << std::endl;
        }
        
        if (result.has_polar_position) {
            std::cout << "极坐标: 距离=" << result.polar_position.r << "mm, 角度=" << result.polar_position.theta_degrees() << "°" << std::endl;
        }
    }
    
    // 7. 保存结果（可选）
    if (results.size() > 0) {
        cv::Mat result_image = image.clone();
        
        for (const auto& result : results) {
            // 绘制检测框
            cv::rectangle(result_image, result.bbox, cv::Scalar(0, 255, 0), 2);
            
            // 绘制标签
            std::string label = "ID:" + std::to_string(result.person_id);
            cv::putText(result_image, label, 
                       cv::Point(result.bbox.x, result.bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        }
        
        std::string output_path = "smart_path_pose_result.jpg";
        bool saved = cv::imwrite(output_path, result_image);
        if (saved) {
            std::cout << "\n✅ 检测结果已保存到: " << output_path << std::endl;
        }
    }
    
    std::cout << "\n=== 智能路径解析测试完成 ===" << std::endl;
    return 0;
}