/**
 * @file pose_image_with_polar.cpp
 * @brief 姿态检测 + 极坐标系统演示程序
 * @author YOLOv8 Detector Library
 * @version 1.0.3
 * @date 2025-08-07
 * 
 * 功能说明：
 * - 演示完整的双坐标系统功能
 * - 同时输出笛卡尔坐标(x,y)和极坐标(r,θ)  
 * - 支持JSON配置文件和手动配置两种方式
 * - 生成包含坐标标注的结果图片
 * 
 * 使用方法：
 * ./pose_image_with_polar
 * 
 * 输出文件：
 * pose_with_polar_result.jpg - 带有双坐标标注的结果图片
 */

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace detector;

int main() {
    std::cout << "=== 姿态检测+极坐标系统测试 ===" << std::endl;
    
    // 1. 智能查找并加载模型文件
    std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
    if (model_path.empty()) {
        std::cerr << "❌ 无法找到姿态检测模型文件" << std::endl;
        return -1;
    }
    
    PoseDetectorLib detector(model_path);
    std::cout << "✓ 检测器创建成功" << std::endl;
    
    // 2. 启用跟踪功能
    detector.enable_tracking(true);
    std::cout << "✓ 启用跟踪功能" << std::endl;
    
    // 3. 智能查找并加载Homography标定文件（包含极坐标配置）
    std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
    bool calib_loaded = false;
    if (!calib_path.empty()) {
        calib_loaded = detector.load_calibration(calib_path);
        if (calib_loaded) {
            std::cout << "✓ Homography标定+极坐标配置加载成功" << std::endl;
        }
    }
    
    if (!calib_loaded) {
        std::cout << "⚠ 标定文件未找到，手动配置极坐标系统" << std::endl;
        // 手动配置极坐标系统（原点偏移设为0,0）
        detector.set_polar_coordinate_system(true, 0.0f, 0.0f);
        std::cout << "✓ 手动启用极坐标系统" << std::endl;
    }
    
    // 4. 设置置信度阈值
    detector.set_confidence_threshold(0.25f);
    
    // 5. 智能查找并加载测试图片
    std::vector<std::string> image_search_paths = {
        "./pose.jpg",
        "../imgs/pose.jpg", 
        "../../imgs/pose.jpg",
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
    
    // 6. 进行姿态检测
    std::cout << "\n开始姿态检测..." << std::endl;
    auto results = detector.detect(image);
    
    std::cout << "检测完成，推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
    std::cout << "检测到 " << results.size() << " 个人" << std::endl;
    
    // 7. 显示检测结果（包含极坐标）
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        std::cout << "\n=== 人员[" << i << "] ===" << std::endl;
        std::cout << "跟踪ID: " << result.person_id << std::endl;
        std::cout << "置信度: " << std::fixed << std::setprecision(2) << result.confidence << std::endl;
        
        // ROI框信息
        std::cout << "ROI框: (" 
                  << result.bbox.x << ", " << result.bbox.y 
                  << ", " << result.bbox.width << ", " << result.bbox.height << ")" << std::endl;
        
        // 计算ROI框底部中点
        float roi_bottom_center_x = result.bbox.x + result.bbox.width / 2.0f;
        float roi_bottom_center_y = result.bbox.y + result.bbox.height;
        
        std::cout << "ROI底部中点: (" << std::fixed << std::setprecision(1) 
                  << roi_bottom_center_x << ", " << roi_bottom_center_y << ")" << std::endl;
        
        // 笛卡尔坐标
        if (result.has_ground_position) {
            std::cout << "笛卡尔坐标: (" << std::fixed << std::setprecision(1) 
                      << result.ground_position.x << ", " << result.ground_position.y << ")mm" << std::endl;
        } else {
            std::cout << "笛卡尔坐标: 未计算" << std::endl;
        }
        
        // 极坐标
        if (result.has_polar_position) {
            std::cout << "极坐标: 距离=" << std::fixed << std::setprecision(1) 
                      << result.polar_position.r << "mm, 角度=" 
                      << std::setprecision(1) << result.polar_position.theta_degrees() << "°" << std::endl;
        } else {
            std::cout << "极坐标: 未计算" << std::endl;
        }
    }
    
    // 8. 绘制检测结果（包含极坐标显示）
    cv::Mat result_image = image.clone();
    
    for (const auto& result : results) {
        // 绘制ROI框
        cv::Scalar box_color = cv::Scalar(0, 255, 0);  // 绿色框
        cv::rectangle(result_image, result.bbox, box_color, 3);
        
        // 计算ROI框底部中点
        float roi_bottom_center_x = result.bbox.x + result.bbox.width / 2.0f;
        float roi_bottom_center_y = result.bbox.y + result.bbox.height;
        cv::Point2f bottom_center(roi_bottom_center_x, roi_bottom_center_y);
        
        // 绘制底部中点（紫色圆点）
        cv::circle(result_image, bottom_center, 6, cv::Scalar(255, 0, 255), -1);
        
        // 绘制ID和置信度标签
        std::string label = "ID:" + std::to_string(result.person_id) + 
                           " (" + std::to_string(int(result.confidence * 100)) + "%)";
        cv::putText(result_image, label, 
                   cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
        
        int text_y_offset = result.bbox.y + result.bbox.height + 25;
        
        // 显示笛卡尔坐标（黄色文字）
        if (result.has_ground_position) {
            std::string cartesian_info = "Cart: (" + 
                std::to_string(int(result.ground_position.x)) + "," +
                std::to_string(int(result.ground_position.y)) + ")mm";
            cv::putText(result_image, cartesian_info,
                       cv::Point(result.bbox.x, text_y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
            text_y_offset += 20;
        }
        
        // 显示极坐标（青色文字）
        if (result.has_polar_position) {
            std::string polar_info = "Polar: (" + 
                std::to_string(int(result.polar_position.r)) + "mm," +
                std::to_string(int(result.polar_position.theta_degrees())) + "deg)";
            cv::putText(result_image, polar_info,
                       cv::Point(result.bbox.x, text_y_offset),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
        }
        
        // 如果有极坐标，绘制指向原点的方向线
        if (result.has_polar_position && result.has_ground_position) {
            // 这里可以添加绘制极坐标方向线的代码
            // 从底部中点指向极坐标原点方向的线段
        }
    }
    
    // 9. 保存结果图片
    const std::string output_path = "pose_with_polar_result.jpg";
    bool saved = cv::imwrite(output_path, result_image);
    
    if (saved) {
        std::cout << "\n✅ 检测结果已保存到: " << output_path << std::endl;
    } else {
        std::cout << "\n⚠ 保存结果图片失败" << std::endl;
    }
    
    std::cout << "\n=== 极坐标功能测试完成 ===" << std::endl;
    return 0;
}