#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"

using namespace detector;

// 相机在世界坐标系中的位置 (毫米)
const float CAMERA_WORLD_X = 0.0f;
const float CAMERA_WORLD_Y = -400.0f;    // 相机Y坐标：-40cm 
const float CAMERA_HEIGHT_MM = 1170.0f;  // 相机高度：117cm

int main() {
    std::cout << "=== 基于ROI的身高测量系统 ===\n";
    
    // 1. 创建检测器
    const std::string model_path = "../../models/Q_yolov8_pose.rknn";
    PoseDetectorLib detector(model_path);
    
    // 2. 配置检测器
    detector.enable_tracking(false);
    detector.set_confidence_threshold(0.25f);
    
    // 3. 加载Homography标定数据
    bool calib_loaded = detector.load_calibration("../../data/2025_8_6_1280_720.json");
    if (!calib_loaded) {
        std::cerr << "❌ 无法加载Homography标定文件" << std::endl;
        return -1;
    }
    std::cout << "✅ Homography标定加载成功" << std::endl;
    
    // 4. 测试多张图片
    std::vector<std::string> test_images = {
        "../../imgs/pose.jpg",
        "../../imgs/pose1.jpg", 
        "../../imgs/pose2.jpg",
        "../../imgs/pose3.jpg"
    };
    
    for (const auto& image_path : test_images) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "测试图片: " << image_path << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        // 加载图片
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cout << "❌ 无法读取图片: " << image_path << std::endl;
            continue;  // 继续处理下一张图片
        }
    
    std::cout << "✓ 图片加载成功: " << image.cols << "x" << image.rows << std::endl;
    
    // 5. 进行检测
    auto results = detector.detect(image);
    std::cout << "✓ 检测到 " << results.size() << " 个人" << std::endl;
    
    if (results.empty()) {
        std::cout << "  无人员检测结果，跳过身高计算" << std::endl;
        return 0;
    }
    
    // 6. 分析每个检测到的人员
    cv::Mat result_image = image.clone();
    
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        
        std::cout << "\n--- 人员 [" << i << "] 身高计算 ---" << std::endl;
        std::cout << "置信度: " << std::fixed << std::setprecision(3) << result.confidence << std::endl;
        
        // ROI框信息
        cv::Rect bbox = result.bbox;
        std::cout << "ROI框: (" << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << ")" << std::endl;
        
        // 计算ROI框的关键位置
        float roi_center_x = bbox.x + bbox.width / 2.0f;
        float roi_top_y = bbox.y;
        float roi_bottom_y = bbox.y + bbox.height;
        
        std::cout << "ROI中心X: " << std::fixed << std::setprecision(1) << roi_center_x << std::endl;
        std::cout << "ROI顶部Y: " << roi_top_y << std::endl; 
        std::cout << "ROI底部Y: " << roi_bottom_y << std::endl;
        
        // 使用地面坐标（脚底位置）进行身高计算
        if (result.has_ground_position) {
            cv::Point2f ground_pos = result.ground_position;
            std::cout << "地面坐标: (" << std::fixed << std::setprecision(1) 
                     << ground_pos.x << ", " << ground_pos.y << ")mm" << std::endl;
            
            // 计算相机到人员地面位置的距离
            float dx = ground_pos.x - CAMERA_WORLD_X;
            float dy = ground_pos.y - CAMERA_WORLD_Y;
            float ground_distance = sqrt(dx*dx + dy*dy);
            
            std::cout << "地面距离: " << ground_distance << "mm (" << (ground_distance/1000.0f) << "m)" << std::endl;
            
            // 基于ROI高度像素比例
            float roi_height_pixels = bbox.height;
            float image_height_pixels = image.rows;
            float roi_height_ratio = roi_height_pixels / image_height_pixels;
            
            std::cout << "ROI高度比例: " << roi_height_pixels << "/" << image_height_pixels 
                     << " = " << std::fixed << std::setprecision(4) << roi_height_ratio << std::endl;
            
            // 估算身高：基于相机视野和距离的几何关系
            // 假设相机垂直视野角度约为45度（这个需要根据实际相机参数调整）
            float camera_fov_vertical_rad = 45.0f * CV_PI / 180.0f;  // 45度转弧度
            
            // 在地面距离处，相机视野覆盖的实际高度
            float ground_fov_height_mm = 2.0f * ground_distance * tan(camera_fov_vertical_rad / 2.0f);
            
            // ROI在实际世界中对应的高度
            float estimated_height_mm = roi_height_ratio * ground_fov_height_mm;
            
            std::cout << "地面视野高度: " << std::fixed << std::setprecision(1) << ground_fov_height_mm << "mm" << std::endl;
            std::cout << "📏 估算身高: " << estimated_height_mm << "mm (" 
                     << (estimated_height_mm/10.0f) << "cm)" << std::endl;
            
            // 在图像上绘制结果
            // 绘制ROI框
            cv::rectangle(result_image, bbox, cv::Scalar(0, 255, 0), 3);
            
            // 绘制底部中心点
            cv::Point2f bottom_center(roi_center_x, roi_bottom_y);
            cv::circle(result_image, bottom_center, 8, cv::Scalar(0, 0, 255), -1);
            
            // 绘制身高信息
            std::string height_text = "Height: " + std::to_string(int(estimated_height_mm/10.0f)) + "cm";
            cv::putText(result_image, height_text,
                       cv::Point(bbox.x, bbox.y - 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
            
            // 绘制ID和置信度
            std::string id_text = "ID:" + std::to_string(result.person_id) + 
                                 " (" + std::to_string(int(result.confidence * 100)) + "%)";
            cv::putText(result_image, id_text,
                       cv::Point(bbox.x, bbox.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            
            // 绘制地面坐标信息
            std::string ground_text = "Ground: (" + 
                std::to_string(int(ground_pos.x)) + "," +
                std::to_string(int(ground_pos.y)) + ")mm";
            cv::putText(result_image, ground_text,
                       cv::Point(bbox.x, bbox.y + bbox.height + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
                       
        } else {
            std::cout << "⚠ 无地面坐标数据，无法计算身高" << std::endl;
            
            // 仍然绘制ROI框
            cv::rectangle(result_image, bbox, cv::Scalar(0, 255, 0), 3);
            
            std::string no_height_text = "No Height Data";
            cv::putText(result_image, no_height_text,
                       cv::Point(bbox.x, bbox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
    }
    
    // 7. 保存结果图片
    std::string output_filename = "height_measurement_result.jpg";
    bool saved = cv::imwrite(output_filename, result_image);
    
    if (saved) {
        std::cout << "\n✅ 结果图片已保存: " << output_filename << std::endl;
    } else {
        std::cout << "\n❌ 保存结果图片失败" << std::endl;
    }
    } // 结束图片循环
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "✅ 所有图片身高测量完成" << std::endl;
    
    return 0;
}