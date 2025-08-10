/*-------------------------------------------
          纯篮筐篮球图片检测程序
          
只测试篮筐篮球检测功能，输入图片
运行: ./rim_basketball_image [图片路径]
-------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "RimBasketballDetectorLib.h"

int main(int argc, char* argv[]) {
    std::cout << "=== 篮筐篮球图片检测程序 ===" << std::endl;
    
    // 统一使用分发包路径约定
    std::string image_path = "../imgs/rim.jpg";  // 默认图片
    std::string model_path = "../models/Q_Rim_Basketball_724_JZ.rknn";  // 默认模型
    
    if (argc >= 2) {
        image_path = argv[1];
    }
    if (argc >= 3) {
        model_path = argv[2];
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
        std::cout << "正在创建篮筐篮球检测器..." << std::endl;
        detector::RimBasketballDetectorLib detector(model_path);
        
        // 配置参数 - 使用工作版本的阈值
        detector.set_confidence_threshold(0.25f);
        detector.set_nms_threshold(0.1f);
        
        // 检测
        std::cout << "正在检测..." << std::endl;
        auto results = detector.detect(image);
        
        std::cout << "✅ 检测完成!" << std::endl;
        std::cout << "推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        std::cout << "检测器状态: " << (int)detector.get_status() << std::endl;
        std::cout << "检测到目标数量: " << results.size() << std::endl;
        
        // 分析结果并绘制
        cv::Mat result_image = image.clone();
        if (results.empty()) {
            std::cout << "未检测到篮筐或篮球" << std::endl;
        } else {
            int rim_count = 0, basketball_count = 0;
            int close_basketballs = 0;
            
            for (const auto& result : results) {
                if (result.class_id == 1) {
                    rim_count++;
                } else if (result.class_id == 0) {
                    basketball_count++;
                    if (result.is_in_rim_roi) {
                        close_basketballs++;
                    }
                }
            }
            
            std::cout << "\n📊 检测统计:" << std::endl;
            std::cout << "  🏀 篮球: " << basketball_count << "个" << std::endl;
            std::cout << "  🎯 篮筐: " << rim_count << "个" << std::endl;
            std::cout << "  ⭐ 靠近篮筐的篮球: " << close_basketballs << "个" << std::endl;
            
            std::cout << "\n📋 详细结果:" << std::endl;
            for (size_t i = 0; i < results.size(); i++) {
                const auto& obj = results[i];
                std::cout << "  目标" << (i+1) << ": " << obj.class_name 
                         << ", 置信度=" << std::fixed << std::setprecision(2) << obj.confidence
                         << ", 位置=(" << std::fixed << std::setprecision(0) 
                         << obj.center.x << "," << obj.center.y << ")";
                
                if (obj.class_id == 0) {  // basketball
                    std::cout << ", 距篮筐=" << std::fixed << std::setprecision(1) << obj.distance_to_rim << "px";
                    if (obj.is_in_rim_roi) {
                        std::cout << " ⭐";
                    }
                }
                std::cout << std::endl;
                
                // 绘制检测框
                cv::Scalar color;
                if (obj.class_id == 0) {  // basketball
                    color = cv::Scalar(0, 165, 255);  // 橙色
                } else {  // rim
                    color = cv::Scalar(0, 255, 0);    // 绿色
                }
                
                cv::rectangle(result_image, obj.bbox, color, 3);
                
                // 绘制置信度和类别标签
                std::string label = obj.class_name + " " + std::to_string(obj.confidence).substr(0, 4);
                int baseline;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                cv::Point label_origin(obj.bbox.x, obj.bbox.y - 10);
                
                cv::rectangle(result_image, 
                             cv::Point(label_origin.x, label_origin.y - label_size.height - baseline),
                             cv::Point(label_origin.x + label_size.width, label_origin.y + baseline), 
                             color, -1);
                cv::putText(result_image, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);
                
                // 绘制中心点
                cv::circle(result_image, obj.center, 4, color, -1);
            }
            
            // 绘制距离连线
            cv::Point2f rim_center(-1, -1);
            bool has_rim = false;
            
            for (const auto& result : results) {
                if (result.class_id == 1) {  // rim
                    rim_center = result.center;
                    has_rim = true;
                    break;
                }
            }
            
            if (has_rim) {
                for (const auto& result : results) {
                    if (result.class_id == 0 && result.distance_to_rim > 0) {  // basketball
                        // 绘制到篮筐的连线
                        cv::line(result_image, result.center, rim_center, cv::Scalar(255, 255, 0), 2);
                        
                        // 显示距离
                        cv::Point mid_point((result.center.x + rim_center.x) / 2, 
                                          (result.center.y + rim_center.y) / 2 - 10);
                        std::string dist_text = std::to_string((int)result.distance_to_rim) + "px";
                        cv::putText(result_image, dist_text, mid_point, cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, cv::Scalar(255, 255, 0), 2);
                    }
                }
            }
        }
        
        // 保存结果图片
        std::string output_filename = "rim_basketball_detection_result.jpg";
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