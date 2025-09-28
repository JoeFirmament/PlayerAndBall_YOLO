/**
 * @file pose_analysis_with_images.cpp
 * @brief 基于真实图片的姿态分析测试 - 使用现有detector库 + 新增分析功能
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <thread>

// 现有检测器库
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

// 新增姿态分析系统
#include "pose_analyzer.h"

using namespace detector;
using namespace pose_analysis;

// 转换数据格式：detector::PoseResult -> pose_analysis::PoseResult
pose_analysis::PoseResult convert_to_analysis_format(const detector::PoseResult& det_result) {
    pose_analysis::PoseResult analysis_result;
    
    // 基本信息
    analysis_result.person_id = det_result.person_id;
    analysis_result.detection_confidence = det_result.confidence;
    analysis_result.timestamp = std::chrono::steady_clock::now();
    
    // 边界框
    analysis_result.bbox = cv::Rect2f(det_result.bbox.x, det_result.bbox.y, 
                                    det_result.bbox.width, det_result.bbox.height);
    
    // 关键点数据
    analysis_result.keypoints = det_result.keypoints;
    analysis_result.keypoint_confidences = det_result.keypoint_scores;
    
    return analysis_result;
}

// 创建对比图片：原图 + 检测结果 + 分析结果
cv::Mat create_comparison_image(const cv::Mat& original_image, 
                               const std::vector<detector::PoseResult>& det_results,
                               const std::vector<PoseAnalysisResult>& analysis_results) {
    
    cv::Mat detection_image = original_image.clone();
    cv::Mat analysis_image = original_image.clone();
    
    // 绘制检测结果
    for (const auto& det_result : det_results) {
        // ROI框
        cv::rectangle(detection_image, det_result.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 关键点
        for (size_t i = 0; i < det_result.keypoints.size(); ++i) {
            if (det_result.keypoint_scores[i] > 0.5f) {
                cv::circle(detection_image, det_result.keypoints[i], 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // 标签
        std::string label = cv::format("ID:%d (%.2f)", det_result.person_id, det_result.confidence);
        cv::putText(detection_image, label, 
                   cv::Point(det_result.bbox.x, det_result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    // 绘制分析结果
    for (size_t i = 0; i < analysis_results.size() && i < det_results.size(); ++i) {
        const auto& analysis_result = analysis_results[i];
        const auto& det_result = det_results[i];
        
        // ROI框（不同颜色）
        cv::Scalar analysis_color(255, 0, 255);  // 紫色
        cv::rectangle(analysis_image, det_result.bbox, analysis_color, 2);
        
        // 关键点
        for (size_t j = 0; j < det_result.keypoints.size(); ++j) {
            if (det_result.keypoint_scores[j] > 0.5f) {
                cv::circle(analysis_image, det_result.keypoints[j], 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        
        // 分析结果文本
        std::vector<std::string> result_texts;
        
        // 身高检测结果
        if (analysis_result.height_result.is_stable) {
            result_texts.push_back(cv::format("Height: %.0fmm (Stable)", 
                                 analysis_result.height_result.estimated_height_mm));
        } else if (analysis_result.height_result.estimated_height_mm > 0) {
            result_texts.push_back(cv::format("Height: %.0fmm (Measuring)", 
                                 analysis_result.height_result.estimated_height_mm));
        } else {
            result_texts.push_back("Height: Not detected");
        }
        
        // 要球动作结果
        if (analysis_result.ball_request_result.is_confirmed) {
            result_texts.push_back(cv::format("Ball Request: Confirmed (%.2f)", 
                                 analysis_result.ball_request_result.request_confidence));
        } else if (analysis_result.ball_request_result.is_requesting) {
            result_texts.push_back(cv::format("Ball Request: Detecting (%.2f)", 
                                 analysis_result.ball_request_result.request_confidence));
        } else {
            result_texts.push_back("Ball Request: Not detected");
        }
        
        // ID优先级
        result_texts.push_back(cv::format("Priority ID: %d", 
                             analysis_result.id_priority_result.priority_id));
        
        // 绘制分析结果文本
        cv::Point text_org(det_result.bbox.x, det_result.bbox.y - 10);
        for (size_t t = 0; t < result_texts.size(); ++t) {
            cv::Point current_pos(text_org.x, text_org.y - (result_texts.size() - t - 1) * 25);
            
            // 背景矩形
            cv::Size text_size = cv::getTextSize(result_texts[t], cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
            cv::rectangle(analysis_image, 
                        current_pos + cv::Point(-3, 3),
                        current_pos + cv::Point(text_size.width + 3, -text_size.height - 3),
                        cv::Scalar(0, 0, 0), -1);
            
            // 文本
            cv::putText(analysis_image, result_texts[t], current_pos,
                      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // 创建三联图：原图 + 检测结果 + 分析结果
    cv::Mat comparison_image;
    std::vector<cv::Mat> images = {original_image, detection_image, analysis_image};
    cv::hconcat(images, comparison_image);
    
    // 添加标题和标签
    cv::Mat titled_image(comparison_image.rows + 60, comparison_image.cols, CV_8UC3, cv::Scalar(40, 40, 40));
    comparison_image.copyTo(titled_image(cv::Rect(0, 60, comparison_image.cols, comparison_image.rows)));
    
    // 主标题
    std::string title = "Pose Analysis Comparison";
    cv::putText(titled_image, title, cv::Point(20, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    
    // 分标题
    int img_width = original_image.cols;
    cv::putText(titled_image, "Original", cv::Point(img_width/3, titled_image.rows-10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(titled_image, "Detection", cv::Point(img_width + img_width/3, titled_image.rows-10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(titled_image, "Analysis", cv::Point(2*img_width + img_width/3, titled_image.rows-10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return titled_image;
}

int main() {
    std::cout << "=== 基于真实图片的姿态分析测试 ===" << std::endl;
    
    try {
        // 1. 初始化姿态检测器
        std::string model_path = "../../models/Q_yolov8_pose.rknn";
        std::cout << "使用模型文件: " << model_path << std::endl;
        
        PoseDetectorLib detector(model_path);
        std::cout << "✓ 姿态检测器创建成功（延迟初始化）" << std::endl;
        
        // 设置检测参数
        detector.enable_tracking(true);  // 启用跟踪以获得person_id
        detector.set_confidence_threshold(0.3f);
        
        // 尝试加载Homography标定
        std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
        if (!calib_path.empty()) {
            bool calib_loaded = detector.load_calibration(calib_path);
            if (calib_loaded) {
                std::cout << "✓ Homography标定加载成功" << std::endl;
            }
        }
        
        // 2. 初始化姿态分析器
        auto config = PoseAnalyzer::create_default_config();
        auto analyzer = std::make_unique<PoseAnalyzer>(config);
        
        if (!analyzer->initialize()) {
            std::cerr << "❌ 姿态分析器初始化失败" << std::endl;
            return -1;
        }
        
        std::cout << "✓ 姿态分析器初始化成功" << std::endl;
        
        // 3. 测试多张图片
        std::vector<std::string> image_files = {
            "../../imgs/pose.jpg",
            "../../imgs/pose1.jpg", 
            "../../imgs/pose2.jpg",
            "../../imgs/pose3.jpg"
        };
        
        for (size_t i = 0; i < image_files.size(); ++i) {
            std::cout << "\n" << std::string(60, '=') << std::endl;
            std::cout << "测试图片 " << (i+1) << ": " << image_files[i] << std::endl;
            std::cout << std::string(60, '=') << std::endl;
            
            // 加载图片
            cv::Mat image = cv::imread(image_files[i]);
            if (image.empty()) {
                std::cout << "❌ 无法加载图片: " << image_files[i] << std::endl;
                continue;
            }
            
            std::cout << "✓ 图片加载成功 (" << image.cols << "x" << image.rows << ")" << std::endl;
            
            // 进行姿态检测
            auto det_results = detector.detect(image);
            std::cout << "✓ 检测到 " << det_results.size() << " 个人，推理时间: " 
                     << detector.get_last_inference_time_ms() << "ms" << std::endl;
            
            if (det_results.empty()) {
                std::cout << "  无姿态检测结果，跳过分析" << std::endl;
                continue;
            }
            
            // 转换数据格式并进行姿态分析
            std::vector<pose_analysis::PoseResult> analysis_input;
            for (const auto& det_result : det_results) {
                analysis_input.push_back(convert_to_analysis_format(det_result));
            }
            
            // 多帧处理模拟（重复处理相同图片以触发多帧验证机制）
            std::vector<PoseAnalysisResult> final_analysis_results;
            
            std::cout << "进行多帧分析验证..." << std::endl;
            for (int frame = 0; frame < 12; ++frame) {
                final_analysis_results = analyzer->analyze(analysis_input);
                
                // 检查是否有稳定结果
                bool has_stable_results = false;
                for (const auto& result : final_analysis_results) {
                    if (result.height_result.is_stable || result.ball_request_result.is_confirmed) {
                        has_stable_results = true;
                        break;
                    }
                }
                
                if (has_stable_results) {
                    std::cout << "  帧 " << frame << ": 检测到稳定结果" << std::endl;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 模拟30fps
            }
            
            // 显示最终分析结果
            for (size_t j = 0; j < final_analysis_results.size(); ++j) {
                const auto& result = final_analysis_results[j];
                const auto& det_result = det_results[j];
                
                std::cout << "\n--- Person " << (j+1) << " ---" << std::endl;
                std::cout << "检测置信度: " << std::fixed << std::setprecision(2) 
                         << det_result.confidence << std::endl;
                
                // 原始检测结果
                std::cout << "ROI: (" << det_result.bbox.x << "," << det_result.bbox.y 
                         << "," << det_result.bbox.width << "," << det_result.bbox.height << ")" << std::endl;
                
                if (det_result.has_ground_position) {
                    std::cout << "世界坐标: (" << std::fixed << std::setprecision(1)
                             << det_result.ground_position.x << "," << det_result.ground_position.y << ")mm" << std::endl;
                }
                
                // 分析结果
                if (result.height_result.is_stable) {
                    std::cout << "✓ 身高: " << std::fixed << std::setprecision(1)
                             << result.height_result.estimated_height_mm << "mm (稳定)" << std::endl;
                } else if (result.height_result.estimated_height_mm > 0) {
                    std::cout << "⏳ 身高: " << std::fixed << std::setprecision(1)
                             << result.height_result.estimated_height_mm << "mm (测量中)" << std::endl;
                } else {
                    std::cout << "⏸️ 身高: 未检测" << std::endl;
                }
                
                if (result.ball_request_result.is_confirmed) {
                    std::cout << "✓ 要球动作: 已确认 (置信度: " << std::fixed << std::setprecision(2)
                             << result.ball_request_result.request_confidence << ")" << std::endl;
                } else {
                    std::cout << "⏸️ 要球动作: 未确认" << std::endl;
                }
                
                std::cout << "ID优先级: " << result.id_priority_result.priority_id << std::endl;
            }
            
            // 创建并保存对比图片
            cv::Mat comparison_image = create_comparison_image(image, det_results, final_analysis_results);
            std::string output_filename = cv::format("pose_analysis_comparison_%d.jpg", (int)(i+1));
            cv::imwrite(output_filename, comparison_image);
            std::cout << "✓ 对比图片已保存: " << output_filename << std::endl;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "✓ 所有图片测试完成" << std::endl;
        std::cout << "成功验证了：YOLOv8 Pose检测 → 姿态分析 → 可视化输出流程" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}