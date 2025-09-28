/**
 * @file pose_image_analysis_test.cpp
 * @brief 使用真实图片进行姿态分析测试
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

// 姿态分析系统
#include "pose_analyzer.h"

using namespace pose_analysis;

// 转换PoseResult格式（从detector到pose_analysis）
std::vector<pose_analysis::PoseResult> convert_pose_results(
    const std::vector<detector::PoseResult>& detector_results) {
    
    std::vector<pose_analysis::PoseResult> analysis_results;
    
    for (const auto& det_result : detector_results) {
        pose_analysis::PoseResult ana_result;
        
        // 基本信息转换
        ana_result.person_id = det_result.person_id;
        ana_result.detection_confidence = det_result.confidence;
        ana_result.timestamp = std::chrono::steady_clock::now();
        
        // bbox转换
        ana_result.bbox = cv::Rect2f(
            det_result.bbox.x, det_result.bbox.y,
            det_result.bbox.width, det_result.bbox.height
        );
        
        // 关键点转换
        ana_result.keypoints = det_result.keypoints;
        ana_result.keypoint_confidences = det_result.keypoint_scores;
        
        analysis_results.push_back(ana_result);
    }
    
    return analysis_results;
}

int main(int argc, char* argv[]) {
    std::cout << "=== 真实图片姿态分析测试 ===" << std::endl;
    
    try {
        // 1. 初始化姿态检测器
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        if (model_path.empty()) {
            std::cerr << "无法找到姿态检测模型" << std::endl;
            return -1;
        }
        
        PoseDetectorLib pose_detector(model_path);
        if (!pose_detector.is_initialized()) {
            std::cerr << "姿态检测器初始化失败" << std::endl;
            return -1;
        }
        
        std::cout << "✓ 姿态检测器初始化成功" << std::endl;
        
        // 2. 初始化姿态分析器
        auto config = pose_analysis::PoseAnalyzer::create_default_config();
        auto analyzer = std::make_unique<pose_analysis::PoseAnalyzer>(config);
        
        if (!analyzer->initialize()) {
            std::cerr << "姿态分析器初始化失败" << std::endl;
            return -1;
        }
        
        std::cout << "✓ 姿态分析器初始化成功" << std::endl;
        
        // 3. 测试多张图片
        std::vector<std::string> image_files = {
            "../../../imgs/pose.jpg",
            "../../../imgs/pose1.jpg", 
            "../../../imgs/pose2.jpg",
            "../../../imgs/pose3.jpg"
        };
        
        for (size_t i = 0; i < image_files.size(); ++i) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "测试图片 " << (i+1) << ": " << image_files[i] << std::endl;
            std::cout << std::string(50, '=') << std::endl;
            
            // 加载图片
            cv::Mat image = cv::imread(image_files[i]);
            if (image.empty()) {
                std::cout << "❌ 无法加载图片: " << image_files[i] << std::endl;
                continue;
            }
            
            std::cout << "✓ 图片加载成功 (" << image.cols << "x" << image.rows << ")" << std::endl;
            
            // 进行姿态检测
            auto detector_results = pose_detector.detect(image);
            std::cout << "✓ 检测到 " << detector_results.size() << " 个人" << std::endl;
            
            if (detector_results.empty()) {
                std::cout << "  无姿态检测结果，跳过分析" << std::endl;
                continue;
            }
            
            // 转换数据格式
            auto analysis_input = convert_pose_results(detector_results);
            
            // 进行姿态分析
            auto analysis_results = analyzer->analyze(analysis_input);
            
            // 创建对比图片：原始图 + 分析结果图
            cv::Mat result_image = image.clone();
            cv::Mat comparison_image;
            
            // 在结果图上绘制分析结果
            for (size_t j = 0; j < analysis_results.size(); ++j) {
                const auto& result = analysis_results[j];
                const auto& det_result = detector_results[j];
                
                std::cout << "\n--- Person " << (j+1) << " ---" << std::endl;
                std::cout << "检测置信度: " << std::fixed << std::setprecision(2) 
                         << det_result.confidence << std::endl;
                std::cout << "ROI框: (" << det_result.bbox.x << ", " << det_result.bbox.y 
                         << ", " << det_result.bbox.width << ", " << det_result.bbox.height << ")" << std::endl;
                
                // 绘制ROI框
                cv::Scalar roi_color(0, 255, 0);  // 绿色
                cv::rectangle(result_image, det_result.bbox, roi_color, 2);
                
                // 绘制关键点和骨架
                for (size_t k = 0; k < det_result.keypoints.size(); ++k) {
                    if (det_result.keypoint_scores[k] > 0.5f) {
                        cv::circle(result_image, det_result.keypoints[k], 3, cv::Scalar(0, 0, 255), -1);
                    }
                }
                
                // 准备分析结果文本
                std::vector<std::string> result_texts;
                
                // 身高检测结果
                if (result.height_result.is_stable) {
                    std::cout << "✓ 身高: " << std::fixed << std::setprecision(1)
                             << result.height_result.estimated_height_mm << "mm (稳定)" << std::endl;
                    result_texts.push_back(cv::format("Height: %.0fmm (Stable)", 
                                          result.height_result.estimated_height_mm));
                } else if (result.height_result.is_measuring) {
                    std::cout << "⏳ 身高测量中: " << std::fixed << std::setprecision(1)
                             << result.height_result.current_measurement_mm << "mm" << std::endl;
                    result_texts.push_back(cv::format("Height: %.0fmm (Measuring)", 
                                          result.height_result.current_measurement_mm));
                } else {
                    std::cout << "⏸️  身高: 未检测 (可能手部位置不符合条件)" << std::endl;
                    result_texts.push_back("Height: Not detected");
                }
                
                // 要球动作结果
                if (result.ball_request_result.is_confirmed) {
                    std::cout << "✓ 要球动作: 已确认 (置信度: " << std::fixed << std::setprecision(2)
                             << result.ball_request_result.confidence << ")" << std::endl;
                    result_texts.push_back(cv::format("Ball Request: Confirmed (%.2f)", 
                                          result.ball_request_result.confidence));
                } else if (result.ball_request_result.is_detecting) {
                    std::cout << "⏳ 要球检测中: 置信度 " << std::fixed << std::setprecision(2)
                             << result.ball_request_result.confidence << std::endl;
                    result_texts.push_back(cv::format("Ball Request: Detecting (%.2f)", 
                                          result.ball_request_result.confidence));
                } else {
                    std::cout << "⏸️  要球动作: 未检测" << std::endl;
                    result_texts.push_back("Ball Request: Not detected");
                }
                
                // ID优先级
                std::cout << "ID优先级: " << result.id_priority_result.priority_id << std::endl;
                result_texts.push_back(cv::format("Priority ID: %d", result.id_priority_result.priority_id));
                
                // 在图像上绘制文本
                cv::Point text_org(det_result.bbox.x, det_result.bbox.y - 10);
                for (size_t t = 0; t < result_texts.size(); ++t) {
                    cv::Point current_pos(text_org.x, text_org.y - (result_texts.size() - t - 1) * 25);
                    
                    // 背景矩形
                    cv::Size text_size = cv::getTextSize(result_texts[t], cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
                    cv::rectangle(result_image, 
                                current_pos + cv::Point(-5, 5),
                                current_pos + cv::Point(text_size.width + 5, -text_size.height - 5),
                                cv::Scalar(0, 0, 0), -1);
                    
                    // 文本
                    cv::putText(result_image, result_texts[t], current_pos,
                              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                }
            }
            
            // 创建对比图：原始图 | 结果图
            cv::hconcat(image, result_image, comparison_image);
            
            // 添加标题
            cv::Mat titled_image(comparison_image.rows + 40, comparison_image.cols, CV_8UC3, cv::Scalar(50, 50, 50));
            comparison_image.copyTo(titled_image(cv::Rect(0, 40, comparison_image.cols, comparison_image.rows)));
            
            // 绘制标题
            std::string title = cv::format("Pose Analysis Test - Image %d", (int)(i+1));
            cv::putText(titled_image, title, cv::Point(20, 25), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            cv::putText(titled_image, "Original", cv::Point(image.cols/4, titled_image.rows-10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            cv::putText(titled_image, "Analysis Result", cv::Point(image.cols + image.cols/4, titled_image.rows-10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            
            // 保存对比图片
            std::string output_filename = cv::format("pose_analysis_result_%d.jpg", (int)(i+1));
            cv::imwrite(output_filename, titled_image);
            std::cout << "✓ 对比图片已保存: " << output_filename << std::endl;
            
            // 模拟多帧处理（相同图片）
            if (!analysis_results.empty()) {
                std::cout << "\n--- 多帧验证测试 (相同图片重复处理) ---" << std::endl;
                for (int frame = 1; frame <= 12; ++frame) {
                    auto multi_results = analyzer->analyze(analysis_input);
                    
                    bool has_changes = false;
                    for (const auto& result : multi_results) {
                        if (result.height_result.is_stable || result.ball_request_result.is_confirmed) {
                            has_changes = true;
                            break;
                        }
                    }
                    
                    if (has_changes) {
                        std::cout << "帧 " << std::setw(2) << frame << ": ";
                        for (const auto& result : multi_results) {
                            if (result.height_result.is_stable) {
                                std::cout << "身高稳定(" << std::fixed << std::setprecision(0)
                                         << result.height_result.estimated_height_mm << "mm) ";
                            }
                            if (result.ball_request_result.is_confirmed) {
                                std::cout << "要球确认 ";
                            }
                        }
                        std::cout << std::endl;
                    }
                    
                    // 模拟帧间隔
                    std::this_thread::sleep_for(std::chrono::milliseconds(33));
                }
            }
        }
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "✓ 所有图片测试完成" << std::endl;
        std::cout << "验证了真实图片的姿态检测→分析流程" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}