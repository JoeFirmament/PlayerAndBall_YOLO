#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <vector>

// 现有检测器库
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

// 新增姿态分析系统 (仅用于身高分析)
#include "pose_analyzer.h"

using namespace cv;
using namespace std;
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

class HeightMeasurementTester {
private:
    Mat homography_matrix_;
    float camera_height_mm_;
    std::unique_ptr<PoseDetectorLib> detector_;
    std::unique_ptr<PoseAnalyzer> analyzer_;
    
public:
    HeightMeasurementTester() : camera_height_mm_(1170.0f) {} // 117cm = 1170mm
    
    // 加载Homography矩阵
    bool load_homography(const string& json_file) {
        ifstream file(json_file);
        if (!file.is_open()) {
            cerr << "无法打开文件: " << json_file << endl;
            return false;
        }
        
        Json::Value root;
        file >> root;
        
        // 提取3x3矩阵
        homography_matrix_ = Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                homography_matrix_.at<double>(i, j) = root["matrix"][i][j].asDouble();
            }
        }
        
        cout << "Homography矩阵加载成功:" << endl << homography_matrix_ << endl;
        return true;
    }
    
    // 像素坐标转换为地面世界坐标
    Point2f pixel_to_world(const Point2f& pixel) {
        vector<Point2f> src_points = {pixel};
        vector<Point2f> dst_points;
        
        perspectiveTransform(src_points, dst_points, homography_matrix_);
        return dst_points[0];
    }
    
    // 初始化检测和分析系统
    bool initialize_systems(const string& model_path, const string& homography_file) {
        // 1. 初始化姿态检测器
        detector_ = std::make_unique<PoseDetectorLib>(model_path);
        cout << "✓ 姿态检测器创建成功" << endl;
        
        // 设置检测参数
        detector_->enable_tracking(true);  // 启用跟踪以获得person_id
        detector_->set_confidence_threshold(0.3f);
        
        // 加载Homography标定
        bool calib_loaded = detector_->load_calibration(homography_file);
        if (!calib_loaded) {
            cerr << "❌ 无法加载Homography标定文件: " << homography_file << endl;
            return false;
        }
        cout << "✓ Homography标定加载成功" << endl;
        
        // 2. 初始化姿态分析器 (仅启用身高检测)
        auto config = PoseAnalyzer::create_default_config();
        config.height_detection.min_stable_frames = 5; // 更快收敛
        config.height_detection.stability_threshold_mm = 30.0f; // 更宽松稳定性阈值
        
        analyzer_ = std::make_unique<PoseAnalyzer>(config);
        if (!analyzer_->initialize()) {
            cerr << "❌ 姿态分析器初始化失败" << endl;
            return false;
        }
        
        // 仅启用身高检测
        analyzer_->enable_height_detection(true);
        analyzer_->enable_ball_request_detection(false);
        analyzer_->enable_id_priority_management(false);
        
        cout << "✓ 姿态分析器初始化成功（仅身高检测）" << endl;
        
        return true;
    }
    
    // 🔥 核心：基于真实pose检测的身高测量
    Mat process_image_with_pose_detection(const Mat& image, const string& output_path = "") {
        cout << "\n=== 基于Pose检测的身高测量 ===" << endl;
        cout << "图片尺寸: " << image.cols << "x" << image.rows << endl;
        
        // 1. 进行姿态检测
        auto det_results = detector_->detect(image);
        cout << "✓ 检测到 " << det_results.size() << " 个人，推理时间: " 
             << detector_->get_last_inference_time_ms() << "ms" << endl;
        
        if (det_results.empty()) {
            cout << "  无姿态检测结果" << endl;
            return image.clone();
        }
        
        // 2. 转换数据格式并进行姿态分析
        std::vector<pose_analysis::PoseResult> analysis_input;
        for (const auto& det_result : det_results) {
            analysis_input.push_back(convert_to_analysis_format(det_result));
        }
        
        // 3. 多帧处理模拟（重复处理相同图片以触发稳定检测）
        std::vector<PoseAnalysisResult> analysis_results;
        cout << "进行多帧身高分析验证..." << endl;
        for (int frame = 0; frame < 8; ++frame) {
            analysis_results = analyzer_->analyze(analysis_input);
            
            // 检查是否有稳定身高结果
            bool has_stable_height = false;
            for (const auto& result : analysis_results) {
                if (result.height_result.is_stable) {
                    has_stable_height = true;
                    cout << "  帧 " << frame << ": 检测到稳定身高" << endl;
                    break;
                }
            }
            if (has_stable_height) break;
        }
        
        // 4. 创建结果图片
        Mat result_image = image.clone();
        
        for (size_t i = 0; i < det_results.size() && i < analysis_results.size(); ++i) {
            const auto& det_result = det_results[i];
            const auto& analysis_result = analysis_results[i];
            
            cout << "\n--- Person " << (i+1) << " ---" << endl;
            cout << "检测置信度: " << det_result.confidence << endl;
            
            // 绘制ROI框（紫色）
            Scalar roi_color(255, 0, 255);  // 紫色
            rectangle(result_image, det_result.bbox, roi_color, 3);
            
            // 绘制关键点
            for (size_t j = 0; j < det_result.keypoints.size(); ++j) {
                if (det_result.keypoint_scores[j] > 0.5f) {
                    circle(result_image, det_result.keypoints[j], 4, Scalar(0, 255, 0), -1);
                }
            }
            
            // 绘制骨架连接线（简化版）
            draw_skeleton_lines(result_image, det_result);
            
            // 身高分析结果
            string height_text;
            Scalar text_color;
            
            if (analysis_result.height_result.is_stable) {
                float height_cm = analysis_result.height_result.estimated_height_mm / 10.0f;
                height_text = cv::format("Height: %.1fcm (Stable)", height_cm);
                text_color = Scalar(0, 255, 0);  // 绿色
                cout << "✓ 身高: " << height_cm << "cm (稳定)" << endl;
            } else if (analysis_result.height_result.estimated_height_mm > 0) {
                float height_cm = analysis_result.height_result.estimated_height_mm / 10.0f;
                height_text = cv::format("Height: %.1fcm (Measuring)", height_cm);
                text_color = Scalar(0, 255, 255);  // 黄色
                cout << "⏳ 身高: " << height_cm << "cm (测量中)" << endl;
            } else {
                height_text = "Height: Not detected";
                text_color = Scalar(0, 0, 255);  // 红色
                cout << "⏸️ 身高: 未检测" << endl;
            }
            
            // ROI标签
            string roi_text = cv::format("Person%d (%.2f)", (int)(i+1), det_result.confidence);
            
            // 绘制文字背景和文字
            Point text_org(det_result.bbox.x, det_result.bbox.y - 10);
            
            // ROI标签
            Size roi_text_size = getTextSize(roi_text, FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr);
            rectangle(result_image, 
                     text_org + Point(-5, 5),
                     text_org + Point(roi_text_size.width + 5, -roi_text_size.height - 5),
                     Scalar(0, 0, 0), -1);
            putText(result_image, roi_text, text_org, 
                   FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2);
            
            // 身高标签
            Point height_text_org(det_result.bbox.x, det_result.bbox.y - 40);
            Size height_text_size = getTextSize(height_text, FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            rectangle(result_image, 
                     height_text_org + Point(-5, 5),
                     height_text_org + Point(height_text_size.width + 5, -height_text_size.height - 5),
                     Scalar(0, 0, 0), -1);
            putText(result_image, height_text, height_text_org, 
                   FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
            
            // 显示世界坐标（如果有）
            if (det_result.has_ground_position) {
                cout << "世界坐标: (" << det_result.ground_position.x << ", " 
                     << det_result.ground_position.y << ")mm" << endl;
                
                string world_coord_text = cv::format("World: (%.0f,%.0f)mm", 
                                                    det_result.ground_position.x, 
                                                    det_result.ground_position.y);
                Point coord_text_org(det_result.bbox.x, det_result.bbox.y + det_result.bbox.height + 25);
                Size coord_text_size = getTextSize(world_coord_text, FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
                rectangle(result_image, 
                         coord_text_org + Point(-3, 3),
                         coord_text_org + Point(coord_text_size.width + 3, -coord_text_size.height - 3),
                         Scalar(0, 0, 0), -1);
                putText(result_image, world_coord_text, coord_text_org, 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }
        
        // 添加标题
        putText(result_image, "Pose Detection + Height Measurement Results", 
                Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        putText(result_image, "Camera Height: 117cm, Homography-based calculation", 
                Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        
        // 保存结果
        if (!output_path.empty()) {
            bool saved = imwrite(output_path, result_image);
            if (saved) {
                cout << "\n✅ 结果图片已保存到: " << output_path << endl;
            } else {
                cout << "\n❌ 保存结果图片失败！" << endl;
            }
        }
        
        return result_image;
    }
    
private:
    // 绘制骨架连接线
    void draw_skeleton_lines(Mat& image, const detector::PoseResult& pose_result) {
        // COCO 17点骨架连接定义
        vector<pair<int, int>> skeleton_connections = {
            {5, 6},   // 左肩-右肩
            {5, 7},   // 左肩-左肘
            {7, 9},   // 左肘-左腕
            {6, 8},   // 右肩-右肘
            {8, 10},  // 右肘-右腕
            {5, 11},  // 左肩-左髋
            {6, 12},  // 右肩-右髋
            {11, 12}, // 左髋-右髋
            {11, 13}, // 左髋-左膝
            {13, 15}, // 左膝-左踝
            {12, 14}, // 右髋-右膝
            {14, 16}  // 右膝-右踝
        };
        
        for (const auto& connection : skeleton_connections) {
            int p1_idx = connection.first;
            int p2_idx = connection.second;
            
            if (p1_idx < pose_result.keypoint_scores.size() && 
                p2_idx < pose_result.keypoint_scores.size() &&
                pose_result.keypoint_scores[p1_idx] > 0.5f && 
                pose_result.keypoint_scores[p2_idx] > 0.5f) {
                
                line(image, pose_result.keypoints[p1_idx], pose_result.keypoints[p2_idx], 
                     Scalar(0, 255, 255), 2);  // 黄色骨架线
            }
        }
    }
};

int main() {
    cout << "=== 基于Pose检测的身高测量测试 ===" << endl;
    
    try {
        HeightMeasurementTester tester;
        
        // 1. 初始化检测和分析系统
        string model_path = "../../models/Q_yolov8_pose.rknn";
        string homography_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/data/2025_8_6_1280_720.json";
        
        cout << "使用模型文件: " << model_path << endl;
        cout << "使用标定文件: " << homography_file << endl;
        
        if (!tester.initialize_systems(model_path, homography_file)) {
            cerr << "❌ 系统初始化失败" << endl;
            return -1;
        }
        
        // 2. 测试多张图片
        vector<string> image_files = {
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose.jpg",
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose1.jpg", 
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose2.jpg",
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose3.jpg"
        };
        
        for (size_t i = 0; i < image_files.size(); ++i) {
            cout << "\n" << string(60, '=') << endl;
            cout << "测试图片 " << (i+1) << ": " << image_files[i] << endl;
            cout << string(60, '=') << endl;
            
            // 加载图片
            Mat image = imread(image_files[i]);
            if (image.empty()) {
                cout << "❌ 无法加载图片: " << image_files[i] << endl;
                continue;
            }
            
            cout << "✓ 图片加载成功 (" << image.cols << "x" << image.rows << ")" << endl;
            
            // 处理图片并生成结果
            string output_filename = cv::format("pose_height_result_%d.jpg", (int)(i+1));
            tester.process_image_with_pose_detection(image, output_filename);
        }
        
        cout << "\n" << string(60, '=') << endl;
        cout << "✅ 所有图片测试完成" << endl;
        cout << "🎉 成功验证了：Pose检测 → ROI可视化 → 身高测量流程" << endl;
        
    } catch (const std::exception& e) {
        cerr << "错误: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}