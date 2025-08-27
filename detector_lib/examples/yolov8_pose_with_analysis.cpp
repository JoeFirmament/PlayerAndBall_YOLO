/**
 * @file yolov8_pose_with_analysis.cpp
 * @brief 集成姿态分析到YOLOv8 Pose检测系统
 * 
 * 展示如何在实际的YOLOv8 Pose + ByteTrack系统中集成姿态分析
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "rknn_api.h"

// YOLOv8 Pose相关
#include "postprocess.h"
#include "BYTETracker.h"

// 姿态分析系统
#include "pose_analyzer.h"
#include "debug_visualizer.h"

using namespace pose_analysis;

// 全局配置
struct AppConfig {
    std::string model_path = "../models/Q_yolov8_pose.rknn";
    std::string homography_file = "../data/2025_7_11pm.json";
    std::string analysis_config = "../data/pose_analysis_config.json";
    int camera_index = 0;
    bool enable_tracking = true;
    bool enable_analysis = true;
    bool enable_visualization = true;
    bool save_video = false;
    std::string output_video = "output_analyzed.mp4";
};

// 将YOLOv8检测结果转换为姿态分析输入
std::vector<PoseResult> convert_to_pose_results(
    const std::vector<STrack>& tracked_stracks,
    const std::vector<PoseDetectResult>& detect_results) {
    
    std::vector<PoseResult> pose_results;
    
    for (const auto& strack : tracked_stracks) {
        PoseResult pose;
        pose.person_id = strack.track_id;  // ByteTrack ID
        pose.detection_confidence = strack.score;
        pose.timestamp = std::chrono::steady_clock::now();
        
        // 转换bbox
        auto tlwh = strack.tlwh;
        pose.bbox = cv::Rect2f(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
        
        // 查找对应的关键点数据
        for (const auto& detect : detect_results) {
            // 匹配检测框（简单的IoU匹配）
            cv::Rect2f detect_box(detect.box.left, detect.box.top,
                                 detect.box.right - detect.box.left,
                                 detect.box.bottom - detect.box.top);
            
            float iou = (pose.bbox & detect_box).area() / 
                       (pose.bbox | detect_box).area();
            
            if (iou > 0.5) {  // IoU阈值
                // 复制关键点
                pose.keypoints.resize(17);
                pose.keypoint_confidences.resize(17);
                
                for (int i = 0; i < 17; ++i) {
                    pose.keypoints[i] = cv::Point2f(
                        detect.keypoints[i].x,
                        detect.keypoints[i].y
                    );
                    pose.keypoint_confidences[i] = detect.keypoints[i].confidence;
                }
                break;
            }
        }
        
        pose_results.push_back(pose);
    }
    
    return pose_results;
}

// 绘制分析结果到图像上
void draw_analysis_on_frame(cv::Mat& frame, 
                           const std::vector<PoseAnalysisResult>& results,
                           const AppConfig& config) {
    
    for (const auto& result : results) {
        // 使用优先级ID显示
        int display_id = result.id_priority_result.priority_id;
        cv::Rect bbox(result.height_result.person_id);  // 需要从其他地方获取bbox
        
        // 绘制ID和身高
        cv::Point text_pos(bbox.x, bbox.y - 10);
        std::string info_text = "ID:" + std::to_string(display_id);
        
        if (result.height_result.is_stable) {
            info_text += " H:" + std::to_string(int(result.height_result.estimated_height_mm)) + "mm";
        }
        
        cv::Scalar color = cv::Scalar(0, 255, 0);  // 默认绿色
        
        // 如果在要球，使用红色
        if (result.ball_request_result.is_confirmed) {
            color = cv::Scalar(0, 0, 255);
            info_text += " [REQUESTING]";
        }
        
        // 绘制文本背景
        cv::Size text_size = cv::getTextSize(info_text, cv::FONT_HERSHEY_SIMPLEX, 
                                            0.6, 1, nullptr);
        cv::rectangle(frame, 
                     cv::Rect(text_pos.x, text_pos.y - text_size.height - 2,
                             text_size.width, text_size.height + 4),
                     cv::Scalar(0, 0, 0), -1);
        
        // 绘制文本
        cv::putText(frame, info_text, text_pos, cv::FONT_HERSHEY_SIMPLEX,
                   0.6, color, 1, cv::LINE_AA);
        
        // 绘制检测框
        cv::rectangle(frame, bbox, color, 2);
        
        // 绘制状态指示器
        if (result.ball_request_result.state == BallRequestState::POTENTIAL_REQUEST) {
            cv::circle(frame, cv::Point(bbox.x + bbox.width - 10, bbox.y + 10),
                      5, cv::Scalar(255, 165, 0), -1);  // 橙色圆圈
        }
    }
    
    // 绘制统计信息
    cv::Point stats_pos(10, 30);
    cv::putText(frame, "Pose Analysis Active", stats_pos, cv::FONT_HERSHEY_SIMPLEX,
               0.5, cv::Scalar(0, 255, 255), 1);
}

int main(int argc, char** argv) {
    AppConfig config;
    
    // 解析命令行参数
    if (argc > 1) config.model_path = argv[1];
    if (argc > 2) config.homography_file = argv[2];
    if (argc > 3) config.camera_index = std::atoi(argv[3]);
    
    std::cout << "YOLOv8 Pose + 姿态分析集成示例\n";
    std::cout << "================================\n";
    std::cout << "模型: " << config.model_path << "\n";
    std::cout << "标定: " << config.homography_file << "\n";
    std::cout << "摄像头: " << config.camera_index << "\n\n";
    
    // ========== 1. 初始化RKNN模型 ==========
    rknn_context ctx;
    // ... RKNN初始化代码（省略）...
    
    // ========== 2. 初始化ByteTracker ==========
    BYTETracker tracker;
    tracker.init(30, 30);  // fps=30, buffer=30
    
    // ========== 3. 初始化姿态分析器 ==========
    auto analyzer = create_pose_analyzer(config.analysis_config);
    
    // 加载Homography矩阵
    cv::Mat homography;
    cv::FileStorage fs(config.homography_file, cv::FileStorage::READ);
    if (fs.isOpened()) {
        fs["homography_matrix"] >> homography;
        analyzer->set_homography(homography);
        std::cout << "已加载Homography矩阵\n";
    }
    
    // ========== 4. 初始化调试工具 ==========
    DebugVisualizer visualizer;
    DataRecorder recorder("./records/");
    PerformanceMonitor perf_monitor;
    
    if (config.enable_analysis) {
        recorder.start_recording();
        perf_monitor.enable_monitoring();
    }
    
    // ========== 5. 打开摄像头 ==========
    cv::VideoCapture cap(config.camera_index);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头 " << config.camera_index << "\n";
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    
    // ========== 6. 创建视频写入器（可选）==========
    cv::VideoWriter video_writer;
    if (config.save_video) {
        video_writer.open(config.output_video,
                         cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                         30, cv::Size(1920, 1080));
    }
    
    // ========== 7. 主处理循环 ==========
    cv::Mat frame;
    int frame_count = 0;
    bool tracking_enabled = config.enable_tracking;
    bool analysis_enabled = config.enable_analysis;
    
    std::cout << "\n开始处理...\n";
    std::cout << "按键控制:\n";
    std::cout << "  T - 切换跟踪\n";
    std::cout << "  A - 切换分析\n";
    std::cout << "  V - 切换可视化\n";
    std::cout << "  S - 保存截图\n";
    std::cout << "  ESC - 退出\n\n";
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // ========== YOLOv8 Pose推理 ==========
        // ... RKNN推理代码（省略）...
        std::vector<PoseDetectResult> detect_results;
        // detect_results = run_yolov8_pose_inference(ctx, frame);
        
        // ========== ByteTrack跟踪 ==========
        std::vector<STrack> output_stracks;
        if (tracking_enabled) {
            // output_stracks = tracker.update(detect_results);
        }
        
        // ========== 姿态分析 ==========
        std::vector<PoseAnalysisResult> analysis_results;
        if (analysis_enabled && !output_stracks.empty()) {
            // 转换数据格式
            auto pose_results = convert_to_pose_results(output_stracks, detect_results);
            
            // 执行分析
            analysis_results = analyzer->analyze(pose_results);
            
            // 记录结果
            recorder.record_frame(analysis_results, frame);
        }
        
        // ========== 可视化 ==========
        if (config.enable_visualization) {
            // 绘制骨架和跟踪框
            // draw_skeletons(frame, detect_results);
            // draw_tracking_boxes(frame, output_stracks);
            
            // 绘制分析结果
            if (analysis_enabled) {
                draw_analysis_on_frame(frame, analysis_results, config);
                
                // 绘制调试信息
                cv::Rect panel_area(frame.cols - 350, 10, 340, 200);
                visualizer.draw_statistics_panel(frame, panel_area);
            }
            
            // 显示FPS
            auto end_time = std::chrono::high_resolution_clock::now();
            float fps = 1000.0f / std::chrono::duration<float, std::milli>(
                end_time - start_time).count();
            
            cv::putText(frame, "FPS: " + std::to_string(int(fps)),
                       cv::Point(10, frame.rows - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5,
                       cv::Scalar(0, 255, 0), 1);
        }
        
        // ========== 显示和保存 ==========
        cv::imshow("YOLOv8 Pose Analysis", frame);
        
        if (video_writer.isOpened()) {
            video_writer.write(frame);
        }
        
        // ========== 按键处理 ==========
        int key = cv::waitKey(1);
        if (key == 27) break;  // ESC
        
        switch (key) {
            case 't':
            case 'T':
                tracking_enabled = !tracking_enabled;
                std::cout << "跟踪: " << (tracking_enabled ? "开启" : "关闭") << "\n";
                break;
                
            case 'a':
            case 'A':
                analysis_enabled = !analysis_enabled;
                std::cout << "分析: " << (analysis_enabled ? "开启" : "关闭") << "\n";
                if (analysis_enabled) {
                    analyzer->reset_all();
                }
                break;
                
            case 'v':
            case 'V':
                config.enable_visualization = !config.enable_visualization;
                std::cout << "可视化: " << (config.enable_visualization ? "开启" : "关闭") << "\n";
                break;
                
            case 's':
            case 'S':
                {
                    std::string filename = "screenshot_" + std::to_string(frame_count) + ".jpg";
                    cv::imwrite(filename, frame);
                    std::cout << "保存截图: " << filename << "\n";
                }
                break;
                
            case 'd':
            case 'D':
                // 输出调试信息
                std::cout << analyzer->get_debug_info() << "\n";
                break;
        }
        
        // 更新性能统计
        perf_monitor.record_frame_end(output_stracks.size());
        frame_count++;
    }
    
    // ========== 8. 清理和报告 ==========
    std::cout << "\n处理完成！\n";
    std::cout << "总帧数: " << frame_count << "\n";
    
    // 保存记录
    if (analysis_enabled) {
        recorder.stop_recording();
        recorder.save_session("yolov8_pose_analysis");
        recorder.generate_analysis_report("analysis_report.txt");
        
        // 输出性能报告
        std::cout << "\n" << perf_monitor.get_performance_summary() << "\n";
        std::cout << analyzer->get_performance_stats() << "\n";
        
        // 输出分析统计
        auto avg_heights = recorder.calculate_average_heights();
        std::cout << "\n平均身高:\n";
        for (const auto& [id, height] : avg_heights) {
            std::cout << "  Person " << id << ": " << height << " mm\n";
        }
    }
    
    // 释放资源
    cap.release();
    cv::destroyAllWindows();
    if (video_writer.isOpened()) {
        video_writer.release();
    }
    
    // rknn_destroy(ctx);
    
    return 0;
}

/**
 * 编译命令（需要链接RKNN和其他依赖）：
 * 
 * g++ -std=c++17 yolov8_pose_with_analysis.cpp \
 *     ../src/pose_analyzer.cpp \
 *     ../src/height_detector.cpp \
 *     ../src/ball_request_detector.cpp \
 *     ../src/id_priority_manager.cpp \
 *     ../../src/postprocess.cc \
 *     ../../src/BYTETracker.cpp \
 *     ../../src/STrack.cpp \
 *     ../../src/kalmanFilter.cpp \
 *     -I../include -I../../include \
 *     -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui \
 *     -lrknn_api -ljsoncpp -pthread \
 *     -o yolov8_pose_with_analysis
 * 
 * 运行命令：
 * ./yolov8_pose_with_analysis [模型路径] [标定文件] [摄像头索引]
 */