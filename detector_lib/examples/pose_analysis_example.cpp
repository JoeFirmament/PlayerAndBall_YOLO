/**
 * @file pose_analysis_example.cpp
 * @brief 姿态分析系统使用示例
 * 
 * 展示如何集成姿态分析系统到现有的YOLOv8 Pose检测流程中
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>

// 姿态分析系统头文件
#include "pose_analyzer.h"
#include "debug_visualizer.h"

using namespace pose_analysis;

// 模拟的ByteTrack输出（实际使用时由ByteTracker提供）
std::vector<PoseResult> simulate_bytetrack_output(int frame_id) {
    std::vector<PoseResult> results;
    
    // 模拟2个人的检测结果
    for (int person_id = 1; person_id <= 2; ++person_id) {
        PoseResult pose;
        pose.person_id = person_id;  // ByteTrack分配的ID
        pose.detection_confidence = 0.8f + (rand() % 20) / 100.0f;
        pose.timestamp = std::chrono::steady_clock::now();
        
        // 模拟bbox (人体检测框)
        pose.bbox = cv::Rect2f(
            100.0f * person_id,      // x
            50.0f,                   // y  
            200.0f,                  // width
            450.0f + (rand() % 50)   // height (带一些变化)
        );
        
        // 模拟17个COCO关键点
        pose.keypoints.resize(17);
        pose.keypoint_confidences.resize(17);
        
        for (int i = 0; i < 17; ++i) {
            pose.keypoints[i] = cv::Point2f(
                pose.bbox.x + pose.bbox.width * 0.5f + (rand() % 20 - 10),
                pose.bbox.y + (i * pose.bbox.height / 17.0f)
            );
            pose.keypoint_confidences[i] = 0.7f + (rand() % 30) / 100.0f;
        }
        
        // 模拟要球动作（person 2在某些帧做要球动作）
        if (person_id == 2 && frame_id >= 30 && frame_id <= 60) {
            // 调整手腕位置模拟要球
            pose.keypoints[9] = cv::Point2f(pose.bbox.x + pose.bbox.width * 0.4f, 
                                           pose.bbox.y + pose.bbox.height * 0.5f);
            pose.keypoints[10] = cv::Point2f(pose.bbox.x + pose.bbox.width * 0.6f,
                                            pose.bbox.y + pose.bbox.height * 0.5f);
        }
        
        results.push_back(pose);
    }
    
    return results;
}

// 格式化输出分析结果
void print_analysis_results(const std::vector<PoseAnalysisResult>& results) {
    std::cout << "\n===== 姿态分析结果 =====\n";
    
    for (const auto& result : results) {
        std::cout << "Person " << result.id_priority_result.priority_id 
                  << " (ByteTrack ID: " << result.person_id << "):\n";
        
        // 身高信息
        if (result.height_result.is_stable) {
            std::cout << "  身高: " << std::fixed << std::setprecision(0) 
                     << result.height_result.estimated_height_mm << " mm"
                     << " (置信度: " << std::setprecision(2) 
                     << result.height_result.confidence << ")\n";
        } else {
            std::cout << "  身高: 测量中... (状态: " 
                     << static_cast<int>(result.height_result.state) << ")\n";
        }
        
        // 要球动作信息
        if (result.ball_request_result.is_confirmed) {
            std::cout << "  要球: 是 (持续: " 
                     << result.ball_request_result.request_duration_ms << " ms, "
                     << "置信度: " << result.ball_request_result.request_confidence << ")\n";
        } else if (result.ball_request_result.is_requesting) {
            std::cout << "  要球: 检测中... (连续帧: " 
                     << result.ball_request_result.continuous_frames << ")\n";
        } else {
            std::cout << "  要球: 否\n";
        }
        
        // ID优先级信息
        if (result.id_priority_result.can_swap_id) {
            std::cout << "  优先级分数: " << result.id_priority_result.priority_score << "\n";
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "姿态分析系统示例程序\n";
    std::cout << "====================\n\n";
    
    // 1. 创建分析器（使用配置文件或默认配置）
    std::unique_ptr<PoseAnalyzer> analyzer;
    
    if (argc > 1) {
        // 从配置文件加载
        std::string config_file = argv[1];
        std::cout << "加载配置文件: " << config_file << "\n";
        analyzer = create_pose_analyzer(config_file);
    } else {
        // 使用Builder模式创建自定义配置
        std::cout << "使用默认配置\n";
        analyzer = PoseAnalyzerBuilder()
            .height_filter_type("median")           // 使用中值滤波
            .height_window_size(15)                 // 15帧窗口
            .height_stability_threshold(50.0f)      // 50mm稳定阈值
            .ball_request_min_frames(5)             // 至少5帧确认要球
            .ball_request_max_interruption(2)       // 允许2帧中断
            .ball_request_confidence_threshold(3.5f) // 累积置信度阈值
            .id_priority_weights(0.3f, 0.4f, 0.3f)  // 置信度、持续时间、稳定性权重
            .id_swap_cooldown(2000)                 // 2秒冷却期
            .global_frame_buffer_size(60)           // 缓存60帧
            .build();
    }
    
    // 2. 设置Homography矩阵（如果有标定数据）
    cv::Mat homography = cv::Mat::eye(3, 3, CV_64F);
    // 实际使用时从标定文件加载：
    // cv::FileStorage fs("calibration.json", cv::FileStorage::READ);
    // fs["homography"] >> homography;
    analyzer->set_homography(homography);
    
    // 3. 创建调试可视化器（可选）
    DebugVisualizer visualizer;
    DataRecorder recorder("./analysis_records/", false);
    PerformanceMonitor perf_monitor;
    
    // 4. 开始记录
    recorder.start_recording();
    perf_monitor.enable_monitoring();
    
    // 5. 模拟处理100帧
    std::cout << "\n开始处理模拟数据...\n";
    
    for (int frame_id = 0; frame_id < 100; ++frame_id) {
        auto frame_start = std::chrono::high_resolution_clock::now();
        
        // 获取ByteTrack输出（实际使用时从ByteTracker获取）
        std::vector<PoseResult> pose_results = simulate_bytetrack_output(frame_id);
        
        // 执行姿态分析
        auto analysis_results = analyzer->analyze(pose_results);
        
        // 记录结果
        recorder.record_frame(analysis_results);
        
        // 每10帧输出一次结果
        if (frame_id % 10 == 0) {
            std::cout << "\n帧 " << frame_id << ":\n";
            print_analysis_results(analysis_results);
        }
        
        // 更新性能统计
        auto frame_end = std::chrono::high_resolution_clock::now();
        float processing_time = std::chrono::duration<float, std::milli>(
            frame_end - frame_start).count();
        perf_monitor.record_processing_time(processing_time);
        perf_monitor.record_frame_end(pose_results.size());
        
        // 模拟30 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    
    // 6. 停止记录并保存
    recorder.stop_recording();
    recorder.save_session("example_session");
    
    // 7. 输出统计信息
    std::cout << "\n\n===== 处理统计 =====\n";
    std::cout << analyzer->get_performance_stats() << "\n";
    std::cout << perf_monitor.get_performance_summary() << "\n";
    
    // 8. 输出调试信息
    std::cout << "\n===== 调试信息 =====\n";
    std::cout << analyzer->get_debug_info() << "\n";
    
    // 9. 生成分析报告
    recorder.generate_analysis_report("analysis_report.txt");
    
    std::cout << "\n示例程序完成！\n";
    std::cout << "记录保存在: ./analysis_records/\n";
    
    return 0;
}

/**
 * 编译命令：
 * g++ -std=c++17 pose_analysis_example.cpp \
 *     ../src/pose_analyzer.cpp \
 *     ../src/height_detector.cpp \
 *     ../src/ball_request_detector.cpp \
 *     ../src/id_priority_manager.cpp \
 *     -I../include \
 *     -lopencv_core -lopencv_imgproc \
 *     -ljsoncpp -pthread \
 *     -o pose_analysis_example
 * 
 * 运行命令：
 * ./pose_analysis_example [配置文件路径]
 */