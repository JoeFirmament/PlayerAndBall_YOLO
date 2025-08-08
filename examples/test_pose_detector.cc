/*-------------------------------------------
              PoseDetector 测试程序
              
测试PoseDetector类的基本功能
编译: g++ -o test_pose_detector test_pose_detector.cc \
      ../src/PoseDetector.cc \
      -I../include -I../src \
      `pkg-config --cflags --libs opencv4` \
      -lrknn_api -pthread

用法: ./test_pose_detector [摄像头ID]
-------------------------------------------*/

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "PoseDetector.h"

// COCO 17关键点连接关系 (骨架)
const std::vector<std::pair<int, int>> skeleton = {
    {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
    {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, 
    {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
    {2, 4}, {3, 5}, {4, 6}, {5, 7}
};

// 绘制姿态检测结果
void draw_pose_results(cv::Mat& frame, const std::vector<PoseResult>& results) {
    for (const auto& pose : results) {
        // 绘制边界框
        cv::rectangle(frame, pose.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 显示人员ID和置信度
        std::string label = "ID:" + std::to_string(pose.person_id) + 
                           " (" + std::to_string((int)(pose.confidence * 100)) + "%)";
        cv::putText(frame, label, cv::Point(pose.bbox.x, pose.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        // 绘制关键点
        for (size_t i = 0; i < pose.keypoints.size(); i++) {
            const auto& kpt = pose.keypoints[i];
            if (pose.keypoint_scores[i] > 0.3f) {  // 只显示置信度较高的关键点
                cv::circle(frame, kpt, 4, cv::Scalar(0, 0, 255), -1);
                // 显示关键点索引
                cv::putText(frame, std::to_string(i), 
                           cv::Point(kpt.x + 5, kpt.y - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
            }
        }
        
        // 绘制骨架连接
        for (const auto& connection : skeleton) {
            int idx1 = connection.first - 1;   // COCO索引从1开始，转换为0开始
            int idx2 = connection.second - 1;
            
            if (idx1 >= 0 && idx1 < pose.keypoints.size() && 
                idx2 >= 0 && idx2 < pose.keypoints.size() &&
                pose.keypoint_scores[idx1] > 0.3f && 
                pose.keypoint_scores[idx2] > 0.3f) {
                
                cv::line(frame, pose.keypoints[idx1], pose.keypoints[idx2], 
                        cv::Scalar(255, 0, 0), 2);
            }
        }
        
        // 显示地面坐标 (如果有)
        if (pose.has_ground_position) {
            std::string ground_pos = "Ground: (" + 
                std::to_string((int)pose.ground_position.x) + ", " +
                std::to_string((int)pose.ground_position.y) + ")";
            cv::putText(frame, ground_pos, 
                       cv::Point(pose.bbox.x, pose.bbox.y + pose.bbox.height + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== PoseDetector 测试程序 ===" << std::endl;
    
    // 解析命令行参数
    int camera_id = 0;
    if (argc >= 2) {
        camera_id = std::atoi(argv[1]);
    }
    
    std::cout << "使用摄像头: " << camera_id << std::endl;
    std::cout << "模型路径: models/Q_yolov8_pose.rknn" << std::endl;
    
    // 1. 创建PoseDetector (这里就是我们封装的核心!)
    std::cout << "正在创建PoseDetector..." << std::endl;
    PoseDetector detector("models/Q_yolov8_pose.rknn");
    
    // 2. 可选配置
    detector.set_confidence_threshold(0.3f);
    detector.enable_tracking(true);
    std::cout << "检测器配置完成" << std::endl;
    
    // 3. 初始化摄像头 (用户自己处理)
    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开摄像头 " << camera_id << std::endl;
        return -1;
    }
    
    // 设置摄像头参数
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    
    std::cout << "摄像头初始化完成" << std::endl;
    std::cout << "按键说明: ESC=退出, T=切换跟踪, 空格=截图" << std::endl;
    
    // 4. 预热检测器 (推荐!)
    std::cout << "正在预热检测器 (首次初始化可能需要1-3秒)..." << std::endl;
    cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    detector.detect(dummy_frame);  // 预热调用
    std::cout << "预热完成！开始实时检测..." << std::endl;
    
    // 5. 主检测循环
    cv::Mat frame;
    bool tracking_enabled = true;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // 采集图像 (用户自己处理)
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "错误: 无法读取摄像头图像" << std::endl;
            break;
        }
        
        frame_count++;
        
        // 核心检测接口 (就这一行代码!)
        auto detect_start = std::chrono::high_resolution_clock::now();
        std::vector<PoseResult> results = detector.detect(frame);  // ⭐ 核心接口
        auto detect_end = std::chrono::high_resolution_clock::now();
        
        // 计算推理时间 (用户自己处理性能统计)
        auto inference_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start);
        
        // 检查检测结果
        if (results.empty() && !detector.is_initialized()) {
            // 初始化失败
            std::cerr << "错误: 检测器初始化失败！检查模型文件和NPU权限" << std::endl;
            break;
        }
        
        // 绘制结果 (用户自己处理显示)
        draw_pose_results(frame, results);
        
        // 显示性能信息
        std::string perf_info = "推理: " + std::to_string(inference_time.count()) + "ms, " + 
                               "检测: " + std::to_string(results.size()) + " 人员";
        cv::putText(frame, perf_info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // 显示跟踪状态
        std::string track_info = "跟踪: " + std::string(tracking_enabled ? "开启" : "关闭");
        cv::putText(frame, track_info, cv::Point(10, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // 计算整体FPS
        if (frame_count % 30 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            float fps = 30000.0f / elapsed.count();
            std::cout << "FPS: " << fps << ", 推理时间: " << inference_time.count() << "ms" << std::endl;
            start_time = current_time;
        }
        
        // 显示结果
        cv::imshow("PoseDetector 测试", frame);
        
        // 按键处理
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) {  // ESC键退出
            break;
        } else if (key == 't' || key == 'T') {  // T键切换跟踪
            tracking_enabled = !tracking_enabled;
            detector.enable_tracking(tracking_enabled);
            std::cout << "ByteTrack跟踪: " << (tracking_enabled ? "开启" : "关闭") << std::endl;
        } else if (key == ' ') {  // 空格键截图
            std::string filename = "pose_screenshot_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "截图保存: " << filename << std::endl;
        }
        
        // 打印详细检测信息 (每10帧)
        if (frame_count % 10 == 0 && !results.empty()) {
            std::cout << "--- 第" << frame_count << "帧检测结果 ---" << std::endl;
            for (size_t i = 0; i < results.size(); i++) {
                const auto& pose = results[i];
                std::cout << "人员" << i << ": ID=" << pose.person_id 
                         << ", 置信度=" << pose.confidence 
                         << ", 边界框=(" << pose.bbox.x << "," << pose.bbox.y 
                         << "," << pose.bbox.width << "," << pose.bbox.height << ")"
                         << ", 关键点=" << pose.keypoints.size() << "个" << std::endl;
            }
        }
    }
    
    // 6. 清理资源 (析构函数自动清理，用户无需手动操作)
    std::cout << "程序退出，资源已自动清理" << std::endl;
    return 0;
}