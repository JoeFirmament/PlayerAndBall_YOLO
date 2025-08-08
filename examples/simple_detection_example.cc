/*-------------------------------------------
              简单检测示例程序
              
演示如何使用PoseDetector和RimBasketballDetector
进行姿态检测和篮筐篮球检测

编译: g++ -o simple_example simple_detection_example.cc \
      -I../include -L../build -lPoseDetector -lRimBasketballDetector \
      `pkg-config --cflags --libs opencv4`

用法: 
./simple_example                                    # 使用默认摄像头
./simple_example /dev/video0 /dev/video2           # 指定双摄像头
-------------------------------------------*/

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "PoseDetector.h"
#include "RimBasketballDetector.h"

// 绘制姿态检测结果
void draw_pose_results(cv::Mat& frame, const std::vector<PoseResult>& results) {
    // COCO 17关键点连接关系 (骨架)
    const std::vector<std::pair<int, int>> skeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
        {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9}, 
        {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
        {2, 4}, {3, 5}, {4, 6}, {5, 7}
    };
    
    for (const auto& pose : results) {
        // 绘制边界框
        cv::rectangle(frame, pose.bbox, cv::Scalar(0, 255, 0), 2);
        
        // 显示人员ID和置信度
        std::string label = "ID:" + std::to_string(pose.person_id) + 
                           " (" + std::to_string((int)(pose.confidence * 100)) + "%)";
        cv::putText(frame, label, cv::Point(pose.bbox.x, pose.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        
        // 绘制关键点
        for (size_t i = 0; i < pose.keypoints.size(); i++) {
            const auto& kpt = pose.keypoints[i];
            if (pose.keypoint_scores[i] > 0.3f) {  // 只显示置信度较高的关键点
                cv::circle(frame, kpt, 3, cv::Scalar(0, 0, 255), -1);
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
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        }
    }
}

// 绘制篮筐篮球检测结果
void draw_rim_basketball_results(cv::Mat& frame, const std::vector<RimBasketballResult>& results) {
    for (const auto& result : results) {
        // 不同类别使用不同颜色
        cv::Scalar color = (result.class_id == 1) ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 255);  // rim=紫色, basketball=黄色
        
        // 绘制边界框
        cv::rectangle(frame, result.bbox, color, 2);
        
        // 显示类别、置信度
        std::string label = result.class_name + " " + 
                           std::to_string((int)(result.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        
        // 显示中心点
        cv::circle(frame, result.center, 5, color, -1);
        
        // 对于篮球，显示额外信息
        if (result.class_id == 0) {  // basketball
            if (result.distance_to_rim > 0) {
                std::string distance_info = "Dist: " + std::to_string((int)result.distance_to_rim);
                cv::putText(frame, distance_info, 
                           cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 15),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
            
            if (result.is_in_rim_roi) {
                cv::putText(frame, "IN RIM ROI", 
                           cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 30),
                           cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0), 1);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== 简单检测示例程序 ===" << std::endl;
    
    // 解析命令行参数
    int pose_camera_id = 0;
    int rim_camera_id = 2;
    
    if (argc >= 2) {
        pose_camera_id = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        rim_camera_id = std::atoi(argv[2]);
    }
    
    std::cout << "姿态检测摄像头: " << pose_camera_id << std::endl;
    std::cout << "篮筐检测摄像头: " << rim_camera_id << std::endl;
    
    // 1. 初始化检测器 (延迟初始化，用户不需要显式调用init)
    std::cout << "正在初始化检测器..." << std::endl;
    
    PoseDetector pose_detector("../models/Q_yolov8_pose.rknn");
    RimBasketballDetector rim_detector("../models/Q_Rim_Basketball_724_JZ.rknn");
    
    // 可选：加载标定文件
    if (!pose_detector.load_calibration("../data/2025_7_11pm.json")) {
        std::cout << "警告: 无法加载标定文件，将不进行坐标映射" << std::endl;
    }
    
    // 可选：调整参数
    pose_detector.set_confidence_threshold(0.3f);
    rim_detector.set_confidence_threshold(0.4f);
    
    // 2. 初始化摄像头 (用户自己处理)
    cv::VideoCapture pose_cap(pose_camera_id);
    cv::VideoCapture rim_cap(rim_camera_id);
    
    if (!pose_cap.isOpened()) {
        std::cerr << "错误: 无法打开姿态检测摄像头 " << pose_camera_id << std::endl;
        return -1;
    }
    
    if (!rim_cap.isOpened()) {
        std::cerr << "错误: 无法打开篮筐检测摄像头 " << rim_camera_id << std::endl;
        return -1;
    }
    
    // 设置摄像头分辨率
    pose_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    pose_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    pose_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    
    rim_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    rim_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    rim_cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    
    std::cout << "摄像头初始化完成，按ESC退出程序" << std::endl;
    
    // 3. 主检测循环
    cv::Mat pose_frame, rim_frame;
    bool tracking_enabled = true;
    
    while (true) {
        // 采集图像 (用户自己处理)
        pose_cap >> pose_frame;
        rim_cap >> rim_frame;
        
        if (pose_frame.empty() || rim_frame.empty()) {
            std::cerr << "错误: 无法读取摄像头图像" << std::endl;
            break;
        }
        
        // 进行推理 (我们的接口，超简单!)
        auto pose_start = std::chrono::high_resolution_clock::now();
        std::vector<PoseResult> pose_results = pose_detector.detect(pose_frame);
        auto pose_end = std::chrono::high_resolution_clock::now();
        
        auto rim_start = std::chrono::high_resolution_clock::now();
        std::vector<RimBasketballResult> rim_results = rim_detector.detect(rim_frame);
        auto rim_end = std::chrono::high_resolution_clock::now();
        
        // 计算推理时间 (用户自己处理性能统计)
        auto pose_time = std::chrono::duration_cast<std::chrono::milliseconds>(pose_end - pose_start);
        auto rim_time = std::chrono::duration_cast<std::chrono::milliseconds>(rim_end - rim_start);
        
        // 绘制结果 (用户自己处理显示)
        draw_pose_results(pose_frame, pose_results);
        draw_rim_basketball_results(rim_frame, rim_results);
        
        // 显示性能信息
        std::string pose_info = "Pose: " + std::to_string(pose_time.count()) + "ms, " + 
                               std::to_string(pose_results.size()) + " persons";
        std::string rim_info = "Rim: " + std::to_string(rim_time.count()) + "ms, " + 
                              std::to_string(rim_results.size()) + " objects";
        
        cv::putText(pose_frame, pose_info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(rim_frame, rim_info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // 显示结果
        cv::imshow("姿态检测", pose_frame);
        cv::imshow("篮筐篮球检测", rim_frame);
        
        // 按键处理
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) {  // ESC键退出
            break;
        } else if (key == 't' || key == 'T') {  // T键切换跟踪
            tracking_enabled = !tracking_enabled;
            pose_detector.enable_tracking(tracking_enabled);
            std::cout << "ByteTrack跟踪: " << (tracking_enabled ? "开启" : "关闭") << std::endl;
        }
        
        // 打印检测统计 (可选)
        if (pose_results.size() > 0 || rim_results.size() > 0) {
            std::cout << "检测到: " << pose_results.size() << " 人员, " 
                     << rim_results.size() << " 目标" << std::endl;
        }
    }
    
    // 4. 清理资源 (析构函数自动清理，用户不需要手动调用)
    std::cout << "程序退出" << std::endl;
    return 0;
}