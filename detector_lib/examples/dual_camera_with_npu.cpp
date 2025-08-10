/*-------------------------------------------
       双摄像头NPU优化示例程序
         
展示如何正确分配NPU核心避免资源冲突
基于RK3588S的3个NPU核心进行优化分配
-------------------------------------------*/

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "RimBasketballDetectorLib.h"
#include "detector_path_utils.h"
#include "npu_utils.h"

using namespace detector;

// 全局运行标志
std::atomic<bool> g_running(true);

// 姿态检测线程
void pose_detection_thread(int camera_id, int npu_core) {
    std::cout << "[姿态检测] 启动 - 摄像头:" << camera_id 
              << " NPU核心:" << npu_core << std::endl;
    
    try {
        // 打开摄像头
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开摄像头" + std::to_string(camera_id));
        }
        
        // 设置摄像头参数
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // 创建姿态检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_yolov8_pose.rknn");
        PoseDetectorLib detector(model_path, npu_core);
        
        // 启用跟踪和坐标映射
        detector.enable_tracking(true);
        std::string calib_path = PathUtils::find_calibration("2025_8_6_1280_720.json");
        if (!calib_path.empty()) {
            detector.load_calibration(calib_path);
        }
        
        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (g_running && cap.read(frame)) {
            // 执行检测
            auto results = detector.detect(frame);
            frame_count++;
            
            // 计算FPS
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed > 0) {
                float fps = frame_count / (float)elapsed;
                
                // 每30帧输出一次统计
                if (frame_count % 30 == 0) {
                    std::cout << "[姿态检测] FPS:" << fps 
                              << " 检测人数:" << results.size()
                              << " 推理时间:" << detector.get_last_inference_time_ms() << "ms"
                              << std::endl;
                }
            }
            
            // 绘制结果
            cv::Mat display = frame.clone();
            for (const auto& pose : results) {
                // 绘制骨架
                cv::rectangle(display, pose.bbox, cv::Scalar(0, 255, 0), 2);
                
                // 显示ID和地面坐标
                std::string info = "ID:" + std::to_string(pose.person_id);
                if (pose.has_ground_position) {
                    info += " Pos:(" + std::to_string((int)pose.ground_position.x) + 
                            "," + std::to_string((int)pose.ground_position.y) + ")mm";
                }
                cv::putText(display, info, cv::Point(pose.bbox.x, pose.bbox.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
            
            cv::imshow("姿态检测 - NPU" + std::to_string(npu_core), display);
            if (cv::waitKey(1) == 27) { // ESC退出
                g_running = false;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[姿态检测] 错误: " << e.what() << std::endl;
    }
}

// 篮筐检测线程
void rim_detection_thread(int camera_id, int npu_core) {
    std::cout << "[篮筐检测] 启动 - 摄像头:" << camera_id 
              << " NPU核心:" << npu_core << std::endl;
    
    try {
        // 打开摄像头
        cv::VideoCapture cap(camera_id);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开摄像头" + std::to_string(camera_id));
        }
        
        // 设置摄像头参数
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // 创建篮筐检测器，指定NPU核心
        std::string model_path = PathUtils::find_model("Q_Rim_Basketball_724_JZ.rknn");
        RimBasketballDetectorLib detector(model_path, npu_core);
        
        cv::Mat frame;
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (g_running && cap.read(frame)) {
            // 执行检测
            auto results = detector.detect(frame);
            frame_count++;
            
            // 统计
            int rim_count = 0, basketball_count = 0;
            for (const auto& obj : results) {
                if (obj.class_id == 1) rim_count++;
                else basketball_count++;
            }
            
            // 计算FPS
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed > 0) {
                float fps = frame_count / (float)elapsed;
                
                // 每30帧输出一次统计
                if (frame_count % 30 == 0) {
                    std::cout << "[篮筐检测] FPS:" << fps 
                              << " 篮筐:" << rim_count
                              << " 篮球:" << basketball_count
                              << " 推理时间:" << detector.get_last_inference_time_ms() << "ms"
                              << std::endl;
                }
            }
            
            // 绘制结果
            cv::Mat display = frame.clone();
            for (const auto& obj : results) {
                cv::Scalar color = (obj.class_id == 0) ? 
                    cv::Scalar(0, 165, 255) :  // 篮球-橙色
                    cv::Scalar(0, 255, 0);      // 篮筐-绿色
                    
                cv::rectangle(display, obj.bbox, color, 2);
                
                std::string label = obj.class_name + " " + 
                    std::to_string(obj.confidence).substr(0, 4);
                cv::putText(display, label, cv::Point(obj.bbox.x, obj.bbox.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
            }
            
            cv::imshow("篮筐检测 - NPU" + std::to_string(npu_core), display);
            if (cv::waitKey(1) == 27) { // ESC退出
                g_running = false;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[篮筐检测] 错误: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================" << std::endl;
    std::cout << "    双摄像头NPU优化示例程序" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // 获取NPU信息
    std::cout << "\n📊 检测NPU状态..." << std::endl;
    NPUInfo npu_info = NPUUtils::get_npu_info();
    std::cout << "NPU核心数: " << npu_info.total_cores << std::endl;
    std::cout << "当前频率: " << npu_info.current_freq_mhz << " MHz" << std::endl;
    std::cout << "温度: " << npu_info.temperature_celsius << "°C" << std::endl;
    std::cout << "调频策略: " << npu_info.governor << std::endl;
    
    // 获取推荐的NPU核心分配
    int pose_npu_core = 0;      // 姿态检测用核心0
    int rim_npu_core = 1;       // 篮筐检测用核心1
    
    // 如果用户指定了参数
    if (argc >= 3) {
        pose_npu_core = std::atoi(argv[1]);
        rim_npu_core = std::atoi(argv[2]);
        std::cout << "\n使用用户指定的NPU分配: " 
                  << "姿态->NPU" << pose_npu_core 
                  << ", 篮筐->NPU" << rim_npu_core << std::endl;
    } else {
        // 自动分配
        pose_npu_core = NPUUtils::get_recommended_core(-1);
        rim_npu_core = NPUUtils::get_recommended_core(pose_npu_core); // 避免使用相同核心
        
        // 确保不使用相同核心
        if (rim_npu_core == pose_npu_core) {
            rim_npu_core = (pose_npu_core + 1) % 3;
        }
        
        std::cout << "\n自动NPU核心分配: " 
                  << "姿态->NPU" << pose_npu_core 
                  << ", 篮筐->NPU" << rim_npu_core << std::endl;
    }
    
    // 摄像头分配
    int pose_camera = 0;  // 默认摄像头0用于姿态检测
    int rim_camera = 2;   // 默认摄像头2用于篮筐检测
    
    std::cout << "\n摄像头分配:" << std::endl;
    std::cout << "姿态检测: 摄像头" << pose_camera << " -> NPU" << pose_npu_core << std::endl;
    std::cout << "篮筐检测: 摄像头" << rim_camera << " -> NPU" << rim_npu_core << std::endl;
    
    std::cout << "\n按ESC键退出程序" << std::endl;
    std::cout << "启动检测线程..." << std::endl;
    
    // 错开初始化时间，避免同时初始化造成的冲突
    std::thread pose_thread(pose_detection_thread, pose_camera, pose_npu_core);
    std::this_thread::sleep_for(std::chrono::seconds(2)); // 延迟2秒
    std::thread rim_thread(rim_detection_thread, rim_camera, rim_npu_core);
    
    // 等待线程结束
    pose_thread.join();
    rim_thread.join();
    
    std::cout << "\n程序正常退出" << std::endl;
    
    return 0;
}