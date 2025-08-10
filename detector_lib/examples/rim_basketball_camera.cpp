/*-------------------------------------------
          篮筐篮球摄像头实时检测程序
          
基于 rim_basketball_image.cpp 创建的摄像头版本
运行: ./rim_basketball_camera [模型路径] [摄像头路径或设备号]
默认使用篮筐检测专用摄像头: usb-DECXIN_CAMERA
按键控制:
  - ESC: 退出程序
  - S: 截图保存当前帧检测结果
-------------------------------------------*/

#include <iostream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "RimBasketballDetectorLib.h"

int main(int argc, char* argv[]) {
    std::cout << "=== 篮筐篮球摄像头实时检测程序 ===" << std::endl;
    
    // 参数解析
    std::string model_path = "../models/Q_Rim_Basketball_724_JZ.rknn";  // 默认模型路径
    std::string camera_path = "/dev/v4l/by-id/usb-DECXIN_CAMERA_DECXIN_CAMERA_01.00.00-video-index0";  // 篮筐检测专用摄像头
    
    if (argc >= 2) {
        model_path = argv[1];
    }
    if (argc >= 3) {
        camera_path = argv[2];  // 支持指定摄像头路径或设备号
    }
    
    std::cout << "模型: " << model_path << std::endl;
    std::cout << "摄像头设备: " << camera_path << std::endl;
    
    try {
        // 初始化摄像头
        cv::VideoCapture cap;
        
        // 检查设备路径是否存在，支持持久化路径和设备号
        bool opened = false;
        if (camera_path.find("/dev/v4l/by-id/") == 0) {
            // 使用持久化路径
            if (!cap.open(camera_path, cv::CAP_V4L2)) {
                std::cout << "⚠️ 持久化路径失败，尝试默认设备..." << std::endl;
                opened = cap.open(0);  // 回退到默认摄像头
            } else {
                opened = true;
            }
        } else {
            // 尝试作为设备号解析
            try {
                int cam_index = std::stoi(camera_path);
                if (!cap.open(cam_index, cv::CAP_V4L2)) {
                    std::cout << "⚠️ V4L2后端失败，尝试默认后端..." << std::endl;
                    opened = cap.open(cam_index);
                } else {
                    opened = true;
                }
            } catch (...) {
                std::cerr << "❌ 无效的摄像头路径或设备号: " << camera_path << std::endl;
                return -1;
            }
        }
        
        if (!opened) {
            std::cerr << "❌ 无法打开摄像头设备: " << camera_path << std::endl;
            return -1;
        }
        
        // 设置摄像头参数 - 先设置MJPEG格式再设置分辨率帧率
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));  // 先设置MJPEG格式
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        
        // 延时让摄像头设置生效
        usleep(200*1000);  // 200ms延时
        
        // 验证摄像头参数设置
        int actual_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int actual_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double actual_fps = cap.get(cv::CAP_PROP_FPS);
        
        std::cout << "✅ 摄像头初始化成功: " << actual_width << "x" << actual_height 
                  << " @ " << actual_fps << "fps" << std::endl;
        
        // 创建篮筐篮球检测器
        std::cout << "正在加载篮筐篮球检测器..." << std::endl;
        detector::RimBasketballDetectorLib detector(model_path);
        
        // 配置检测参数 - 使用与图像版本相同的阈值
        detector.set_confidence_threshold(0.25f);
        detector.set_nms_threshold(0.1f);
        
        std::cout << "✅ 检测器初始化完成" << std::endl;
        std::cout << "\n按键控制:" << std::endl;
        std::cout << "  ESC - 退出程序" << std::endl;
        std::cout << "  S   - 截图保存当前帧" << std::endl;
        std::cout << "\n开始实时检测..." << std::endl;
        
        cv::Mat frame;
        int frame_count = 0;
        int screenshot_count = 1;
        
        // FPS计算变量
        auto start_time = std::chrono::steady_clock::now();
        int fps_frame_count = 0;
        double current_fps = 0.0;
        
        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "❌ 摄像头读取失败" << std::endl;
                break;
            }
            
            frame_count++;
            fps_frame_count++;
            
            // 每30帧计算一次FPS
            if (fps_frame_count >= 30) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                current_fps = fps_frame_count * 1000.0 / elapsed.count();
                start_time = current_time;
                fps_frame_count = 0;
            }
            
            // 执行检测
            auto results = detector.detect(frame);
            
            // 分析检测结果
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
            
            // 创建显示图像
            cv::Mat display_frame = frame.clone();
            
            // 绘制检测结果
            cv::Point2f rim_center(-1, -1);
            bool has_rim = false;
            
            for (const auto& obj : results) {
                // 确定颜色
                cv::Scalar color;
                if (obj.class_id == 0) {  // basketball
                    color = cv::Scalar(0, 165, 255);  // 橙色
                } else {  // rim
                    color = cv::Scalar(0, 255, 0);    // 绿色
                    rim_center = obj.center;
                    has_rim = true;
                }
                
                // 绘制检测框
                cv::rectangle(display_frame, obj.bbox, color, 3);
                
                // 绘制置信度和类别标签
                std::string label = obj.class_name + " " + std::to_string(obj.confidence).substr(0, 4);
                int baseline;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
                cv::Point label_origin(obj.bbox.x, obj.bbox.y - 10);
                
                // 标签背景
                cv::rectangle(display_frame, 
                             cv::Point(label_origin.x, label_origin.y - label_size.height - baseline),
                             cv::Point(label_origin.x + label_size.width, label_origin.y + baseline), 
                             color, -1);
                
                // 标签文字
                cv::putText(display_frame, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);
                
                // 绘制中心点
                cv::circle(display_frame, obj.center, 4, color, -1);
                
                // 如果是篮球且靠近篮筐，添加特殊标记
                if (obj.class_id == 0 && obj.is_in_rim_roi) {
                    cv::putText(display_frame, "CLOSE", 
                               cv::Point(obj.bbox.x, obj.bbox.y + obj.bbox.height + 20), 
                               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                }
            }
            
            // 绘制距离连线
            if (has_rim) {
                for (const auto& result : results) {
                    if (result.class_id == 0 && result.distance_to_rim > 0) {  // basketball
                        // 绘制到篮筐的连线
                        cv::line(display_frame, result.center, rim_center, cv::Scalar(255, 255, 0), 2);
                        
                        // 显示距离
                        cv::Point mid_point((result.center.x + rim_center.x) / 2, 
                                          (result.center.y + rim_center.y) / 2 - 10);
                        std::string dist_text = std::to_string((int)result.distance_to_rim) + "px";
                        cv::putText(display_frame, dist_text, mid_point, cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, cv::Scalar(255, 255, 0), 2);
                    }
                }
            }
            
            // 显示每个目标的NCWH信息
            for (const auto& obj : results) {
                std::string ncwh_info;
                if (obj.class_id == 0) {  // basketball
                    ncwh_info = "B: " + std::to_string(obj.bbox.width) + "x" + std::to_string(obj.bbox.height);
                } else {  // rim
                    ncwh_info = "R: " + std::to_string(obj.bbox.width) + "x" + std::to_string(obj.bbox.height);
                }
                
                // 显示NCWH信息在检测框正中心
                cv::putText(display_frame, ncwh_info, 
                           cv::Point(obj.bbox.x + obj.bbox.width/2 - 25, obj.bbox.y + obj.bbox.height/2), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            }
            
            // 绘制信息面板
            cv::Scalar panel_color(50, 50, 50);
            cv::rectangle(display_frame, cv::Point(10, 10), cv::Point(400, 120), panel_color, -1);
            
            // 显示统计信息
            cv::putText(display_frame, "Rim Basketball Detector", cv::Point(20, 35), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
            
            char info_text[200];
            snprintf(info_text, sizeof(info_text), "Basketballs: %d, Rims: %d, Close: %d", 
                     basketball_count, rim_count, close_basketballs);
            cv::putText(display_frame, info_text, cv::Point(20, 60), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            
            snprintf(info_text, sizeof(info_text), "FPS: %.1f, Frame: %d", current_fps, frame_count);
            cv::putText(display_frame, info_text, cv::Point(20, 85), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            
            snprintf(info_text, sizeof(info_text), "Inference: %dms", detector.get_last_inference_time_ms());
            cv::putText(display_frame, info_text, cv::Point(20, 105), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
            
            // 显示图像
            cv::imshow("Rim Basketball Camera Detection", display_frame);
            
            // 按键处理
            int key = cv::waitKey(1) & 0xFF;
            if (key == 27) {  // ESC键退出
                std::cout << "\n用户按ESC键，程序退出" << std::endl;
                break;
            } else if (key == 's' || key == 'S') {  // S键截图
                char filename[64];
                snprintf(filename, sizeof(filename), "rim_basketball_screenshot_%04d.jpg", screenshot_count);
                bool save_success = cv::imwrite(filename, display_frame);
                
                if (save_success) {
                    std::cout << "📷 截图已保存: " << filename << std::endl;
                    screenshot_count++;
                } else {
                    std::cout << "❌ 截图保存失败!" << std::endl;
                }
            }
            
            // 每100帧输出一次检测统计（可选，用于调试）
            if (frame_count % 100 == 0) {
                std::cout << "[Frame " << frame_count << "] "
                          << "🏀:" << basketball_count << " 🎯:" << rim_count 
                          << " ⭐:" << close_basketballs 
                          << " FPS:" << std::fixed << std::setprecision(1) << current_fps << std::endl;
            }
        }
        
        // 清理资源
        cap.release();
        cv::destroyAllWindows();
        
        std::cout << "\n✅ 程序正常结束" << std::endl;
        std::cout << "总共处理帧数: " << frame_count << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 程序异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}