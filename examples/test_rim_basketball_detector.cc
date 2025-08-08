/*-------------------------------------------
          RimBasketballDetector 测试程序
              
测试RimBasketballDetector类的基本功能
编译: g++ -o test_rim_basketball_detector test_rim_basketball_detector.cc \
      ../src/RimBasketballDetector.cc \
      -I../include -I../src \
      `pkg-config --cflags --libs opencv4` \
      -lrknn_api -pthread

用法: ./test_rim_basketball_detector [摄像头ID]
-------------------------------------------*/

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "RimBasketballDetector.h"

// 绘制篮筐篮球检测结果
void draw_rim_basketball_results(cv::Mat& frame, const std::vector<RimBasketballResult>& results) {
    for (const auto& result : results) {
        // 不同类别使用不同颜色
        cv::Scalar color;
        if (result.class_id == 1) {  // rim
            color = cv::Scalar(255, 0, 255);  // 紫色
        } else {  // basketball
            color = cv::Scalar(0, 255, 255);  // 黄色
        }
        
        // 绘制边界框
        cv::rectangle(frame, result.bbox, color, 3);
        
        // 显示类别、置信度
        std::string label = result.class_name + " " + 
                           std::to_string((int)(result.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(result.bbox.x, result.bbox.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        
        // 显示中心点
        cv::circle(frame, result.center, 6, color, -1);
        cv::circle(frame, result.center, 8, cv::Scalar(255, 255, 255), 2);
        
        // 对于篮球，显示额外信息
        if (result.class_id == 0) {  // basketball
            if (result.distance_to_rim > 0) {
                std::string distance_info = "距离篮筐: " + std::to_string((int)result.distance_to_rim) + "px";
                cv::putText(frame, distance_info, 
                           cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
            
            if (result.is_in_rim_roi) {
                cv::putText(frame, "⭐ 靠近篮筐!", 
                           cv::Point(result.bbox.x, result.bbox.y + result.bbox.height + 40),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                
                // 绘制闪烁效果
                static int blink_counter = 0;
                blink_counter++;
                if ((blink_counter / 10) % 2 == 0) {
                    cv::rectangle(frame, result.bbox, cv::Scalar(0, 255, 0), 5);
                }
            }
        }
        
        // 绘制检测序号
        cv::putText(frame, "#" + std::to_string(&result - &results[0]), 
                   cv::Point(result.center.x - 10, result.center.y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 3);
        cv::putText(frame, "#" + std::to_string(&result - &results[0]), 
                   cv::Point(result.center.x - 10, result.center.y + 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

// 分析检测结果并显示统计信息
void analyze_results(cv::Mat& frame, const std::vector<RimBasketballResult>& results) {
    int rim_count = 0, basketball_count = 0;
    int close_basketballs = 0;
    
    // 统计各类别数量
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
    
    // 显示统计信息
    std::string stats = "篮筐: " + std::to_string(rim_count) + 
                       ", 篮球: " + std::to_string(basketball_count) + 
                       " (靠近: " + std::to_string(close_basketballs) + ")";
    
    cv::rectangle(frame, cv::Point(10, frame.rows - 80), cv::Point(600, frame.rows - 10), 
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(frame, stats, cv::Point(20, frame.rows - 50), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // 显示场景判断
    std::string scene_info;
    if (rim_count > 0 && basketball_count > 0) {
        if (close_basketballs > 0) {
            scene_info = "🏀 投篮场景 - 篮球靠近篮筐!";
        } else {
            scene_info = "🏀 篮球场景 - 篮球远离篮筐";
        }
    } else if (rim_count > 0) {
        scene_info = "🏀 篮筐场景 - 等待篮球出现";
    } else if (basketball_count > 0) {
        scene_info = "🏀 篮球场景 - 寻找篮筐中";
    } else {
        scene_info = "🔍 搜索中 - 未发现目标";
    }
    
    cv::putText(frame, scene_info, cv::Point(20, frame.rows - 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
}

int main(int argc, char* argv[]) {
    std::cout << "=== RimBasketballDetector 测试程序 ===" << std::endl;
    
    // 解析命令行参数
    int camera_id = 2;  // 默认使用摄像头2 (篮筐检测摄像头)
    if (argc >= 2) {
        camera_id = std::atoi(argv[1]);
    }
    
    std::cout << "使用摄像头: " << camera_id << std::endl;
    std::cout << "模型路径: models/Q_Rim_Basketball_724_JZ.rknn" << std::endl;
    
    // 1. 创建RimBasketballDetector (这里就是我们封装的核心!)
    std::cout << "正在创建RimBasketballDetector..." << std::endl;
    RimBasketballDetector detector("models/Q_Rim_Basketball_724_JZ.rknn");
    
    // 2. 可选配置
    detector.set_confidence_threshold(0.4f);  // 篮球检测要求更高置信度
    detector.set_nms_threshold(0.5f);
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
    std::cout << "按键说明: ESC=退出, S=截图, C=调整置信度" << std::endl;
    
    // 4. 预热检测器 (推荐!)
    std::cout << "正在预热检测器 (首次初始化可能需要1-3秒)..." << std::endl;
    cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);
    detector.detect(dummy_frame);  // 预热调用
    std::cout << "预热完成！开始实时检测..." << std::endl;
    
    // 5. 主检测循环
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    float current_conf_threshold = 0.4f;
    
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
        std::vector<RimBasketballResult> results = detector.detect(frame);  // ⭐ 核心接口
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
        draw_rim_basketball_results(frame, results);
        analyze_results(frame, results);
        
        // 显示性能信息
        std::string perf_info = "推理: " + std::to_string(inference_time.count()) + "ms, " + 
                               "目标: " + std::to_string(results.size()) + " 个";
        cv::putText(frame, perf_info, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // 显示置信度阈值
        std::string conf_info = "置信度阈值: " + std::to_string(current_conf_threshold);
        cv::putText(frame, conf_info, cv::Point(10, 60), 
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
        cv::imshow("RimBasketballDetector 测试", frame);
        
        // 按键处理
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) {  // ESC键退出
            break;
        } else if (key == 's' || key == 'S') {  // S键截图
            std::string filename = "rim_basketball_screenshot_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(filename, frame);
            std::cout << "截图保存: " << filename << std::endl;
        } else if (key == 'c' || key == 'C') {  // C键调整置信度
            current_conf_threshold += 0.1f;
            if (current_conf_threshold > 0.9f) current_conf_threshold = 0.1f;
            detector.set_confidence_threshold(current_conf_threshold);
            std::cout << "置信度阈值调整为: " << current_conf_threshold << std::endl;
        }
        
        // 打印详细检测信息 (每20帧，避免刷屏)
        if (frame_count % 20 == 0 && !results.empty()) {
            std::cout << "--- 第" << frame_count << "帧检测结果 ---" << std::endl;
            for (size_t i = 0; i < results.size(); i++) {
                const auto& obj = results[i];
                std::cout << "目标" << i << ": " << obj.class_name 
                         << ", 置信度=" << obj.confidence 
                         << ", 边界框=(" << obj.bbox.x << "," << obj.bbox.y 
                         << "," << obj.bbox.width << "," << obj.bbox.height << ")"
                         << ", 中心=(" << obj.center.x << "," << obj.center.y << ")";
                
                if (obj.class_id == 0) {  // basketball
                    std::cout << ", 距离篮筐=" << obj.distance_to_rim 
                             << ", ROI内=" << (obj.is_in_rim_roi ? "是" : "否");
                }
                std::cout << std::endl;
            }
        }
    }
    
    // 6. 清理资源 (析构函数自动清理，用户无需手动操作)
    std::cout << "程序退出，资源已自动清理" << std::endl;
    return 0;
}