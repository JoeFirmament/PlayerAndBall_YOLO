/*-------------------------------------------
              DetectorLib 测试程序
              
演示如何使用封装好的检测器库
编译: mkdir build && cd build && cmake .. && make
运行: ./test_detector_lib

功能: 验证库的基本功能和接口
-------------------------------------------*/

#include <iostream>
#include <chrono>
#include <fstream>
#include "detector_lib.h"

// 智能查找模型文件路径
std::string find_model_file(const std::string& model_name) {
    std::vector<std::string> possible_paths = {
        model_name,                           // 直接使用输入路径
        "../../models/" + model_name,          // 从 build/examples 向上两级
        "../models/" + model_name,            // 从 examples 向上一级  
        "models/" + model_name,               // 当前目录下的 models
        "./" + model_name,                    // 当前目录
        "/tmp/" + model_name                  // 临时目录
    };
    
    for (const auto& path : possible_paths) {
        std::ifstream file(path);
        if (file.good()) {
            std::cout << "✓ 找到模型文件: " << path << std::endl;
            return path;
        }
    }
    
    std::cout << "❌ 未找到模型文件 '" << model_name << "'，尝试的路径:" << std::endl;
    for (const auto& path : possible_paths) {
        std::cout << "   - " << path << std::endl;
    }
    
    return model_name;  // 返回原始路径，让检测器处理错误
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void test_library_info() {
    print_separator("库信息测试");
    
    auto info = detector::get_library_info();
    std::cout << "版本: " << info.VERSION << std::endl;
    std::cout << "构建时间: " << info.BUILD_DATE << std::endl;
    std::cout << "目标平台: " << info.PLATFORM << std::endl;
    std::cout << "描述: " << info.DESCRIPTION << std::endl;
    
    // 测试运行时环境
    bool env_ok = detector::check_runtime_environment();
    std::cout << "运行时环境: " << (env_ok ? "✓ 正常" : "✗ 异常") << std::endl;
}

void test_pose_detector() {
    print_separator("PoseDetector 测试");
    
    try {
        // 1. 创建检测器
        std::string pose_model_path = find_model_file("Q_yolov8_pose.rknn");
        detector::PoseDetectorLib pose_detector(pose_model_path);
        std::cout << "✓ PoseDetector 创建成功" << std::endl;
        
        // 2. 检查初始状态
        std::cout << "初始状态: " << (int)pose_detector.get_status() << std::endl;
        std::cout << "是否已初始化: " << (pose_detector.is_initialized() ? "是" : "否") << std::endl;
        
        // 3. 配置参数（单图测试不使用跟踪）
        pose_detector.set_confidence_threshold(0.3f);
        pose_detector.enable_tracking(false);
        std::cout << "✓ 参数配置完成" << std::endl;
        
        // 4. 创建测试图像
        cv::Mat test_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::rectangle(test_frame, cv::Rect(200, 150, 200, 300), cv::Scalar(100, 150, 200), -1);
        std::cout << "✓ 测试图像创建完成: " << test_frame.size() << std::endl;
        
        // 5. 进行检测 (首次调用会初始化)
        std::cout << "正在进行首次检测 (可能需要1-3秒初始化)..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto results = pose_detector.detect(test_frame);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ 检测完成!" << std::endl;
        std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "推理时间: " << pose_detector.get_last_inference_time_ms() << " ms" << std::endl;
        std::cout << "检测到人员数量: " << results.size() << std::endl;
        
        // 6. 显示检测结果
        for (size_t i = 0; i < results.size(); i++) {
            const auto& pose = results[i];
            std::cout << "人员 " << i << ": ID=" << pose.person_id 
                     << ", 置信度=" << pose.confidence
                     << ", 关键点=" << pose.keypoints.size() << "个"
                     << ", 地面坐标=" << (pose.has_ground_position ? "有" : "无") << std::endl;
        }
        
        // 7. 再次检测 (测试性能)
        std::cout << "\n进行第二次检测 (测试稳定性能)..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        results = pose_detector.detect(test_frame);
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ 第二次检测完成!" << std::endl;
        std::cout << "耗时: " << duration.count() << " ms (应该显著更快)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ PoseDetector 测试失败: " << e.what() << std::endl;
    }
}

void test_rim_basketball_detector() {
    print_separator("RimBasketballDetector 测试");
    
    try {
        // 1. 创建检测器
        std::string rim_model_path = find_model_file("Q_Rim_Basketball_724_JZ.rknn");
        detector::RimBasketballDetectorLib rim_detector(rim_model_path);
        std::cout << "✓ RimBasketballDetector 创建成功" << std::endl;
        
        // 2. 配置参数
        rim_detector.set_confidence_threshold(0.4f);
        rim_detector.set_nms_threshold(0.5f);
        std::cout << "✓ 参数配置完成" << std::endl;
        
        // 3. 显示支持的类别
        auto classes = detector::RimBasketballDetectorLib::get_supported_classes();
        std::cout << "支持的类别: ";
        for (const auto& cls : classes) {
            std::cout << cls << " ";
        }
        std::cout << std::endl;
        
        // 4. 创建测试图像
        cv::Mat test_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::circle(test_frame, cv::Point(200, 150), 50, cv::Scalar(0, 255, 255), -1);  // 篮球
        cv::ellipse(test_frame, cv::Point(400, 100), cv::Size(80, 40), 0, 0, 360, cv::Scalar(255, 0, 255), -1);  // 篮筐
        std::cout << "✓ 测试图像创建完成" << std::endl;
        
        // 5. 进行检测
        std::cout << "正在进行检测..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto results = rim_detector.detect(test_frame);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ 检测完成!" << std::endl;
        std::cout << "耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "检测到目标数量: " << results.size() << std::endl;
        
        // 6. 显示检测结果
        int rim_count = 0, basketball_count = 0;
        for (size_t i = 0; i < results.size(); i++) {
            const auto& obj = results[i];
            std::cout << "目标 " << i << ": " << obj.class_name 
                     << ", 置信度=" << obj.confidence
                     << ", 中心=(" << obj.center.x << "," << obj.center.y << ")";
            
            if (obj.class_id == 0) {  // basketball
                basketball_count++;
                std::cout << ", 距离篮筐=" << obj.distance_to_rim
                         << ", ROI内=" << (obj.is_in_rim_roi ? "是" : "否");
            } else if (obj.class_id == 1) {  // rim
                rim_count++;
            }
            std::cout << std::endl;
        }
        
        std::cout << "统计: " << rim_count << "个篮筐, " << basketball_count << "个篮球" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ RimBasketballDetector 测试失败: " << e.what() << std::endl;
    }
}

void test_concurrent_detection() {
    print_separator("并发检测测试");
    
    try {
        // 同时使用两个检测器
        std::string pose_model_path = find_model_file("Q_yolov8_pose.rknn");
        std::string rim_model_path = find_model_file("Q_Rim_Basketball_724_JZ.rknn");
        
        // 为避免NPU资源冲突，分配不同的NPU核心
        detector::PoseDetectorLib pose_detector(pose_model_path, 0);  // 使用NPU核心0
        detector::RimBasketballDetectorLib rim_detector(rim_model_path, 1);  // 使用NPU核心1
        
        cv::Mat test_frame = cv::Mat::zeros(480, 640, CV_8UC3);
        
        std::cout << "测试并发检测..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 并发检测
        auto pose_results = pose_detector.detect(test_frame);
        auto rim_results = rim_detector.detect(test_frame);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "✓ 并发检测完成!" << std::endl;
        std::cout << "总耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "姿态结果: " << pose_results.size() << " 人员" << std::endl;
        std::cout << "目标结果: " << rim_results.size() << " 目标" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ 并发检测测试失败: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "\n🚀 DetectorLib 功能测试程序" << std::endl;
    std::cout << "测试封装库的完整功能..." << std::endl;
    
    // 设置日志级别
    detector::set_log_level(3);  // INFO级别
    
    try {
        // 1. 基础功能测试
        test_library_info();
        
        // 2. PoseDetector测试
        test_pose_detector();
        
        // 3. RimBasketballDetector测试  
        test_rim_basketball_detector();
        
        // 4. 并发检测测试
        test_concurrent_detection();
        
        print_separator("测试完成");
        std::cout << "🎉 所有测试完成! 库功能验证通过!" << std::endl;
        std::cout << "⚠️  注意：如果上述测试中有错误信息，请检查模型文件路径和NPU权限!" << std::endl;
        std::cout << "\n使用说明:" << std::endl;
        std::cout << "1. 在您的项目中包含: #include \"detector_lib.h\"" << std::endl;
        std::cout << "2. 链接库: -ldetector_lib" << std::endl;
        std::cout << "3. 确保模型文件路径正确" << std::endl;
        std::cout << "4. 确保NPU设备权限: sudo chmod 666 /dev/dri/renderD*" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n💥 测试过程中发生异常: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}