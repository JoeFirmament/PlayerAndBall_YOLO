#include <iostream>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"

using namespace detector;

int main() {
    // 测试ByteTrack和Homography集成功能
    std::cout << "=== 姿态检测库集成测试 ===" << std::endl;
    
    // 1. 创建检测器实例
    const std::string model_path = "../models/Q_yolov8_pose.rknn";
    PoseDetectorLib detector(model_path);
    
    std::cout << "✓ 检测器创建成功" << std::endl;
    
    // 2. 测试ByteTrack功能开关
    detector.enable_tracking(true);
    std::cout << "✓ 启用ByteTrack跟踪" << std::endl;
    
    // 3. 测试Homography标定加载
    const std::string calibration_file = "../data/2025_7_11pm.json";
    bool calib_result = detector.load_calibration(calibration_file);
    if (calib_result) {
        std::cout << "✓ Homography标定加载成功" << std::endl;
    } else {
        std::cout << "⚠ Homography标定文件不存在或加载失败，将跳过地面坐标计算" << std::endl;
    }
    
    // 4. 设置置信度阈值
    detector.set_confidence_threshold(0.25f);
    std::cout << "✓ 设置置信度阈值: 0.25" << std::endl;
    
    // 5. 测试模拟图像检测（创建一个空白图像）
    cv::Mat test_frame = cv::Mat::zeros(640, 640, CV_8UC3);
    std::cout << "✓ 创建测试图像: " << test_frame.size() << std::endl;
    
    // 6. 进行检测
    std::cout << "\n开始检测测试..." << std::endl;
    auto results = detector.detect(test_frame);
    
    // 7. 显示结果
    std::cout << "检测结果数量: " << results.size() << std::endl;
    
    for (size_t i = 0; i < results.size(); i++) {
        const PoseResult& result = results[i];
        std::cout << "人员[" << i << "]:" << std::endl;
        std::cout << "  - person_id: " << result.person_id << std::endl;
        std::cout << "  - confidence: " << result.confidence << std::endl;
        std::cout << "  - bbox: (" << result.bbox.x << ", " << result.bbox.y 
                  << ", " << result.bbox.width << ", " << result.bbox.height << ")" << std::endl;
        std::cout << "  - keypoints count: " << result.keypoints.size() << std::endl;
        std::cout << "  - has_ground_position: " << (result.has_ground_position ? "是" : "否") << std::endl;
        
        if (result.has_ground_position) {
            std::cout << "  - ground_position: (" << result.ground_position.x 
                      << ", " << result.ground_position.y << ")mm" << std::endl;
        }
    }
    
    // 8. 测试状态查询
    std::cout << "\n检测器状态:" << std::endl;
    std::cout << "  - is_initialized: " << (detector.is_initialized() ? "是" : "否") << std::endl;
    std::cout << "  - status: " << detector.get_status() << std::endl;
    std::cout << "  - last_inference_time: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}