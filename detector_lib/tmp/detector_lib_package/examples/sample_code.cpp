#include "detector_lib.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <图片路径>" << std::endl;
        return -1;
    }
    
    try {
        // 1. 创建检测器
        detector::PoseDetectorLib detector("models/Q_yolov8_pose.rknn");
        
        // 2. 加载图片
        cv::Mat image = cv::imread(argv[1]);
        if (image.empty()) {
            std::cout << "错误: 无法加载图片 " << argv[1] << std::endl;
            return -1;
        }
        
        // 3. 执行检测
        std::cout << "正在检测..." << std::endl;
        auto results = detector.detect(image);
        
        // 4. 显示结果
        std::cout << "检测到 " << results.size() << " 个人员:" << std::endl;
        for (const auto& pose : results) {
            std::cout << "  - 人员ID: " << pose.person_id 
                      << ", 置信度: " << pose.confidence
                      << ", 位置: (" << pose.bbox.x << "," << pose.bbox.y 
                      << "," << pose.bbox.width << "," << pose.bbox.height << ")"
                      << std::endl;
        }
        
        std::cout << "检测完成！推理时间: " << detector.get_last_inference_time_ms() << "ms" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
