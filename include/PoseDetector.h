#ifndef _POSE_DETECTOR_H_
#define _POSE_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "rknn_api.h"

// 姿态检测结果结构体
struct PoseResult {
    int person_id;                      // 人员ID (ByteTrack跟踪)
    float confidence;                   // 检测置信度
    cv::Rect bbox;                      // 边界框
    std::vector<cv::Point2f> keypoints; // 17个关键点 (COCO格式)
    std::vector<float> keypoint_scores; // 关键点置信度
    cv::Point2f ground_position;        // 地面映射坐标 (如果启用Homography)
    bool has_ground_position;           // 是否有有效的地面坐标
};

class PoseDetector {
public:
    // 构造函数：指定模型路径
    explicit PoseDetector(const std::string& model_path);
    
    // 析构函数：自动清理资源
    ~PoseDetector();
    
    // 禁止拷贝和赋值
    PoseDetector(const PoseDetector&) = delete;
    PoseDetector& operator=(const PoseDetector&) = delete;
    
    // 核心检测接口：输入图像，输出检测结果
    std::vector<PoseResult> detect(const cv::Mat& frame);
    
    // 可选配置接口
    void enable_tracking(bool enable = true);
    bool load_calibration(const std::string& calibration_file);
    void set_confidence_threshold(float threshold);
    
    // 状态查询
    bool is_initialized() const;
    
    // 手动释放资源（析构函数会自动调用）
    void destroy();

private:
    // 前向声明，使用Pimpl模式隐藏实现细节
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

#endif // _POSE_DETECTOR_H_