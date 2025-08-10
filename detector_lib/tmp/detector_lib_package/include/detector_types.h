#ifndef DETECTOR_TYPES_H
#define DETECTOR_TYPES_H

#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace detector {

// 极坐标结构体
struct PolarCoordinate {
    double r;      // 距离（半径，单位：mm）
    double theta;  // 角度（弧度，-π到π）
    
    // 角度转换辅助函数
    double theta_degrees() const { return theta * 180.0 / M_PI; }
};

// 前向声明避免头文件依赖
struct PoseResult {
    int person_id = -1;                     // ByteTrack分配的人员ID
    float confidence = 0.0f;                // 检测置信度 [0-1]
    cv::Rect bbox;                          // 边界框 (x, y, width, height)
    std::vector<cv::Point2f> keypoints;     // 17个COCO关键点坐标
    std::vector<float> keypoint_scores;     // 关键点置信度 [0-1]
    cv::Point2f ground_position;            // Homography映射的地面坐标 (笛卡尔坐标，单位mm)
    PolarCoordinate polar_position;         // 极坐标位置 (距离mm，角度弧度)
    bool has_ground_position = false;       // 是否有有效地面坐标
    bool has_polar_position = false;        // 是否有有效极坐标
};

struct RimBasketballResult {
    int class_id = -1;                      // 类别ID: 0=basketball, 1=rim
    std::string class_name;                 // 类别名称
    float confidence = 0.0f;                // 检测置信度 [0-1]
    cv::Rect bbox;                          // 边界框 (x, y, width, height)
    cv::Point2f center;                     // 中心点
    float distance_to_rim = 0.0f;           // 篮球到篮筐的距离 (仅对basketball有效)
    bool is_in_rim_roi = false;             // 是否在篮筐ROI区域内
};

// COCO 17关键点索引枚举
enum COCOKeypoints {
    NOSE = 0,           // 鼻子
    LEFT_EYE = 1,       // 左眼
    RIGHT_EYE = 2,      // 右眼
    LEFT_EAR = 3,       // 左耳
    RIGHT_EAR = 4,      // 右耳
    LEFT_SHOULDER = 5,  // 左肩
    RIGHT_SHOULDER = 6, // 右肩
    LEFT_ELBOW = 7,     // 左肘
    RIGHT_ELBOW = 8,    // 右肘
    LEFT_WRIST = 9,     // 左腕
    RIGHT_WRIST = 10,   // 右腕
    LEFT_HIP = 11,      // 左髋
    RIGHT_HIP = 12,     // 右髋
    LEFT_KNEE = 13,     // 左膝
    RIGHT_KNEE = 14,    // 右膝
    LEFT_ANKLE = 15,    // 左踝
    RIGHT_ANKLE = 16    // 右踝
};

// 检测器状态枚举
enum DetectorStatus {
    DETECTOR_UNINITIALIZED = 0,
    DETECTOR_INITIALIZING = 1,
    DETECTOR_READY = 2,
    DETECTOR_ERROR = -1
};

} // namespace detector

#endif // DETECTOR_TYPES_H