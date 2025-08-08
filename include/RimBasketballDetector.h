#ifndef _RIM_BASKETBALL_DETECTOR_H_
#define _RIM_BASKETBALL_DETECTOR_H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

// 篮筐篮球检测结果结构体
struct RimBasketballResult {
    int class_id;                    // 类别ID: 0=basketball, 1=rim
    std::string class_name;          // 类别名称
    float confidence;                // 置信度
    cv::Rect bbox;                   // 边界框
    cv::Point2f center;              // 中心点
    float distance_to_rim;           // 篮球到篮筐的距离 (仅对basketball有效)
    bool is_in_rim_roi;             // 是否在篮筐ROI区域内
};

class RimBasketballDetector {
public:
    // 构造函数：指定模型路径
    explicit RimBasketballDetector(const std::string& model_path);
    
    // 析构函数：自动清理资源
    ~RimBasketballDetector();
    
    // 禁止拷贝和赋值
    RimBasketballDetector(const RimBasketballDetector&) = delete;
    RimBasketballDetector& operator=(const RimBasketballDetector&) = delete;
    
    // 核心检测接口：输入图像，输出检测结果
    std::vector<RimBasketballResult> detect(const cv::Mat& frame);
    
    // 可选配置接口
    void set_confidence_threshold(float threshold);
    void set_nms_threshold(float threshold);
    
    // 状态查询
    bool is_initialized() const;
    
    // 手动释放资源（析构函数会自动调用）
    void destroy();

private:
    // 前向声明，使用Pimpl模式隐藏实现细节
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

#endif // _RIM_BASKETBALL_DETECTOR_H_