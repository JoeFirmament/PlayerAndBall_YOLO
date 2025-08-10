#ifndef RIM_BASKETBALL_DETECTOR_LIB_H
#define RIM_BASKETBALL_DETECTOR_LIB_H

#include "detector_types.h"
#include <memory>

namespace detector {

/**
 * @brief 篮筐篮球检测器类 - 对外接口
 * 
 * 封装了YOLOv8目标检测，支持篮筐(rim)和篮球(basketball)检测
 * 内置ROI分析和距离计算功能
 * 使用延迟初始化策略，构造函数不会抛出异常
 * 支持零拷贝NPU优化，保持高性能
 */
class RimBasketballDetectorLib {
public:
    /**
     * @brief 构造函数
     * @param model_path RKNN模型文件路径 (绝对路径或相对路径)
     * @note 构造函数只保存参数，不进行资源初始化，不会抛出异常
     */
    explicit RimBasketballDetectorLib(const std::string& model_path);
    
    /**
     * @brief 析构函数
     * @note 自动释放所有NPU资源，用户无需手动清理
     */
    ~RimBasketballDetectorLib();
    
    // 禁止拷贝和赋值
    RimBasketballDetectorLib(const RimBasketballDetectorLib&) = delete;
    RimBasketballDetectorLib& operator=(const RimBasketballDetectorLib&) = delete;
    RimBasketballDetectorLib(RimBasketballDetectorLib&&) = delete;
    RimBasketballDetectorLib& operator=(RimBasketballDetectorLib&&) = delete;
    
    /**
     * @brief 核心检测接口
     * @param frame 输入图像 (BGR格式，推荐1280x960，支持任意分辨率)
     * @return 检测结果数组，可能包含篮筐和篮球，失败时返回空vector
     * @note 首次调用时会自动初始化(1-3秒)，建议先预热
     * 
     * 返回的RimBasketballResult包含：
     * - class_id: 类别ID (0=basketball, 1=rim)
     * - class_name: 类别名称 ("basketball" 或 "rim")
     * - confidence: 检测置信度 [0.0-1.0]
     * - bbox: 边界框 cv::Rect(x, y, width, height)
     * - center: 目标中心点 cv::Point2f(x, y)
     * - distance_to_rim: 篮球到篮筐距离(像素) (仅basketball有效)
     * - is_in_rim_roi: 是否在篮筐ROI区域内 bool
     */
    std::vector<RimBasketballResult> detect(const cv::Mat& frame);
    
    /**
     * @brief 设置检测置信度阈值
     * @param threshold 置信度阈值 [0.01-0.99]
     * @note 默认值为0.4，篮球检测通常需要较高置信度
     */
    void set_confidence_threshold(float threshold);
    
    /**
     * @brief 设置NMS(非极大值抑制)阈值
     * @param threshold NMS阈值 [0.01-0.99]
     * @note 默认值为0.45，控制重叠检测框的过滤程度
     */
    void set_nms_threshold(float threshold);
    
    /**
     * @brief 查询检测器初始化状态
     * @return true=已初始化，false=未初始化或初始化失败
     */
    bool is_initialized() const;
    
    /**
     * @brief 获取检测器状态
     * @return DetectorStatus 状态枚举
     */
    DetectorStatus get_status() const;
    
    /**
     * @brief 手动释放资源
     * @note 析构函数会自动调用，一般情况下用户无需手动调用
     */
    void release();
    
    /**
     * @brief 获取最近一次的推理时间 (毫秒)
     * @return 推理时间，-1表示无有效数据
     */
    int get_last_inference_time_ms() const;
    
    /**
     * @brief 获取支持的类别列表
     * @return 类别名称数组
     */
    static std::vector<std::string> get_supported_classes();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace detector

#endif // RIM_BASKETBALL_DETECTOR_LIB_H