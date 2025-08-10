#ifndef POSE_DETECTOR_LIB_H
#define POSE_DETECTOR_LIB_H

#include "detector_types.h"
#include <memory>

namespace detector {

/**
 * @brief 姿态检测器类 - 对外接口
 * 
 * 封装了YOLOv8姿态检测、ByteTrack跟踪、Homography坐标映射等功能
 * 使用延迟初始化策略，构造函数不会抛出异常
 * 支持零拷贝NPU优化，保持高性能
 */
class PoseDetectorLib {
public:
    /**
     * @brief 构造函数
     * @param model_path RKNN模型文件路径 (绝对路径或相对路径)
     * @note 构造函数只保存参数，不进行资源初始化，不会抛出异常
     */
    explicit PoseDetectorLib(const std::string& model_path);
    
    /**
     * @brief 析构函数
     * @note 自动释放所有NPU资源，用户无需手动清理
     */
    ~PoseDetectorLib();
    
    // 禁止拷贝和赋值
    PoseDetectorLib(const PoseDetectorLib&) = delete;
    PoseDetectorLib& operator=(const PoseDetectorLib&) = delete;
    PoseDetectorLib(PoseDetectorLib&&) = delete;
    PoseDetectorLib& operator=(PoseDetectorLib&&) = delete;
    
    /**
     * @brief 核心检测接口
     * @param frame 输入图像 (BGR格式，推荐1280x720，支持任意分辨率)
     * @return 检测结果数组，失败时返回空vector
     * @note 首次调用时会自动初始化(1-3秒)，建议先预热
     * @note 坐标映射功能需要使用与标定文件相同的分辨率
     * 
     * 返回的PoseResult包含：
     * - person_id: ByteTrack跟踪ID (启用跟踪时>0，否则为-1)
     * - confidence: 检测置信度 [0.0-1.0]
     * - bbox: ROI边界框 cv::Rect(x, y, width, height)
     * - keypoints: 17个COCO关键点坐标 std::vector<cv::Point2f>
     * - keypoint_scores: 17个关键点置信度 std::vector<float> [0.0-1.0]
     * - ground_position: 笛卡尔地面坐标 cv::Point2f(x_mm, y_mm) (启用坐标映射时有效)
     * - polar_position: 极坐标地面坐标 PolarCoordinate(r_mm, theta_rad) (启用极坐标时有效)
     * - has_ground_position: 是否有有效笛卡尔地面坐标 bool
     * - has_polar_position: 是否有有效极坐标 bool
     */
    std::vector<PoseResult> detect(const cv::Mat& frame);
    
    /**
     * @brief 启用/禁用ByteTrack跟踪
     * @param enable true=启用，false=禁用
     * @note 启用后person_id会保持连续，禁用时person_id为-1
     */
    void enable_tracking(bool enable = true);
    
    /**
     * @brief 加载Homography标定文件
     * @param calibration_file 标定文件路径 (JSON格式)
     * @return true=加载成功，false=加载失败
     * @note 成功加载后，PoseResult中会包含ground_position坐标
     * @note 如果JSON中有极坐标配置，会自动启用极坐标功能
     */
    bool load_calibration(const std::string& calibration_file);
    
    /**
     * @brief 设置极坐标系统配置
     * @param enable 是否启用极坐标计算
     * @param origin_offset_x 极坐标原点X偏移量 (mm)
     * @param origin_offset_y 极坐标原点Y偏移量 (mm)
     * @note 启用后PoseResult中会包含polar_position坐标
     */
    void set_polar_coordinate_system(bool enable, float origin_offset_x = 0.0f, float origin_offset_y = 0.0f);
    
    /**
     * @brief 设置检测置信度阈值
     * @param threshold 置信度阈值 [0.01-0.99]
     * @note 默认值为0.25，阈值越高检测越严格
     */
    void set_confidence_threshold(float threshold);
    
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

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace detector

#endif // POSE_DETECTOR_LIB_H