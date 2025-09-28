#pragma once

#include "pose_analysis_types.h"
#include "temporal_buffer.h"
#include "filter_interface.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <unordered_map>

namespace pose_analysis {

class BallRequestDetector {
private:
    // 要球动作的上下文信息
    struct RequestContext {
        int person_id;
        BallRequestState state;
        
        // 帧计数器
        int continuous_frames;          // 连续检测帧数
        int total_frames;              // 总检测帧数
        int interruption_frames;        // 中断帧数
        int cooldown_frames;           // 冷却期帧数
        
        // 置信度累积
        float accumulated_confidence;   // 累积置信度
        float max_confidence;          // 最高置信度
        
        // 时序数据缓冲
        TemporalBuffer<float> hands_distance_buffer;    // 双手距离历史
        TemporalBuffer<cv::Point2f> left_wrist_buffer;  // 左手腕位置历史
        TemporalBuffer<cv::Point2f> right_wrist_buffer; // 右手腕位置历史
        TemporalBuffer<float> confidence_buffer;        // 置信度历史
        
        // 滤波器
        std::unique_ptr<IFilter> confidence_filter;
        std::unique_ptr<IFilter> position_filter;
        
        // 时间戳
        std::chrono::steady_clock::time_point request_start_time;
        std::chrono::steady_clock::time_point last_detected_time;
        std::chrono::steady_clock::time_point last_confirmed_time;
        
        RequestContext(int id, const BallRequestConfig& config)
            : person_id(id), state(BallRequestState::NO_REQUEST),
              continuous_frames(0), total_frames(0), interruption_frames(0), cooldown_frames(0),
              accumulated_confidence(0.0f), max_confidence(0.0f),
              hands_distance_buffer(config.detection_window_frames),
              left_wrist_buffer(config.detection_window_frames),
              right_wrist_buffer(config.detection_window_frames),
              confidence_buffer(config.detection_window_frames),
              request_start_time(std::chrono::steady_clock::now()),
              last_detected_time(std::chrono::steady_clock::now()),
              last_confirmed_time(std::chrono::steady_clock::now()) {
            
            // 创建滤波器
            if (config.smooth_confidence) {
                confidence_filter = FilterFactory::create_moving_average_filter(
                    5, config.confidence_smoothing_alpha);
            }
            
            position_filter = FilterFactory::create_moving_average_filter(
                3, config.position_smoothing_alpha);
        }
    };
    
    // 配置参数
    BallRequestConfig config_;
    
    // 每个person的请求上下文
    std::unordered_map<int, std::unique_ptr<RequestContext>> person_contexts_;
    
    // Homography矩阵用于坐标转换
    cv::Mat homography_matrix_;
    bool has_homography_;
    
    // 帧计数器
    int frame_counter_;
    
public:
    explicit BallRequestDetector(const BallRequestConfig& config);
    ~BallRequestDetector() = default;
    
    // 设置Homography矩阵
    void set_homography(const cv::Mat& homography);
    
    // 处理单帧姿态结果
    std::vector<BallRequestResult> process_frame(const std::vector<PoseResult>& pose_results);
    
    // 处理单个person
    BallRequestResult process_person(const PoseResult& pose);
    
    // 批量处理
    std::vector<BallRequestResult> process_batch(const std::vector<std::vector<PoseResult>>& batch_poses);
    
    // 获取配置
    const BallRequestConfig& get_config() const { return config_; }
    
    // 更新配置
    void update_config(const BallRequestConfig& config);
    
    // 重置所有状态
    void reset();
    
    // 重置特定person的状态
    void reset_person(int person_id);
    
    // 清理过期的person上下文
    void cleanup_stale_contexts(int max_age_ms = 3000);
    
    // 获取调试信息
    std::string get_debug_info(int person_id = -1) const;
    
    // 获取所有活跃的person ID
    std::vector<int> get_active_person_ids() const;
    
    // 获取当前正在要球的person ID列表
    std::vector<int> get_requesting_person_ids() const;

private:
    // 检测原始要球动作
    bool detect_raw_request(const PoseResult& pose, float& confidence) const;
    
    // 计算手势特征
    struct GestureFeatures {
        // 绝对距离（兼容性保留）
        float hands_distance_mm;        // 双手距离（毫米）
        float hands_to_chest_distance;  // 手到胸部距离（毫米）
        
        // 🔥 新增：ROI相对距离（距离无关）
        float hands_distance_ratio;     // 双手距离相对于人体宽度的比例
        float hands_to_chest_ratio;     // 手到胸部距离相对于人体宽度的比例
        
        // 位置信息
        cv::Point2f hands_center;       // 双手中心点
        cv::Point2f chest_center;       // 胸部中心点
        float hand_height_ratio;        // 手的高度比例（已是相对值）
        
        // 状态标记
        bool hands_in_front;           // 手是否在身体前方
        bool gesture_valid;            // 手势是否有效
        float feature_confidence;       // 特征置信度
    };
    
    GestureFeatures calculate_gesture_features(const PoseResult& pose) const;
    
    // 计算手势稳定性
    float calculate_gesture_stability(const RequestContext& context) const;
    
    // 计算动作置信度
    float calculate_request_confidence(const GestureFeatures& features) const;
    
    // 状态转换逻辑
    void update_state_machine(RequestContext& context, bool is_requesting, float confidence);
    
    // 具体的状态转换函数
    void transition_to_potential_request(RequestContext& context, float confidence);
    void transition_to_confirmed_request(RequestContext& context);
    void transition_to_ending_request(RequestContext& context);
    void transition_to_no_request(RequestContext& context);
    
    // 验证确认条件
    bool validate_confirmation_conditions(const RequestContext& context) const;
    
    // 清理过期状态
    void cleanup_stale_contexts();
    
    // 获取或创建person上下文
    RequestContext& get_or_create_context(int person_id);
    
    // 坐标转换工具
    cv::Point2f pixel_to_world(const cv::Point2f& pixel_point) const;
    float calculate_world_distance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    
    // 关键点验证
    bool is_keypoint_valid(const PoseResult& pose, COCOKeypoint keypoint) const;
    cv::Point2f get_keypoint_position(const PoseResult& pose, COCOKeypoint keypoint) const;
    
    // 几何计算工具
    bool is_point_in_front_of_chest(const cv::Point2f& hand_pos, 
                                   const cv::Point2f& chest_center,
                                   const PoseResult& pose) const;
    
    cv::Point2f calculate_chest_center(const PoseResult& pose) const;
    
    // 时序分析工具
    float calculate_confidence_trend(const RequestContext& context) const;
    float calculate_gesture_consistency(const RequestContext& context) const;
};

} // namespace pose_analysis