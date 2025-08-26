#pragma once

#include "pose_analysis_types.h"
#include "temporal_buffer.h"
#include "filter_interface.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <unordered_map>

namespace pose_analysis {

class HeightDetector {
private:
    // 每个person的状态上下文
    struct PersonHeightContext {
        int person_id;
        HeightDetectionState state;
        
        // 测量数据缓冲区
        TemporalBuffer<HeightMeasurement> measurements;
        TimestampedBuffer<float> raw_heights;
        
        // 滤波器
        std::unique_ptr<IFilter> height_filter;
        std::unique_ptr<IFilter> confidence_filter;
        
        // 状态计数器
        int stable_frames_count;
        int measuring_frames_count;
        int invalid_frames_count;
        
        // 时间戳
        std::chrono::steady_clock::time_point measurement_start_time;
        std::chrono::steady_clock::time_point last_valid_measurement;
        
        // 当前稳定结果
        float stable_height_mm;
        float stable_confidence;
        
        PersonHeightContext(int id, const HeightDetectionConfig& config)
            : person_id(id), state(HeightDetectionState::IDLE),
              measurements(config.measurement_window_frames),
              raw_heights(config.measurement_window_frames, config.max_measurement_gap_ms),
              stable_frames_count(0), measuring_frames_count(0), invalid_frames_count(0),
              measurement_start_time(std::chrono::steady_clock::now()),
              last_valid_measurement(std::chrono::steady_clock::now()),
              stable_height_mm(-1.0f), stable_confidence(0.0f) {
            
            // 创建滤波器
            height_filter = FilterFactory::create_filter(config.filter_type, 
                                                       config.window_size,
                                                       config.kalman_process_noise,
                                                       config.kalman_measurement_noise);
            confidence_filter = FilterFactory::create_moving_average_filter(config.window_size, 0.8f);
        }
    };
    
    // 配置参数
    HeightDetectionConfig config_;
    
    // 每个person的状态管理
    std::unordered_map<int, std::unique_ptr<PersonHeightContext>> person_contexts_;
    
    // Homography矩阵用于坐标转换
    cv::Mat homography_matrix_;
    bool has_homography_;
    
    // 帧计数器
    int frame_counter_;
    
public:
    explicit HeightDetector(const HeightDetectionConfig& config);
    ~HeightDetector() = default;
    
    // 设置Homography矩阵
    void set_homography(const cv::Mat& homography);
    
    // 处理单帧姿态结果
    std::vector<HeightResult> process_frame(const std::vector<PoseResult>& pose_results);
    
    // 处理单个person
    HeightResult process_person(const PoseResult& pose);
    
    // 获取配置
    const HeightDetectionConfig& get_config() const { return config_; }
    
    // 更新配置
    void update_config(const HeightDetectionConfig& config);
    
    // 重置所有状态
    void reset();
    
    // 重置特定person的状态
    void reset_person(int person_id);
    
    // 清理过期的person上下文
    void cleanup_stale_contexts(int max_age_ms = 5000);
    
    // 获取调试信息
    std::string get_debug_info(int person_id = -1) const;
    
    // 获取所有活跃的person ID
    std::vector<int> get_active_person_ids() const;

private:
    // 检查是否可以测量身高 (双手不超过头部)
    bool can_measure_height(const PoseResult& pose) const;
    
    // 计算原始身高 (像素到毫米转换)
    float calculate_raw_height(const PoseResult& pose) const;
    
    // 计算测量置信度
    float calculate_measurement_confidence(const PoseResult& pose) const;
    
    // 异常值检测
    bool is_outlier(float height, const PersonHeightContext& context) const;
    
    // 计算身高测量的稳定性指标
    float calculate_stability(const PersonHeightContext& context) const;
    
    // 更新person的状态机
    void update_state_machine(PersonHeightContext& context, const HeightMeasurement& measurement);
    
    // 状态转换逻辑
    void transition_to_measuring(PersonHeightContext& context);
    void transition_to_stable(PersonHeightContext& context);
    void transition_to_invalid(PersonHeightContext& context);
    void transition_to_idle(PersonHeightContext& context);
    
    // 计算最终的稳定身高值
    float calculate_final_height(const PersonHeightContext& context) const;
    
    // 获取或创建person上下文
    PersonHeightContext& get_or_create_context(int person_id);
    
    // 像素坐标转换为世界坐标 (使用Homography)
    cv::Point2f pixel_to_world(const cv::Point2f& pixel_point) const;
    
    // 计算两个世界坐标点之间的距离 (毫米)
    float calculate_world_distance(const cv::Point2f& p1, const cv::Point2f& p2) const;
    
    // 验证关键点的有效性
    bool is_keypoint_valid(const PoseResult& pose, COCOKeypoint keypoint) const;
    
    // 获取关键点的世界坐标
    cv::Point2f get_keypoint_world_pos(const PoseResult& pose, COCOKeypoint keypoint) const;
};

} // namespace pose_analysis