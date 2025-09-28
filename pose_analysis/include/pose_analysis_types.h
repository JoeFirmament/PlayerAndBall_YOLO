#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <map>

namespace pose_analysis {

// 基础姿态结果结构
struct PoseResult {
    int person_id;
    cv::Rect2f bbox;
    std::vector<cv::Point2f> keypoints;  // 17个关键点 (COCO格式)
    std::vector<float> keypoint_confidences;
    float detection_confidence;
    std::chrono::steady_clock::time_point timestamp;
    
    PoseResult() : person_id(-1), detection_confidence(0.0f), 
                   timestamp(std::chrono::steady_clock::now()) {
        keypoints.resize(17);
        keypoint_confidences.resize(17);
    }
};

// 身高测量结果
struct HeightMeasurement {
    float height_mm;
    float confidence;
    bool is_valid;
    int frame_id;
    std::chrono::steady_clock::time_point timestamp;
    
    HeightMeasurement() : height_mm(-1.0f), confidence(0.0f), 
                         is_valid(false), frame_id(-1),
                         timestamp(std::chrono::steady_clock::now()) {}
};

// 身高检测状态
enum class HeightDetectionState {
    IDLE,           // 空闲状态
    MEASURING,      // 测量中
    STABLE,         // 稳定输出
    INVALID         // 无效状态
};

// 身高检测结果
struct HeightResult {
    int person_id;
    float estimated_height_mm;
    float confidence;
    bool is_stable;
    HeightDetectionState state;
    int stable_frames_count;
    std::chrono::steady_clock::time_point measurement_start_time;
    
    HeightResult() : person_id(-1), estimated_height_mm(-1.0f), 
                    confidence(0.0f), is_stable(false), 
                    state(HeightDetectionState::IDLE), stable_frames_count(0),
                    measurement_start_time(std::chrono::steady_clock::now()) {}
};

// 要球动作状态
enum class BallRequestState {
    NO_REQUEST,         // 未要球
    POTENTIAL_REQUEST,  // 可能要球（检测到但未满足时间）
    CONFIRMED_REQUEST,  // 确认要球
    ENDING_REQUEST     // 结束要球（冷却期）
};

// 要球动作结果
struct BallRequestResult {
    int person_id;
    bool is_requesting;
    bool is_confirmed;
    float request_confidence;
    BallRequestState state;
    int continuous_frames;
    int total_frames;
    int interruption_frames;
    float accumulated_confidence;
    float gesture_stability;
    std::chrono::steady_clock::time_point request_start_time;
    int request_duration_ms;
    
    BallRequestResult() : person_id(-1), is_requesting(false), is_confirmed(false),
                         request_confidence(0.0f), state(BallRequestState::NO_REQUEST),
                         continuous_frames(0), total_frames(0), interruption_frames(0),
                         accumulated_confidence(0.0f), gesture_stability(1.0f),
                         request_start_time(std::chrono::steady_clock::now()),
                         request_duration_ms(0) {}
};

// ID优先级结果
struct IDPriorityResult {
    int person_id;
    int priority_id;  // 优先级排序后的ID
    float priority_score;
    bool can_swap_id;
    std::chrono::steady_clock::time_point last_swap_time;
    
    IDPriorityResult() : person_id(-1), priority_id(-1), priority_score(0.0f),
                        can_swap_id(true), 
                        last_swap_time(std::chrono::steady_clock::now()) {}
};

// 综合分析结果
struct PoseAnalysisResult {
    int person_id;
    
    // 身高检测结果
    HeightResult height_result;
    
    // 要球动作结果
    BallRequestResult ball_request_result;
    
    // ID优先级结果
    IDPriorityResult id_priority_result;
    
    // 整体状态
    bool analysis_valid;
    std::chrono::steady_clock::time_point analysis_timestamp;
    
    PoseAnalysisResult() : person_id(-1), analysis_valid(false),
                          analysis_timestamp(std::chrono::steady_clock::now()) {}
};

// COCO 17个关键点索引
enum class COCOKeypoint : int {
    NOSE = 0,
    LEFT_EYE = 1,
    RIGHT_EYE = 2,
    LEFT_EAR = 3,
    RIGHT_EAR = 4,
    LEFT_SHOULDER = 5,
    RIGHT_SHOULDER = 6,
    LEFT_ELBOW = 7,
    RIGHT_ELBOW = 8,
    LEFT_WRIST = 9,
    RIGHT_WRIST = 10,
    LEFT_HIP = 11,
    RIGHT_HIP = 12,
    LEFT_KNEE = 13,
    RIGHT_KNEE = 14,
    LEFT_ANKLE = 15,
    RIGHT_ANKLE = 16
};

// 配置参数结构
struct HeightDetectionConfig {
    // 基础检测参数
    float min_keypoint_confidence = 0.5f;
    int head_offset_pixels = 30;
    float height_correction_factor = 1.05f;
    int min_roi_height_pixels = 100;
    int max_roi_height_pixels = 800;
    
    // 滤波参数
    std::string filter_type = "median";  // "median", "kalman", "moving_average"
    int window_size = 15;
    int min_stable_frames = 10;
    float outlier_threshold_sigma = 3.0f;
    float stability_threshold_mm = 50.0f;
    float kalman_process_noise = 0.01f;
    float kalman_measurement_noise = 10.0f;
    
    // 时序参数
    int measurement_window_frames = 30;
    float min_confidence = 0.7f;
    int max_measurement_gap_ms = 500;
};

struct BallRequestConfig {
    // 手势检测参数
    float min_keypoint_confidence = 0.5f;
    
    // 🔥 距离参数（兼容性保留 + 新增相对参数）
    float max_hands_distance_mm = 400.0f;        // 绝对距离（兼容性保留）
    float max_hands_distance_ratio = 0.35f;      // 相对于人体宽度的比例阈值
    float chest_region_scale = 1.5f;             // 胸部区域缩放
    float max_chest_distance_ratio = 0.25f;      // 手到胸部最大距离比例
    
    // 高度参数（已是相对值）
    float min_hand_height_ratio = 0.3f;
    float max_hand_height_ratio = 0.7f;
    
    // 时序检测参数
    int min_continuous_frames = 5;
    int max_interruption_frames = 2;
    float min_total_confidence = 3.5f;
    int cooldown_frames = 10;
    float gesture_stability_threshold = 0.2f;
    int detection_window_frames = 20;
    
    // 滤波参数
    bool smooth_confidence = true;
    float confidence_smoothing_alpha = 0.7f;
    float position_smoothing_alpha = 0.8f;
};

struct IDManagementConfig {
    // 优先级权重
    float confidence_weight = 0.3f;
    float duration_weight = 0.4f;
    float stability_weight = 0.3f;
    float decay_rate = 0.95f;
    
    // 时序参数
    int min_request_frames_for_swap = 15;
    int swap_cooldown_ms = 2000;
    int max_pending_swaps = 5;
    int priority_update_interval_ms = 100;
    
    // 规则参数
    bool allow_multiple_requesters = false;
    bool preserve_original_order = true;
    int max_tracked_persons = 10;
};

struct GlobalConfig {
    bool enable_temporal_smoothing = true;
    bool enable_outlier_rejection = true;
    bool enable_predictive_tracking = false;
    int frame_buffer_size = 60;
    int time_window_ms = 2000;
};

struct DebugConfig {
    bool show_filter_states = true;
    bool show_temporal_buffers = true;
    bool show_state_machines = true;
    bool log_measurements = true;
    bool save_debug_video = false;
    std::string debug_output_path = "./debug/";
};

// 完整配置结构
struct PoseAnalysisConfig {
    HeightDetectionConfig height_detection;
    BallRequestConfig ball_request;
    IDManagementConfig id_management;
    GlobalConfig global;
    DebugConfig debug;
};

} // namespace pose_analysis