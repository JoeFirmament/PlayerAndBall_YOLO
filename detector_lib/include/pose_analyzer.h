#pragma once

#include "pose_analysis_types.h"
#include "height_detector.h"
#include "ball_request_detector.h"
#include "id_priority_manager.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace pose_analysis {

class PoseAnalyzer {
private:
    // 三个核心模块
    std::unique_ptr<HeightDetector> height_detector_;
    std::unique_ptr<BallRequestDetector> ball_request_detector_;
    std::unique_ptr<IDPriorityManager> id_priority_manager_;
    
    // 配置
    PoseAnalysisConfig config_;
    
    // 状态
    bool initialized_;
    bool enabled_;
    
    // 统计信息
    int processed_frames_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    // 构造函数
    explicit PoseAnalyzer(const PoseAnalysisConfig& config);
    explicit PoseAnalyzer(const std::string& config_file_path);
    ~PoseAnalyzer() = default;
    
    // 初始化和配置
    bool initialize();
    void set_homography(const cv::Mat& homography_matrix);
    void update_config(const PoseAnalysisConfig& config);
    void update_config_from_file(const std::string& config_file_path);
    
    // 主要分析接口
    std::vector<PoseAnalysisResult> analyze(const std::vector<PoseResult>& pose_results);
    
    // 批量处理接口
    std::vector<std::vector<PoseAnalysisResult>> analyze_batch(
        const std::vector<std::vector<PoseResult>>& batch_pose_results);
    
    // 单个模块访问接口
    std::vector<HeightResult> analyze_height_only(const std::vector<PoseResult>& pose_results);
    std::vector<BallRequestResult> analyze_ball_request_only(const std::vector<PoseResult>& pose_results);
    std::vector<IDPriorityResult> analyze_id_priority_only(const std::vector<BallRequestResult>& ball_requests);
    
    // 模块开关控制
    void enable_height_detection(bool enable);
    void enable_ball_request_detection(bool enable);
    void enable_id_priority_management(bool enable);
    
    // 状态控制
    void enable(bool enable) { enabled_ = enable; }
    bool is_enabled() const { return enabled_; }
    void reset_all();
    void reset_person(int person_id);
    
    // 获取配置
    const PoseAnalysisConfig& get_config() const { return config_; }
    
    // 获取模块实例 (用于高级控制)
    HeightDetector* get_height_detector() const { return height_detector_.get(); }
    BallRequestDetector* get_ball_request_detector() const { return ball_request_detector_.get(); }
    IDPriorityManager* get_id_priority_manager() const { return id_priority_manager_.get(); }
    
    // 调试和监控
    std::string get_debug_info() const;
    std::string get_performance_stats() const;
    
    // 配置文件管理
    bool save_config_to_file(const std::string& file_path) const;
    static PoseAnalysisConfig load_config_from_file(const std::string& file_path);
    
    // 清理过期数据
    void cleanup_stale_data(int max_age_ms = 5000);

private:
    // 内部辅助函数
    void initialize_modules();
    PoseAnalysisResult merge_results(int person_id,
                                   const HeightResult& height_result,
                                   const BallRequestResult& ball_request_result,
                                   const IDPriorityResult& id_priority_result) const;
    
    // 结果验证和后处理
    std::vector<PoseAnalysisResult> post_process_results(
        const std::vector<HeightResult>& height_results,
        const std::vector<BallRequestResult>& ball_request_results,
        const std::vector<IDPriorityResult>& id_priority_results) const;
    
    // 配置验证
    bool validate_config(const PoseAnalysisConfig& config) const;
    
    // 默认配置生成
    static PoseAnalysisConfig create_default_config();
    
    // JSON配置解析
    bool parse_config_from_json(const std::string& json_content, PoseAnalysisConfig& config) const;
    std::string serialize_config_to_json(const PoseAnalysisConfig& config) const;
};

// 便利函数
class PoseAnalyzerBuilder {
private:
    PoseAnalysisConfig config_;
    
public:
    PoseAnalyzerBuilder();
    
    // 链式配置方法
    PoseAnalyzerBuilder& with_height_detection(bool enable = true);
    PoseAnalyzerBuilder& with_ball_request_detection(bool enable = true);
    PoseAnalyzerBuilder& with_id_priority_management(bool enable = true);
    
    PoseAnalyzerBuilder& height_filter_type(const std::string& filter_type);
    PoseAnalyzerBuilder& height_window_size(int window_size);
    PoseAnalyzerBuilder& height_stability_threshold(float threshold_mm);
    
    PoseAnalyzerBuilder& ball_request_min_frames(int min_frames);
    PoseAnalyzerBuilder& ball_request_max_interruption(int max_interruption);
    PoseAnalyzerBuilder& ball_request_confidence_threshold(float threshold);
    
    PoseAnalyzerBuilder& id_priority_weights(float confidence_weight, float duration_weight, float stability_weight);
    PoseAnalyzerBuilder& id_swap_cooldown(int cooldown_ms);
    
    PoseAnalyzerBuilder& global_frame_buffer_size(int buffer_size);
    PoseAnalyzerBuilder& global_time_window(int time_window_ms);
    
    PoseAnalyzerBuilder& debug_enable_all(bool enable = true);
    PoseAnalyzerBuilder& debug_output_path(const std::string& path);
    
    // 从文件加载配置
    PoseAnalyzerBuilder& from_config_file(const std::string& file_path);
    
    // 构建分析器
    std::unique_ptr<PoseAnalyzer> build();
    
    // 获取配置
    const PoseAnalysisConfig& get_config() const { return config_; }
};

// 工厂函数
std::unique_ptr<PoseAnalyzer> create_pose_analyzer(const std::string& config_file_path);
std::unique_ptr<PoseAnalyzer> create_default_pose_analyzer();

} // namespace pose_analysis