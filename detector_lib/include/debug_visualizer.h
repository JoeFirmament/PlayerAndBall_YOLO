#pragma once

#include "pose_analysis_types.h"
#include "temporal_buffer.h"
#include "filter_interface.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

namespace pose_analysis {

// 调试可视化器
class DebugVisualizer {
private:
    struct VisualizationConfig {
        // 颜色配置
        cv::Scalar height_color = cv::Scalar(0, 255, 0);           // 绿色 - 身高检测
        cv::Scalar ball_request_color = cv::Scalar(0, 0, 255);     // 红色 - 要球动作
        cv::Scalar priority_color = cv::Scalar(255, 165, 0);       // 橙色 - 优先级
        cv::Scalar stable_color = cv::Scalar(0, 255, 255);         // 黄色 - 稳定状态
        cv::Scalar text_color = cv::Scalar(255, 255, 255);         // 白色 - 文本
        cv::Scalar background_color = cv::Scalar(0, 0, 0);         // 黑色 - 背景
        
        // 绘制参数
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.5;
        int thickness = 1;
        int line_thickness = 2;
        
        // 图表参数
        int chart_width = 300;
        int chart_height = 200;
        int chart_margin = 10;
        bool show_grid = true;
    } config_;
    
    // 图表数据缓冲区
    std::map<int, TemporalBuffer<float>> height_history_;
    std::map<int, TemporalBuffer<float>> confidence_history_;
    std::map<int, TemporalBuffer<bool>> request_history_;
    
    // 统计计数器
    struct Statistics {
        int total_frames = 0;
        int stable_height_frames = 0;
        int confirmed_request_frames = 0;
        int id_swap_count = 0;
        std::chrono::steady_clock::time_point start_time;
        
        Statistics() : start_time(std::chrono::steady_clock::now()) {}
    } stats_;

public:
    DebugVisualizer();
    ~DebugVisualizer() = default;
    
    // 主要绘制接口
    void draw_analysis_results(cv::Mat& frame, 
                             const std::vector<PoseAnalysisResult>& results);
    
    // 分模块绘制函数
    void draw_height_info(cv::Mat& frame, const HeightResult& height_result, 
                         const cv::Rect& person_bbox);
    
    void draw_ball_request_info(cv::Mat& frame, const BallRequestResult& request_result,
                               const cv::Rect& person_bbox);
    
    void draw_id_priority_info(cv::Mat& frame, const IDPriorityResult& id_result,
                              const cv::Rect& person_bbox);
    
    // 状态机可视化
    void draw_state_machine(cv::Mat& frame, int person_id,
                           HeightDetectionState height_state,
                           BallRequestState request_state,
                           const cv::Point& position);
    
    // 图表绘制
    void draw_height_chart(cv::Mat& frame, int person_id, 
                          const std::vector<float>& height_data,
                          const cv::Rect& chart_area);
    
    void draw_confidence_chart(cv::Mat& frame, int person_id,
                              const std::vector<float>& confidence_data,
                              const cv::Rect& chart_area);
    
    void draw_request_timeline(cv::Mat& frame, int person_id,
                              const std::vector<bool>& request_data,
                              const cv::Rect& chart_area);
    
    // 滤波器状态可视化
    void draw_filter_comparison(cv::Mat& frame, 
                               const std::vector<IFilter*>& filters,
                               const cv::Rect& display_area);
    
    // 时序缓冲区可视化
    template<typename T>
    void draw_temporal_buffer(cv::Mat& frame, 
                             const TemporalBuffer<T>& buffer,
                             const cv::Rect& display_area,
                             const std::string& title);
    
    // 统计信息面板
    void draw_statistics_panel(cv::Mat& frame, const cv::Rect& panel_area);
    
    // 性能指标显示
    void draw_performance_metrics(cv::Mat& frame, 
                                 float fps, 
                                 float avg_processing_time_ms,
                                 const cv::Point& position);
    
    // 配置控制
    void set_visualization_config(const VisualizationConfig& config) { config_ = config; }
    const VisualizationConfig& get_config() const { return config_; }
    
    // 数据更新接口
    void update_height_data(int person_id, float height_mm);
    void update_confidence_data(int person_id, float confidence);
    void update_request_data(int person_id, bool is_requesting);
    void increment_id_swap_count() { stats_.id_swap_count++; }
    
    // 重置统计
    void reset_statistics();
    
    // 获取统计信息
    std::string get_statistics_text() const;

private:
    // 辅助绘制函数
    void draw_text_with_background(cv::Mat& frame, const std::string& text,
                                  const cv::Point& position, 
                                  const cv::Scalar& text_color = cv::Scalar(255, 255, 255),
                                  const cv::Scalar& bg_color = cv::Scalar(0, 0, 0, 128));
    
    void draw_progress_bar(cv::Mat& frame, float progress, 
                          const cv::Rect& bar_area,
                          const cv::Scalar& color);
    
    void draw_grid(cv::Mat& frame, const cv::Rect& area, 
                   int grid_size = 20,
                   const cv::Scalar& color = cv::Scalar(128, 128, 128));
    
    cv::Rect calculate_text_size(const std::string& text) const;
    
    // 颜色工具函数
    cv::Scalar get_state_color(HeightDetectionState state) const;
    cv::Scalar get_state_color(BallRequestState state) const;
    cv::Scalar interpolate_color(const cv::Scalar& color1, const cv::Scalar& color2, float ratio) const;
    
    // 数据格式化
    std::string format_height(float height_mm) const;
    std::string format_confidence(float confidence) const;
    std::string format_duration(int duration_ms) const;
    std::string format_state_name(HeightDetectionState state) const;
    std::string format_state_name(BallRequestState state) const;
};

// 数据记录器 - 用于离线分析
class DataRecorder {
private:
    struct FrameRecord {
        int frame_id;
        std::chrono::steady_clock::time_point timestamp;
        std::vector<PoseAnalysisResult> results;
        cv::Mat frame_image;  // 可选保存图像
        
        FrameRecord(int id) : frame_id(id), timestamp(std::chrono::steady_clock::now()) {}
    };
    
    std::vector<FrameRecord> recorded_frames_;
    std::string output_directory_;
    bool save_images_;
    bool is_recording_;
    int frame_counter_;

public:
    explicit DataRecorder(const std::string& output_dir = "./debug_records/", 
                         bool save_images = false);
    ~DataRecorder();
    
    // 记录控制
    void start_recording();
    void stop_recording();
    void pause_recording();
    bool is_recording() const { return is_recording_; }
    
    // 数据记录
    void record_frame(const std::vector<PoseAnalysisResult>& results,
                     const cv::Mat& frame = cv::Mat());
    
    void record_event(const std::string& event_name, 
                     const std::string& description,
                     int person_id = -1);
    
    // 数据保存
    bool save_session(const std::string& session_name = "") const;
    bool save_csv_report(const std::string& filename) const;
    bool save_json_report(const std::string& filename) const;
    
    // 数据加载和回放
    bool load_session(const std::string& session_file);
    std::vector<FrameRecord> get_recorded_frames() const { return recorded_frames_; }
    
    // 统计分析
    void generate_analysis_report(const std::string& report_file) const;
    std::map<int, float> calculate_average_heights() const;
    std::map<int, float> calculate_request_frequencies() const;
    std::map<int, int> calculate_id_swap_counts() const;
    
    // 清理
    void clear_records();
    void set_output_directory(const std::string& dir) { output_directory_ = dir; }
    
    // 获取统计信息
    size_t get_recorded_frame_count() const { return recorded_frames_.size(); }
    std::chrono::milliseconds get_recording_duration() const;

private:
    void ensure_output_directory() const;
    std::string generate_timestamp_string() const;
};

// 实时性能监控器
class PerformanceMonitor {
private:
    struct PerformanceMetrics {
        TemporalBuffer<float> frame_times;
        TemporalBuffer<float> processing_times;
        TemporalBuffer<int> detection_counts;
        
        float current_fps = 0.0f;
        float avg_processing_time_ms = 0.0f;
        float peak_processing_time_ms = 0.0f;
        int total_detections = 0;
        
        std::chrono::steady_clock::time_point last_frame_time;
        
        PerformanceMetrics() : frame_times(60), processing_times(60), detection_counts(60),
                              last_frame_time(std::chrono::steady_clock::now()) {}
    } metrics_;
    
    bool monitoring_enabled_;

public:
    PerformanceMonitor();
    
    // 监控控制
    void enable_monitoring(bool enable = true) { monitoring_enabled_ = enable; }
    bool is_monitoring() const { return monitoring_enabled_; }
    
    // 性能数据记录
    void record_frame_start();
    void record_frame_end(int detection_count = 0);
    void record_processing_time(float processing_time_ms);
    
    // 性能指标获取
    float get_current_fps() const { return metrics_.current_fps; }
    float get_average_processing_time() const { return metrics_.avg_processing_time_ms; }
    float get_peak_processing_time() const { return metrics_.peak_processing_time_ms; }
    int get_total_detections() const { return metrics_.total_detections; }
    
    // 性能报告
    std::string get_performance_summary() const;
    void print_performance_report() const;
    
    // 重置统计
    void reset_metrics();
    
    // 获取历史数据
    std::vector<float> get_fps_history(int num_frames = 30) const;
    std::vector<float> get_processing_time_history(int num_frames = 30) const;

private:
    void update_metrics();
};

// 配置管理器 - 用于调试时动态调整参数
class DebugConfigManager {
private:
    PoseAnalysisConfig current_config_;
    std::map<std::string, std::function<void(float)>> parameter_setters_;
    std::map<std::string, std::function<float()>> parameter_getters_;
    
public:
    explicit DebugConfigManager(const PoseAnalysisConfig& initial_config);
    
    // 参数动态调整接口
    void register_parameter(const std::string& name,
                           std::function<void(float)> setter,
                           std::function<float()> getter);
    
    bool set_parameter(const std::string& name, float value);
    float get_parameter(const std::string& name) const;
    std::vector<std::string> get_parameter_names() const;
    
    // 配置管理
    const PoseAnalysisConfig& get_config() const { return current_config_; }
    void update_config(const PoseAnalysisConfig& new_config);
    
    // 预设配置
    void load_preset_config(const std::string& preset_name);
    void save_current_as_preset(const std::string& preset_name);
    std::vector<std::string> get_available_presets() const;
    
    // 配置验证
    bool validate_config() const;
    std::vector<std::string> get_validation_errors() const;

private:
    void setup_default_parameters();
    void update_internal_config();
};

} // namespace pose_analysis