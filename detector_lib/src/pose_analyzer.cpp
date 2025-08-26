#include "pose_analyzer.h"
#include <fstream>
#include <sstream>
#include <json/json.h>
#include <algorithm>

namespace pose_analysis {

PoseAnalyzer::PoseAnalyzer(const PoseAnalysisConfig& config)
    : config_(config), initialized_(false), enabled_(true), processed_frames_(0),
      start_time_(std::chrono::steady_clock::now()) {
    
    if (validate_config(config)) {
        initialize_modules();
        initialized_ = true;
    }
}

PoseAnalyzer::PoseAnalyzer(const std::string& config_file_path)
    : initialized_(false), enabled_(true), processed_frames_(0),
      start_time_(std::chrono::steady_clock::now()) {
    
    config_ = load_config_from_file(config_file_path);
    
    if (validate_config(config_)) {
        initialize_modules();
        initialized_ = true;
    }
}

bool PoseAnalyzer::initialize() {
    if (initialized_) {
        return true;
    }
    
    if (!validate_config(config_)) {
        return false;
    }
    
    try {
        initialize_modules();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        initialized_ = false;
        return false;
    }
}

void PoseAnalyzer::initialize_modules() {
    // 创建身高检测模块
    height_detector_ = std::make_unique<HeightDetector>(config_.height_detection);
    
    // 创建要球动作检测模块
    ball_request_detector_ = std::make_unique<BallRequestDetector>(config_.ball_request);
    
    // 创建ID优先级管理模块
    id_priority_manager_ = std::make_unique<IDPriorityManager>(config_.id_management);
}

void PoseAnalyzer::set_homography(const cv::Mat& homography_matrix) {
    if (!initialized_) return;
    
    if (height_detector_) {
        height_detector_->set_homography(homography_matrix);
    }
    
    if (ball_request_detector_) {
        ball_request_detector_->set_homography(homography_matrix);
    }
}

std::vector<PoseAnalysisResult> PoseAnalyzer::analyze(const std::vector<PoseResult>& pose_results) {
    if (!initialized_ || !enabled_ || pose_results.empty()) {
        return {};
    }
    
    processed_frames_++;
    
    std::vector<HeightResult> height_results;
    std::vector<BallRequestResult> ball_request_results;
    std::vector<IDPriorityResult> id_priority_results;
    
    // 1. 身高检测
    if (height_detector_) {
        height_results = height_detector_->process_frame(pose_results);
    }
    
    // 2. 要球动作检测
    if (ball_request_detector_) {
        ball_request_results = ball_request_detector_->process_frame(pose_results);
    }
    
    // 3. ID优先级管理 (基于要球动作结果)
    if (id_priority_manager_ && !ball_request_results.empty()) {
        id_priority_results = id_priority_manager_->update(ball_request_results);
    }
    
    // 4. 合并结果
    return post_process_results(height_results, ball_request_results, id_priority_results);
}

std::vector<std::vector<PoseAnalysisResult>> PoseAnalyzer::analyze_batch(
    const std::vector<std::vector<PoseResult>>& batch_pose_results) {
    
    std::vector<std::vector<PoseAnalysisResult>> batch_results;
    batch_results.reserve(batch_pose_results.size());
    
    for (const auto& frame_poses : batch_pose_results) {
        batch_results.push_back(analyze(frame_poses));
    }
    
    return batch_results;
}

std::vector<HeightResult> PoseAnalyzer::analyze_height_only(const std::vector<PoseResult>& pose_results) {
    if (!initialized_ || !height_detector_) {
        return {};
    }
    
    return height_detector_->process_frame(pose_results);
}

std::vector<BallRequestResult> PoseAnalyzer::analyze_ball_request_only(const std::vector<PoseResult>& pose_results) {
    if (!initialized_ || !ball_request_detector_) {
        return {};
    }
    
    return ball_request_detector_->process_frame(pose_results);
}

std::vector<IDPriorityResult> PoseAnalyzer::analyze_id_priority_only(const std::vector<BallRequestResult>& ball_requests) {
    if (!initialized_ || !id_priority_manager_) {
        return {};
    }
    
    return id_priority_manager_->update(ball_requests);
}

PoseAnalysisResult PoseAnalyzer::merge_results(int person_id,
                                             const HeightResult& height_result,
                                             const BallRequestResult& ball_request_result,
                                             const IDPriorityResult& id_priority_result) const {
    
    PoseAnalysisResult result;
    result.person_id = person_id;
    result.analysis_timestamp = std::chrono::steady_clock::now();
    
    // 复制各模块的结果
    result.height_result = height_result;
    result.ball_request_result = ball_request_result;
    result.id_priority_result = id_priority_result;
    
    // 计算整体有效性
    result.analysis_valid = (height_result.person_id == person_id || height_result.person_id < 0) &&
                           (ball_request_result.person_id == person_id || ball_request_result.person_id < 0) &&
                           (id_priority_result.person_id == person_id || id_priority_result.person_id < 0);
    
    return result;
}

std::vector<PoseAnalysisResult> PoseAnalyzer::post_process_results(
    const std::vector<HeightResult>& height_results,
    const std::vector<BallRequestResult>& ball_request_results,
    const std::vector<IDPriorityResult>& id_priority_results) const {
    
    // 收集所有person ID
    std::set<int> all_person_ids;
    
    for (const auto& hr : height_results) {
        if (hr.person_id >= 0) all_person_ids.insert(hr.person_id);
    }
    
    for (const auto& brr : ball_request_results) {
        if (brr.person_id >= 0) all_person_ids.insert(brr.person_id);
    }
    
    for (const auto& ipr : id_priority_results) {
        if (ipr.person_id >= 0) all_person_ids.insert(ipr.person_id);
    }
    
    // 创建索引映射以便快速查找
    std::unordered_map<int, size_t> height_index_map;
    for (size_t i = 0; i < height_results.size(); ++i) {
        if (height_results[i].person_id >= 0) {
            height_index_map[height_results[i].person_id] = i;
        }
    }
    
    std::unordered_map<int, size_t> ball_request_index_map;
    for (size_t i = 0; i < ball_request_results.size(); ++i) {
        if (ball_request_results[i].person_id >= 0) {
            ball_request_index_map[ball_request_results[i].person_id] = i;
        }
    }
    
    std::unordered_map<int, size_t> id_priority_index_map;
    for (size_t i = 0; i < id_priority_results.size(); ++i) {
        if (id_priority_results[i].person_id >= 0) {
            id_priority_index_map[id_priority_results[i].person_id] = i;
        }
    }
    
    // 为每个person创建合并结果
    std::vector<PoseAnalysisResult> merged_results;
    merged_results.reserve(all_person_ids.size());
    
    for (int person_id : all_person_ids) {
        HeightResult height_result;
        height_result.person_id = person_id;  // 设置默认值
        
        BallRequestResult ball_request_result;
        ball_request_result.person_id = person_id;  // 设置默认值
        
        IDPriorityResult id_priority_result;
        id_priority_result.person_id = person_id;  // 设置默认值
        
        // 查找对应的结果
        auto height_it = height_index_map.find(person_id);
        if (height_it != height_index_map.end()) {
            height_result = height_results[height_it->second];
        }
        
        auto ball_request_it = ball_request_index_map.find(person_id);
        if (ball_request_it != ball_request_index_map.end()) {
            ball_request_result = ball_request_results[ball_request_it->second];
        }
        
        auto id_priority_it = id_priority_index_map.find(person_id);
        if (id_priority_it != id_priority_index_map.end()) {
            id_priority_result = id_priority_results[id_priority_it->second];
        }
        
        // 合并结果
        merged_results.push_back(merge_results(person_id, height_result, ball_request_result, id_priority_result));
    }
    
    // 按优先级ID排序
    std::sort(merged_results.begin(), merged_results.end(),
              [](const PoseAnalysisResult& a, const PoseAnalysisResult& b) {
                  return a.id_priority_result.priority_id < b.id_priority_result.priority_id;
              });
    
    return merged_results;
}

void PoseAnalyzer::enable_height_detection(bool enable) {
    // 可以动态控制模块的启用状态
    // 这里可以添加具体的控制逻辑
}

void PoseAnalyzer::enable_ball_request_detection(bool enable) {
    // 可以动态控制模块的启用状态
}

void PoseAnalyzer::enable_id_priority_management(bool enable) {
    // 可以动态控制模块的启用状态
}

void PoseAnalyzer::update_config(const PoseAnalysisConfig& config) {
    if (!validate_config(config)) {
        return;
    }
    
    config_ = config;
    
    // 更新各模块的配置
    if (height_detector_) {
        height_detector_->update_config(config_.height_detection);
    }
    
    if (ball_request_detector_) {
        ball_request_detector_->update_config(config_.ball_request);
    }
    
    if (id_priority_manager_) {
        id_priority_manager_->update_config(config_.id_management);
    }
}

void PoseAnalyzer::update_config_from_file(const std::string& config_file_path) {
    auto new_config = load_config_from_file(config_file_path);
    update_config(new_config);
}

void PoseAnalyzer::reset_all() {
    if (height_detector_) {
        height_detector_->reset();
    }
    
    if (ball_request_detector_) {
        ball_request_detector_->reset();
    }
    
    if (id_priority_manager_) {
        id_priority_manager_->reset_all_mappings();
    }
    
    processed_frames_ = 0;
    start_time_ = std::chrono::steady_clock::now();
}

void PoseAnalyzer::reset_person(int person_id) {
    if (height_detector_) {
        height_detector_->reset_person(person_id);
    }
    
    if (ball_request_detector_) {
        ball_request_detector_->reset_person(person_id);
    }
    
    if (id_priority_manager_) {
        id_priority_manager_->unregister_person(person_id);
    }
}

void PoseAnalyzer::cleanup_stale_data(int max_age_ms) {
    if (height_detector_) {
        height_detector_->cleanup_stale_contexts(max_age_ms);
    }
    
    if (ball_request_detector_) {
        ball_request_detector_->cleanup_stale_contexts(max_age_ms);
    }
    
    if (id_priority_manager_) {
        id_priority_manager_->cleanup_stale_tracking_info(max_age_ms);
    }
}

std::string PoseAnalyzer::get_debug_info() const {
    if (!initialized_) {
        return "PoseAnalyzer not initialized";
    }
    
    std::stringstream ss;
    ss << "=== Pose Analyzer Debug Info ===\n";
    ss << "Initialized: " << (initialized_ ? "YES" : "NO") << "\n";
    ss << "Enabled: " << (enabled_ ? "YES" : "NO") << "\n";
    ss << "Processed frames: " << processed_frames_ << "\n";
    
    if (height_detector_) {
        ss << "\n=== Height Detector ===\n";
        ss << height_detector_->get_debug_info();
    }
    
    if (ball_request_detector_) {
        ss << "\n=== Ball Request Detector ===\n";
        ss << ball_request_detector_->get_debug_info();
    }
    
    if (id_priority_manager_) {
        ss << "\n=== ID Priority Manager ===\n";
        ss << id_priority_manager_->get_debug_info();
    }
    
    return ss.str();
}

std::string PoseAnalyzer::get_performance_stats() const {
    std::stringstream ss;
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    ss << "=== Performance Statistics ===\n";
    ss << "Running time: " << elapsed.count() << "ms\n";
    ss << "Processed frames: " << processed_frames_ << "\n";
    
    if (elapsed.count() > 0) {
        float fps = (processed_frames_ * 1000.0f) / elapsed.count();
        ss << "Average FPS: " << fps << "\n";
    }
    
    // 可以添加更多性能统计信息
    
    return ss.str();
}

bool PoseAnalyzer::validate_config(const PoseAnalysisConfig& config) const {
    // 基本参数验证
    if (config.height_detection.window_size <= 0 ||
        config.height_detection.min_stable_frames <= 0) {
        return false;
    }
    
    if (config.ball_request.min_continuous_frames <= 0 ||
        config.ball_request.max_interruption_frames < 0) {
        return false;
    }
    
    if (config.id_management.max_tracked_persons <= 0 ||
        config.id_management.swap_cooldown_ms < 0) {
        return false;
    }
    
    if (config.global.frame_buffer_size <= 0 ||
        config.global.time_window_ms <= 0) {
        return false;
    }
    
    // 权重验证
    float weight_sum = config.id_management.confidence_weight + 
                      config.id_management.duration_weight + 
                      config.id_management.stability_weight;
    
    if (std::abs(weight_sum - 1.0f) > 0.1f) {
        // 权重和应该接近1.0
        return false;
    }
    
    return true;
}

PoseAnalysisConfig PoseAnalyzer::create_default_config() {
    PoseAnalysisConfig config;
    
    // 身高检测默认配置
    config.height_detection.min_keypoint_confidence = 0.5f;
    config.height_detection.filter_type = "median";
    config.height_detection.window_size = 15;
    config.height_detection.min_stable_frames = 10;
    config.height_detection.stability_threshold_mm = 50.0f;
    
    // 要球动作检测默认配置
    config.ball_request.min_continuous_frames = 5;
    config.ball_request.max_interruption_frames = 2;
    config.ball_request.min_total_confidence = 3.5f;
    config.ball_request.gesture_stability_threshold = 0.2f;
    
    // ID优先级管理默认配置
    config.id_management.confidence_weight = 0.3f;
    config.id_management.duration_weight = 0.4f;
    config.id_management.stability_weight = 0.3f;
    config.id_management.min_request_frames_for_swap = 15;
    config.id_management.swap_cooldown_ms = 2000;
    
    // 全局配置
    config.global.frame_buffer_size = 60;
    config.global.time_window_ms = 2000;
    
    return config;
}

PoseAnalysisConfig PoseAnalyzer::load_config_from_file(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return create_default_config();  // 文件不存在时返回默认配置
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();
    
    PoseAnalysisConfig config;
    if (!parse_config_from_json(buffer.str(), config)) {
        return create_default_config();  // 解析失败时返回默认配置
    }
    
    return config;
}

bool PoseAnalyzer::save_config_to_file(const std::string& file_path) const {
    std::string json_content = serialize_config_to_json(config_);
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << json_content;
    file.close();
    
    return true;
}

bool PoseAnalyzer::parse_config_from_json(const std::string& json_content, PoseAnalysisConfig& config) const {
    try {
        Json::Value root;
        Json::Reader reader;
        
        if (!reader.parse(json_content, root)) {
            return false;
        }
        
        // 解析身高检测配置
        if (root.isMember("height_detection")) {
            const Json::Value& hd = root["height_detection"];
            if (hd.isMember("measurement")) {
                config.height_detection.min_keypoint_confidence = hd["measurement"].get("min_keypoint_confidence", 0.5f).asFloat();
                config.height_detection.head_offset_pixels = hd["measurement"].get("head_offset_pixels", 30).asInt();
                config.height_detection.height_correction_factor = hd["measurement"].get("height_correction_factor", 1.05f).asFloat();
            }
            if (hd.isMember("filtering")) {
                config.height_detection.filter_type = hd["filtering"].get("type", "median").asString();
                config.height_detection.window_size = hd["filtering"].get("window_size", 15).asInt();
                config.height_detection.stability_threshold_mm = hd["filtering"].get("stability_threshold_mm", 50.0f).asFloat();
            }
        }
        
        // 解析要球动作检测配置
        if (root.isMember("ball_request_detection")) {
            const Json::Value& brd = root["ball_request_detection"];
            if (brd.isMember("temporal")) {
                config.ball_request.min_continuous_frames = brd["temporal"].get("min_continuous_frames", 5).asInt();
                config.ball_request.max_interruption_frames = brd["temporal"].get("max_interruption_frames", 2).asInt();
                config.ball_request.min_total_confidence = brd["temporal"].get("min_total_confidence", 3.5f).asFloat();
            }
        }
        
        // 解析ID管理配置
        if (root.isMember("id_management")) {
            const Json::Value& im = root["id_management"];
            if (im.isMember("priority")) {
                config.id_management.confidence_weight = im["priority"].get("confidence_weight", 0.3f).asFloat();
                config.id_management.duration_weight = im["priority"].get("duration_weight", 0.4f).asFloat();
                config.id_management.stability_weight = im["priority"].get("stability_weight", 0.3f).asFloat();
            }
            if (im.isMember("temporal")) {
                config.id_management.swap_cooldown_ms = im["temporal"].get("swap_cooldown_ms", 2000).asInt();
                config.id_management.min_request_frames_for_swap = im["temporal"].get("min_request_frames_for_swap", 15).asInt();
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::string PoseAnalyzer::serialize_config_to_json(const PoseAnalysisConfig& config) const {
    Json::Value root;
    Json::StreamWriterBuilder builder;
    
    // 身高检测配置
    Json::Value& hd = root["height_detection"];
    hd["enabled"] = true;
    
    Json::Value& hd_measurement = hd["measurement"];
    hd_measurement["min_keypoint_confidence"] = config.height_detection.min_keypoint_confidence;
    hd_measurement["head_offset_pixels"] = config.height_detection.head_offset_pixels;
    hd_measurement["height_correction_factor"] = config.height_detection.height_correction_factor;
    
    Json::Value& hd_filtering = hd["filtering"];
    hd_filtering["type"] = config.height_detection.filter_type;
    hd_filtering["window_size"] = config.height_detection.window_size;
    hd_filtering["stability_threshold_mm"] = config.height_detection.stability_threshold_mm;
    
    // 要球动作检测配置
    Json::Value& brd = root["ball_request_detection"];
    brd["enabled"] = true;
    
    Json::Value& brd_temporal = brd["temporal"];
    brd_temporal["min_continuous_frames"] = config.ball_request.min_continuous_frames;
    brd_temporal["max_interruption_frames"] = config.ball_request.max_interruption_frames;
    brd_temporal["min_total_confidence"] = config.ball_request.min_total_confidence;
    
    // ID管理配置
    Json::Value& im = root["id_management"];
    im["enabled"] = true;
    
    Json::Value& im_priority = im["priority"];
    im_priority["confidence_weight"] = config.id_management.confidence_weight;
    im_priority["duration_weight"] = config.id_management.duration_weight;
    im_priority["stability_weight"] = config.id_management.stability_weight;
    
    Json::Value& im_temporal = im["temporal"];
    im_temporal["swap_cooldown_ms"] = config.id_management.swap_cooldown_ms;
    im_temporal["min_request_frames_for_swap"] = config.id_management.min_request_frames_for_swap;
    
    return Json::writeString(builder, root);
}

// ===== PoseAnalyzerBuilder 实现 =====

PoseAnalyzerBuilder::PoseAnalyzerBuilder() {
    config_ = PoseAnalyzer::create_default_config();
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::with_height_detection(bool enable) {
    // 可以控制模块启用状态
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::height_filter_type(const std::string& filter_type) {
    config_.height_detection.filter_type = filter_type;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::height_window_size(int window_size) {
    config_.height_detection.window_size = window_size;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::height_stability_threshold(float threshold_mm) {
    config_.height_detection.stability_threshold_mm = threshold_mm;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::ball_request_min_frames(int min_frames) {
    config_.ball_request.min_continuous_frames = min_frames;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::ball_request_max_interruption(int max_interruption) {
    config_.ball_request.max_interruption_frames = max_interruption;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::id_priority_weights(float confidence_weight, 
                                                            float duration_weight, 
                                                            float stability_weight) {
    config_.id_management.confidence_weight = confidence_weight;
    config_.id_management.duration_weight = duration_weight;
    config_.id_management.stability_weight = stability_weight;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::id_swap_cooldown(int cooldown_ms) {
    config_.id_management.swap_cooldown_ms = cooldown_ms;
    return *this;
}

PoseAnalyzerBuilder& PoseAnalyzerBuilder::from_config_file(const std::string& file_path) {
    config_ = PoseAnalyzer::load_config_from_file(file_path);
    return *this;
}

std::unique_ptr<PoseAnalyzer> PoseAnalyzerBuilder::build() {
    return std::make_unique<PoseAnalyzer>(config_);
}

// ===== 工厂函数实现 =====

std::unique_ptr<PoseAnalyzer> create_pose_analyzer(const std::string& config_file_path) {
    return std::make_unique<PoseAnalyzer>(config_file_path);
}

std::unique_ptr<PoseAnalyzer> create_default_pose_analyzer() {
    auto config = PoseAnalyzer::create_default_config();
    return std::make_unique<PoseAnalyzer>(config);
}

} // namespace pose_analysis