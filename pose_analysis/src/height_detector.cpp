#include "height_detector.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace pose_analysis {

HeightDetector::HeightDetector(const HeightDetectionConfig& config)
    : config_(config), has_homography_(false), frame_counter_(0) {
}

void HeightDetector::set_homography(const cv::Mat& homography) {
    homography_matrix_ = homography.clone();
    has_homography_ = !homography.empty();
}

std::vector<HeightResult> HeightDetector::process_frame(const std::vector<PoseResult>& pose_results) {
    frame_counter_++;
    std::vector<HeightResult> results;
    
    // 处理每个检测到的person
    for (const auto& pose : pose_results) {
        if (pose.person_id >= 0) {  // 有效的person ID (考虑ByteTrack分配的ID)
            auto result = process_person(pose);
            results.push_back(result);
        }
    }
    
    // 清理过期的上下文
    cleanup_stale_contexts();
    
    return results;
}

HeightResult HeightDetector::process_person(const PoseResult& pose) {
    HeightResult result;
    result.person_id = pose.person_id;
    result.measurement_start_time = pose.timestamp;
    
    // 获取或创建person的上下文
    auto& context = get_or_create_context(pose.person_id);
    
    // 强制测量模式：跳过姿势检查，总是尝试计算身高
    // if (!can_measure_height(pose)) {
    //     // 不满足测量条件，可能需要重置状态
    //     if (context.state == HeightDetectionState::MEASURING) {
    //         context.invalid_frames_count++;
    //         if (context.invalid_frames_count > 10) {  // 连续10帧无效，重置
    //             transition_to_idle(context);
    //         }
    //     }
    //     
    //     result.state = context.state;
    //     result.is_stable = false;
    //     result.estimated_height_mm = context.stable_height_mm;
    //     result.confidence = 0.0f;
    //     return result;
    // }
    
    // 计算原始身高
    float raw_height = calculate_raw_height(pose);
    
    if (raw_height <= 0) {
        result.state = context.state;
        result.is_stable = false;
        result.estimated_height_mm = context.stable_height_mm;
        result.confidence = 0.0f;
        return result;
    }
    
    // 创建测量记录
    HeightMeasurement measurement;
    measurement.height_mm = raw_height;
    measurement.confidence = calculate_measurement_confidence(pose);
    measurement.frame_id = frame_counter_;
    measurement.timestamp = pose.timestamp;
    measurement.is_valid = !is_outlier(raw_height, context);
    
    // 更新状态机
    update_state_machine(context, measurement);
    
    // 如果测量有效，进行滤波处理
    if (measurement.is_valid) {
        context.raw_heights.push(raw_height);
        
        if (context.height_filter) {
            context.height_filter->process(raw_height);
        }
        if (context.confidence_filter) {
            context.confidence_filter->process(measurement.confidence);
        }
        
        context.invalid_frames_count = 0;  // 重置无效帧计数
        context.last_valid_measurement = pose.timestamp;
    } else {
        context.invalid_frames_count++;
    }
    
    // 添加到测量缓冲区
    context.measurements.push(measurement);
    
    // 更新结果
    result.state = context.state;
    result.stable_frames_count = context.stable_frames_count;
    result.measurement_start_time = context.measurement_start_time;
    
    if (context.state == HeightDetectionState::STABLE) {
        result.is_stable = true;
        result.estimated_height_mm = calculate_final_height(context);
        result.confidence = context.stable_confidence;
    } else {
        result.is_stable = false;
        result.estimated_height_mm = context.stable_height_mm;
        result.confidence = context.height_filter && context.confidence_filter ? 
                           context.confidence_filter->process(measurement.confidence) : 0.0f;
    }
    
    return result;
}

bool HeightDetector::can_measure_height(const PoseResult& pose) const {
    // 检查必要关键点的置信度
    if (!is_keypoint_valid(pose, COCOKeypoint::NOSE) ||
        !is_keypoint_valid(pose, COCOKeypoint::LEFT_ANKLE) ||
        !is_keypoint_valid(pose, COCOKeypoint::RIGHT_ANKLE)) {
        return false;
    }
    
    // 检查双手是否都在头部以下 (要球时双手可能举起)
    bool left_wrist_valid = is_keypoint_valid(pose, COCOKeypoint::LEFT_WRIST);
    bool right_wrist_valid = is_keypoint_valid(pose, COCOKeypoint::RIGHT_WRIST);
    
    if (left_wrist_valid || right_wrist_valid) {
        cv::Point2f nose_pos = pose.keypoints[static_cast<int>(COCOKeypoint::NOSE)];
        
        // 允许手腕高度稍微超过鼻子，但不能太多
        float nose_y = nose_pos.y;
        float height_threshold = nose_y - config_.head_offset_pixels;
        
        if (left_wrist_valid) {
            cv::Point2f left_wrist = pose.keypoints[static_cast<int>(COCOKeypoint::LEFT_WRIST)];
            if (left_wrist.y < height_threshold) {
                return false;  // 左手举得太高
            }
        }
        
        if (right_wrist_valid) {
            cv::Point2f right_wrist = pose.keypoints[static_cast<int>(COCOKeypoint::RIGHT_WRIST)];
            if (right_wrist.y < height_threshold) {
                return false;  // 右手举得太高
            }
        }
    }
    
    // 检查ROI高度是否在合理范围内
    float roi_height = pose.bbox.height;
    if (roi_height < config_.min_roi_height_pixels || 
        roi_height > config_.max_roi_height_pixels) {
        return false;
    }
    
    return true;
}

float HeightDetector::calculate_raw_height(const PoseResult& pose) const {
    if (!has_homography_) {
        // 没有Homography矩阵，使用像素估算
        return pose.bbox.height * config_.height_correction_factor;
    }
    
    // 使用关键点计算更准确的身高
    cv::Point2f head_pos, foot_pos;
    
    // 优先使用鼻子作为头部位置
    if (is_keypoint_valid(pose, COCOKeypoint::NOSE)) {
        head_pos = pose.keypoints[static_cast<int>(COCOKeypoint::NOSE)];
        // 向上偏移以估计头顶位置
        head_pos.y -= config_.head_offset_pixels;
    } else {
        head_pos = cv::Point2f(pose.bbox.x + pose.bbox.width/2, pose.bbox.y);
    }
    
    // 使用脚踝的平均位置作为脚部位置
    bool left_ankle_valid = is_keypoint_valid(pose, COCOKeypoint::LEFT_ANKLE);
    bool right_ankle_valid = is_keypoint_valid(pose, COCOKeypoint::RIGHT_ANKLE);
    
    if (left_ankle_valid && right_ankle_valid) {
        cv::Point2f left_ankle = pose.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)];
        cv::Point2f right_ankle = pose.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)];
        foot_pos = (left_ankle + right_ankle) * 0.5f;
    } else if (left_ankle_valid) {
        foot_pos = pose.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)];
    } else if (right_ankle_valid) {
        foot_pos = pose.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)];
    } else {
        foot_pos = cv::Point2f(pose.bbox.x + pose.bbox.width/2, pose.bbox.y + pose.bbox.height);
    }
    
    // 转换为世界坐标并计算距离
    return calculate_world_distance(head_pos, foot_pos);
}

float HeightDetector::calculate_measurement_confidence(const PoseResult& pose) const {
    float confidence = pose.detection_confidence;
    
    // 考虑关键点的置信度
    std::vector<COCOKeypoint> important_keypoints = {
        COCOKeypoint::NOSE, COCOKeypoint::LEFT_ANKLE, COCOKeypoint::RIGHT_ANKLE
    };
    
    float keypoint_confidence_sum = 0.0f;
    int valid_keypoints = 0;
    
    for (auto keypoint : important_keypoints) {
        int idx = static_cast<int>(keypoint);
        if (idx < pose.keypoint_confidences.size() && 
            pose.keypoint_confidences[idx] > config_.min_keypoint_confidence) {
            keypoint_confidence_sum += pose.keypoint_confidences[idx];
            valid_keypoints++;
        }
    }
    
    if (valid_keypoints > 0) {
        float avg_keypoint_confidence = keypoint_confidence_sum / valid_keypoints;
        confidence = (confidence + avg_keypoint_confidence) * 0.5f;
    }
    
    // 考虑ROI的稳定性
    float roi_stability = 1.0f;
    if (pose.bbox.width > 0 && pose.bbox.height > 0) {
        float aspect_ratio = pose.bbox.width / pose.bbox.height;
        // 人体的宽高比应该在0.3-0.8之间
        if (aspect_ratio < 0.2f || aspect_ratio > 1.0f) {
            roi_stability *= 0.5f;
        }
    }
    
    return confidence * roi_stability;
}

bool HeightDetector::is_outlier(float height, const PersonHeightContext& context) const {
    if (context.measurements.size() < 3) {
        return false;  // 数据不足，不判定为异常值
    }
    
    // 获取最近的测量值
    auto recent_measurements = context.measurements.get_window(config_.window_size);
    std::vector<float> heights;
    for (const auto& m : recent_measurements) {
        if (m.is_valid) {
            heights.push_back(m.height_mm);
        }
    }
    
    if (heights.size() < 3) {
        return false;
    }
    
    // 使用3σ原则或MAD方法检测异常值
    auto outliers = BufferStatistics::detect_outliers_3sigma(heights, config_.outlier_threshold_sigma);
    
    // 检查当前值是否为异常值（与最近的值比较）
    float median_height = BufferStatistics::median(heights);
    float deviation = std::abs(height - median_height);
    float mad = BufferStatistics::median_absolute_deviation(heights);
    
    return deviation > (config_.outlier_threshold_sigma * mad);
}

float HeightDetector::calculate_stability(const PersonHeightContext& context) const {
    auto recent_heights = context.raw_heights.get_recent_data();
    if (recent_heights.size() < config_.min_stable_frames) {
        return 1000.0f;  // 不稳定
    }
    
    return BufferStatistics::standard_deviation(recent_heights);
}

void HeightDetector::update_state_machine(PersonHeightContext& context, 
                                        const HeightMeasurement& measurement) {
    switch (context.state) {
        case HeightDetectionState::IDLE:
            if (measurement.is_valid && measurement.confidence > config_.min_confidence) {
                transition_to_measuring(context);
            }
            break;
            
        case HeightDetectionState::MEASURING:
            if (measurement.is_valid) {
                context.measuring_frames_count++;
                
                // 检查是否达到稳定状态
                if (context.measuring_frames_count >= config_.min_stable_frames) {
                    float stability = calculate_stability(context);
                    if (stability < config_.stability_threshold_mm && 
                        context.height_filter && context.height_filter->is_stable()) {
                        transition_to_stable(context);
                    }
                }
            } else {
                // 测量无效，可能需要重置或转为无效状态
                if (context.invalid_frames_count > 5) {
                    transition_to_invalid(context);
                }
            }
            break;
            
        case HeightDetectionState::STABLE:
            if (measurement.is_valid) {
                context.stable_frames_count++;
                // 更新稳定的身高值
                context.stable_height_mm = calculate_final_height(context);
                if (context.confidence_filter) {
                    context.stable_confidence = context.confidence_filter->process(measurement.confidence);
                }
            } else {
                // 稳定状态下出现无效测量，检查是否需要转换状态
                if (context.invalid_frames_count > 10) {
                    transition_to_measuring(context);
                }
            }
            break;
            
        case HeightDetectionState::INVALID:
            if (measurement.is_valid && measurement.confidence > config_.min_confidence) {
                // 重新开始测量
                transition_to_measuring(context);
            }
            // 在无效状态停留一段时间后自动重置
            if (context.invalid_frames_count > 30) {
                transition_to_idle(context);
            }
            break;
    }
}

void HeightDetector::transition_to_measuring(PersonHeightContext& context) {
    context.state = HeightDetectionState::MEASURING;
    context.measuring_frames_count = 1;
    context.stable_frames_count = 0;
    context.invalid_frames_count = 0;
    context.measurement_start_time = std::chrono::steady_clock::now();
    
    // 重置滤波器
    if (context.height_filter) context.height_filter->reset();
    if (context.confidence_filter) context.confidence_filter->reset();
}

void HeightDetector::transition_to_stable(PersonHeightContext& context) {
    context.state = HeightDetectionState::STABLE;
    context.stable_frames_count = 1;
    context.invalid_frames_count = 0;
    
    // 计算并保存稳定的身高值
    context.stable_height_mm = calculate_final_height(context);
    context.stable_confidence = context.confidence_filter ? 
                               context.confidence_filter->process(0.9f) : 0.9f;
}

void HeightDetector::transition_to_invalid(PersonHeightContext& context) {
    context.state = HeightDetectionState::INVALID;
    context.measuring_frames_count = 0;
    context.stable_frames_count = 0;
}

void HeightDetector::transition_to_idle(PersonHeightContext& context) {
    context.state = HeightDetectionState::IDLE;
    context.measuring_frames_count = 0;
    context.stable_frames_count = 0;
    context.invalid_frames_count = 0;
    context.stable_height_mm = -1.0f;
    context.stable_confidence = 0.0f;
    
    // 清空缓冲区但保留滤波器状态以便快速恢复
    context.measurements.clear();
    context.raw_heights.clear();
}

float HeightDetector::calculate_final_height(const PersonHeightContext& context) const {
    if (!context.height_filter || !context.height_filter->is_stable()) {
        // 滤波器不稳定，使用统计方法
        auto recent_heights = context.raw_heights.get_recent_data();
        if (!recent_heights.empty()) {
            auto filtered_heights = BufferStatistics::filter_outliers(recent_heights);
            if (!filtered_heights.empty()) {
                return BufferStatistics::median(filtered_heights) * config_.height_correction_factor;
            }
        }
        return context.stable_height_mm;
    }
    
    // 使用滤波器的当前输出
    auto recent_heights = context.raw_heights.get_recent_data();
    if (!recent_heights.empty()) {
        float filtered_result = context.height_filter->process(recent_heights.back());
        return filtered_result * config_.height_correction_factor;
    }
    
    return context.stable_height_mm;
}

HeightDetector::PersonHeightContext& HeightDetector::get_or_create_context(int person_id) {
    auto it = person_contexts_.find(person_id);
    if (it == person_contexts_.end()) {
        person_contexts_[person_id] = std::make_unique<PersonHeightContext>(person_id, config_);
    }
    return *person_contexts_[person_id];
}

cv::Point2f HeightDetector::pixel_to_world(const cv::Point2f& pixel_point) const {
    if (!has_homography_) {
        return pixel_point;  // 直接返回像素坐标
    }
    
    std::vector<cv::Point2f> src_points = {pixel_point};
    std::vector<cv::Point2f> dst_points;
    cv::perspectiveTransform(src_points, dst_points, homography_matrix_);
    
    return dst_points[0];
}

float HeightDetector::calculate_world_distance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    cv::Point2f world_p1 = pixel_to_world(p1);
    cv::Point2f world_p2 = pixel_to_world(p2);
    
    float dx = world_p1.x - world_p2.x;
    float dy = world_p1.y - world_p2.y;
    
    return std::sqrt(dx * dx + dy * dy);  // 世界坐标单位为毫米
}

bool HeightDetector::is_keypoint_valid(const PoseResult& pose, COCOKeypoint keypoint) const {
    int idx = static_cast<int>(keypoint);
    if (idx >= pose.keypoint_confidences.size()) return false;
    
    return pose.keypoint_confidences[idx] > config_.min_keypoint_confidence;
}

cv::Point2f HeightDetector::get_keypoint_world_pos(const PoseResult& pose, COCOKeypoint keypoint) const {
    int idx = static_cast<int>(keypoint);
    if (idx >= pose.keypoints.size()) return cv::Point2f(-1, -1);
    
    return pixel_to_world(pose.keypoints[idx]);
}

void HeightDetector::update_config(const HeightDetectionConfig& config) {
    config_ = config;
    
    // 重新创建所有person的滤波器
    for (auto& pair : person_contexts_) {
        auto& context = *pair.second;
        context.height_filter = FilterFactory::create_filter(config.filter_type, 
                                                           config.window_size,
                                                           config.kalman_process_noise,
                                                           config.kalman_measurement_noise);
        context.confidence_filter = FilterFactory::create_moving_average_filter(config.window_size, 0.8f);
    }
}

void HeightDetector::reset() {
    person_contexts_.clear();
    frame_counter_ = 0;
}

void HeightDetector::reset_person(int person_id) {
    auto it = person_contexts_.find(person_id);
    if (it != person_contexts_.end()) {
        transition_to_idle(*it->second);
    }
}

void HeightDetector::cleanup_stale_contexts(int max_age_ms) {
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = person_contexts_.begin(); it != person_contexts_.end();) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second->last_valid_measurement).count();
        
        if (elapsed > max_age_ms) {
            it = person_contexts_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string HeightDetector::get_debug_info(int person_id) const {
    std::stringstream ss;
    
    if (person_id >= 0) {
        auto it = person_contexts_.find(person_id);
        if (it != person_contexts_.end()) {
            const auto& context = *it->second;
            ss << "Person " << person_id << ":\n";
            ss << "  State: " << static_cast<int>(context.state) << "\n";
            ss << "  Stable height: " << context.stable_height_mm << "mm\n";
            ss << "  Stable frames: " << context.stable_frames_count << "\n";
            ss << "  Measuring frames: " << context.measuring_frames_count << "\n";
            ss << "  Invalid frames: " << context.invalid_frames_count << "\n";
            
            if (context.height_filter) {
                ss << "  Height filter: " << context.height_filter->get_status() << "\n";
            }
            if (context.confidence_filter) {
                ss << "  Confidence filter: " << context.confidence_filter->get_status() << "\n";
            }
        }
    } else {
        ss << "Height Detector Status:\n";
        ss << "  Active persons: " << person_contexts_.size() << "\n";
        ss << "  Frame counter: " << frame_counter_ << "\n";
        ss << "  Has homography: " << (has_homography_ ? "Yes" : "No") << "\n";
        
        for (const auto& pair : person_contexts_) {
            const auto& context = *pair.second;
            ss << "  Person " << pair.first << ": state=" << static_cast<int>(context.state) 
               << ", height=" << context.stable_height_mm << "mm\n";
        }
    }
    
    return ss.str();
}

std::vector<int> HeightDetector::get_active_person_ids() const {
    std::vector<int> ids;
    for (const auto& pair : person_contexts_) {
        ids.push_back(pair.first);
    }
    return ids;
}

} // namespace pose_analysis