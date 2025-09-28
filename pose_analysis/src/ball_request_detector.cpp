#include "ball_request_detector.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace pose_analysis {

BallRequestDetector::BallRequestDetector(const BallRequestConfig& config)
    : config_(config), has_homography_(false), frame_counter_(0) {
}

void BallRequestDetector::set_homography(const cv::Mat& homography) {
    homography_matrix_ = homography.clone();
    has_homography_ = !homography.empty();
}

std::vector<BallRequestResult> BallRequestDetector::process_frame(const std::vector<PoseResult>& pose_results) {
    frame_counter_++;
    std::vector<BallRequestResult> results;
    
    // 处理每个检测到的person
    for (const auto& pose : pose_results) {
        if (pose.person_id >= 0) {  // 有效的person ID (ByteTrack分配的ID)
            auto result = process_person(pose);
            results.push_back(result);
        }
    }
    
    // 清理过期的上下文
    cleanup_stale_contexts(3000);
    
    return results;
}

BallRequestResult BallRequestDetector::process_person(const PoseResult& pose) {
    BallRequestResult result;
    result.person_id = pose.person_id;
    result.request_start_time = pose.timestamp;
    
    // 获取或创建person的上下文
    auto& context = get_or_create_context(pose.person_id);
    
    // 检测原始要球动作
    float confidence = 0.0f;
    bool is_requesting = detect_raw_request(pose, confidence);
    
    // 更新时序缓冲区
    if (is_requesting) {
        auto features = calculate_gesture_features(pose);
        context.hands_distance_buffer.push(features.hands_distance_mm);
        context.left_wrist_buffer.push(get_keypoint_position(pose, COCOKeypoint::LEFT_WRIST));
        context.right_wrist_buffer.push(get_keypoint_position(pose, COCOKeypoint::RIGHT_WRIST));
    }
    
    // 更新置信度缓冲区
    context.confidence_buffer.push(confidence);
    
    // 更新状态机
    update_state_machine(context, is_requesting, confidence);
    
    // 设置返回结果
    result.is_requesting = is_requesting;
    result.is_confirmed = (context.state == BallRequestState::CONFIRMED_REQUEST);
    result.state = context.state;
    result.continuous_frames = context.continuous_frames;
    result.total_frames = context.total_frames;
    result.interruption_frames = context.interruption_frames;
    result.accumulated_confidence = context.accumulated_confidence;
    result.request_confidence = context.continuous_frames > 0 ? 
                               context.accumulated_confidence / context.continuous_frames : 0.0f;
    result.gesture_stability = calculate_gesture_stability(context);
    result.request_start_time = context.request_start_time;
    result.request_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        pose.timestamp - context.request_start_time).count();
    
    return result;
}

std::vector<BallRequestResult> BallRequestDetector::process_batch(
    const std::vector<std::vector<PoseResult>>& batch_poses) {
    
    std::vector<BallRequestResult> all_results;
    
    for (const auto& frame_poses : batch_poses) {
        auto frame_results = process_frame(frame_poses);
        all_results.insert(all_results.end(), frame_results.begin(), frame_results.end());
    }
    
    return all_results;
}

bool BallRequestDetector::detect_raw_request(const PoseResult& pose, float& confidence) const {
    confidence = 0.0f;
    
    // 计算手势特征
    auto features = calculate_gesture_features(pose);
    
    // 调试输出：要球检测详细参数
    std::cout << "=== 要球检测调试 Person " << pose.person_id << " ===" << std::endl;
    std::cout << "手势有效性: " << (features.gesture_valid ? "是" : "否") << std::endl;
    
    if (!features.gesture_valid) {
        std::cout << "  原因：缺少必要关键点或关键点置信度太低" << std::endl;
        return false;
    }
    
    // 🔥 改进：使用ROI相对距离判断（距离无关）
    bool hands_close = features.hands_distance_ratio <= config_.max_hands_distance_ratio;
    bool hands_in_chest_area = features.hands_to_chest_ratio <= config_.max_chest_distance_ratio;
    bool height_appropriate = features.hand_height_ratio >= config_.min_hand_height_ratio &&
                             features.hand_height_ratio <= config_.max_hand_height_ratio;
    
    // 计算置信度
    confidence = calculate_request_confidence(features);
    
    // 详细调试输出
    std::cout << "🔥 双手距离比例: " << features.hands_distance_ratio << " (限制: <=" << config_.max_hands_distance_ratio << ")" << std::endl;
    std::cout << "双手距离检查: " << (hands_close ? "通过" : "失败") << std::endl;
    std::cout << "   [兼容] 绝对距离: " << features.hands_distance_mm << "mm" << std::endl;
    
    std::cout << "🔥 手到胸部比例: " << features.hands_to_chest_ratio << " (限制: <=" << config_.max_chest_distance_ratio << ")" << std::endl;
    std::cout << "胸部区域检查: " << (hands_in_chest_area ? "通过" : "失败") << std::endl;
    std::cout << "   [兼容] 绝对距离: " << features.hands_to_chest_distance << "mm" << std::endl;
    
    std::cout << "手高度比例: " << features.hand_height_ratio << " (范围: " 
              << config_.min_hand_height_ratio << "-" << config_.max_hand_height_ratio << ")" << std::endl;
    std::cout << "高度检查: " << (height_appropriate ? "通过" : "失败") << std::endl;
    
    std::cout << "双手在前方: " << (features.hands_in_front ? "是" : "否") << std::endl;
    std::cout << "置信度: " << confidence << " (需要 >0.3)" << std::endl;
    
    bool final_result = hands_close && hands_in_chest_area && height_appropriate && 
                       features.hands_in_front && confidence > 0.3f;
    std::cout << "最终要球判断: " << (final_result ? "要球" : "不要球") << std::endl;
    std::cout << "=========================================" << std::endl;
    
    return final_result;
}

BallRequestDetector::GestureFeatures BallRequestDetector::calculate_gesture_features(
    const PoseResult& pose) const {
    
    GestureFeatures features;
    features.gesture_valid = false;
    features.feature_confidence = 0.0f;
    
    // 检查必要的关键点
    if (!is_keypoint_valid(pose, COCOKeypoint::LEFT_WRIST) ||
        !is_keypoint_valid(pose, COCOKeypoint::RIGHT_WRIST)) {
        return features;
    }
    
    cv::Point2f left_wrist = get_keypoint_position(pose, COCOKeypoint::LEFT_WRIST);
    cv::Point2f right_wrist = get_keypoint_position(pose, COCOKeypoint::RIGHT_WRIST);
    
    // 计算双手距离
    features.hands_distance_mm = calculate_world_distance(left_wrist, right_wrist);
    features.hands_center = (left_wrist + right_wrist) * 0.5f;
    
    // 🔥 新增：计算ROI相对距离（距离无关）
    float hands_distance_pixels = cv::norm(left_wrist - right_wrist);
    features.hands_distance_ratio = (pose.bbox.width > 0) ? 
        hands_distance_pixels / pose.bbox.width : 0.0f;
    
    // 计算胸部中心
    features.chest_center = calculate_chest_center(pose);
    
    // 计算手到胸部的距离
    features.hands_to_chest_distance = calculate_world_distance(features.hands_center, features.chest_center);
    
    // 🔥 新增：计算手到胸部的相对距离
    float chest_distance_pixels = cv::norm(features.hands_center - features.chest_center);
    features.hands_to_chest_ratio = (pose.bbox.width > 0) ? 
        chest_distance_pixels / pose.bbox.width : 0.0f;
    
    // 计算手的高度比例
    if (pose.bbox.height > 0) {
        float hand_y = features.hands_center.y;
        float bbox_top = pose.bbox.y;
        float bbox_bottom = pose.bbox.y + pose.bbox.height;
        features.hand_height_ratio = (hand_y - bbox_top) / pose.bbox.height;
    }
    
    // 检查手是否在身体前方
    features.hands_in_front = is_point_in_front_of_chest(features.hands_center, features.chest_center, pose);
    
    // 计算特征置信度
    float wrist_confidence = (pose.keypoint_confidences[static_cast<int>(COCOKeypoint::LEFT_WRIST)] +
                             pose.keypoint_confidences[static_cast<int>(COCOKeypoint::RIGHT_WRIST)]) * 0.5f;
    features.feature_confidence = wrist_confidence * pose.detection_confidence;
    
    features.gesture_valid = features.feature_confidence > config_.min_keypoint_confidence;
    
    return features;
}

float BallRequestDetector::calculate_gesture_stability(const RequestContext& context) const {
    // 计算手腕位置的稳定性
    auto left_wrist_window = context.left_wrist_buffer.get_window(5);
    auto right_wrist_window = context.right_wrist_buffer.get_window(5);
    
    if (left_wrist_window.size() < 3) {
        return 1.0f;  // 数据不足，返回不稳定值
    }
    
    // 计算位置变化的方差
    float total_variance = 0.0f;
    
    // 计算左手腕的方差
    cv::Point2f left_mean(0, 0);
    for (const auto& pt : left_wrist_window) {
        left_mean += pt;
    }
    left_mean /= static_cast<float>(left_wrist_window.size());
    
    float left_variance = 0.0f;
    for (const auto& pt : left_wrist_window) {
        float dist = cv::norm(pt - left_mean);
        left_variance += dist * dist;
    }
    left_variance /= left_wrist_window.size();
    
    // 计算右手腕的方差
    cv::Point2f right_mean(0, 0);
    for (const auto& pt : right_wrist_window) {
        right_mean += pt;
    }
    right_mean /= static_cast<float>(right_wrist_window.size());
    
    float right_variance = 0.0f;
    for (const auto& pt : right_wrist_window) {
        float dist = cv::norm(pt - right_mean);
        right_variance += dist * dist;
    }
    right_variance /= right_wrist_window.size();
    
    total_variance = (left_variance + right_variance) * 0.5f;
    
    // 归一化稳定性指标 (方差越小越稳定)
    return std::sqrt(total_variance) / 100.0f;
}

float BallRequestDetector::calculate_request_confidence(const GestureFeatures& features) const {
    if (!features.gesture_valid) {
        return 0.0f;
    }
    
    float confidence = features.feature_confidence;
    
    // 🔥 改进：使用相对距离计算置信度因子（距离无关）
    float distance_factor = 1.0f;
    if (features.hands_distance_ratio > 0) {
        float ideal_ratio = config_.max_hands_distance_ratio * 0.7f;  // 理想比例为最大比例的70%
        float ratio_diff = std::abs(features.hands_distance_ratio - ideal_ratio);
        distance_factor = std::max(0.0f, 1.0f - ratio_diff / ideal_ratio);
    }
    
    // 🔥 改进：使用相对距离计算位置因子
    float position_factor = 1.0f;
    if (features.hands_to_chest_ratio > 0) {
        float ideal_chest_ratio = config_.max_chest_distance_ratio * 0.7f;  // 理想胸部比例
        float chest_ratio_diff = std::abs(features.hands_to_chest_ratio - ideal_chest_ratio);
        position_factor = std::max(0.0f, 1.0f - chest_ratio_diff / ideal_chest_ratio);
    }
    
    // 高度因子：手的高度在合适范围内
    float height_factor = 1.0f;
    if (features.hand_height_ratio >= config_.min_hand_height_ratio &&
        features.hand_height_ratio <= config_.max_hand_height_ratio) {
        float ideal_height = (config_.min_hand_height_ratio + config_.max_hand_height_ratio) * 0.5f;
        float height_diff = std::abs(features.hand_height_ratio - ideal_height);
        float height_range = config_.max_hand_height_ratio - config_.min_hand_height_ratio;
        height_factor = std::max(0.0f, 1.0f - 2 * height_diff / height_range);
    } else {
        height_factor = 0.0f;
    }
    
    // 前方因子：手在身体前方
    float front_factor = features.hands_in_front ? 1.0f : 0.5f;
    
    return confidence * distance_factor * position_factor * height_factor * front_factor;
}

void BallRequestDetector::update_state_machine(RequestContext& context, bool is_requesting, float confidence) {
    auto now = std::chrono::steady_clock::now();
    
    switch (context.state) {
        case BallRequestState::NO_REQUEST:
            if (is_requesting && confidence > 0.3f) {
                transition_to_potential_request(context, confidence);
            }
            break;
            
        case BallRequestState::POTENTIAL_REQUEST:
            if (is_requesting) {
                context.continuous_frames++;
                context.total_frames++;
                context.accumulated_confidence += confidence;
                context.max_confidence = std::max(context.max_confidence, confidence);
                context.interruption_frames = 0;
                context.last_detected_time = now;
                
                // 检查是否满足确认条件
                if (validate_confirmation_conditions(context)) {
                    transition_to_confirmed_request(context);
                }
            } else {
                // 允许短暂中断
                context.interruption_frames++;
                if (context.interruption_frames > config_.max_interruption_frames) {
                    // 中断太久，重置状态
                    transition_to_no_request(context);
                }
            }
            break;
            
        case BallRequestState::CONFIRMED_REQUEST:
            if (is_requesting) {
                context.continuous_frames++;
                context.total_frames++;
                context.accumulated_confidence += confidence;
                context.interruption_frames = 0;
                context.last_detected_time = now;
                context.last_confirmed_time = now;
            } else {
                context.interruption_frames++;
                if (context.interruption_frames > config_.max_interruption_frames) {
                    transition_to_ending_request(context);
                }
            }
            break;
            
        case BallRequestState::ENDING_REQUEST:
            context.cooldown_frames++;
            if (context.cooldown_frames >= config_.cooldown_frames) {
                transition_to_no_request(context);
            } else if (is_requesting && confidence > 0.5f) {
                // 在冷却期内重新开始要球
                transition_to_potential_request(context, confidence);
            }
            break;
    }
}

bool BallRequestDetector::validate_confirmation_conditions(const RequestContext& context) const {
    // 基础条件：连续帧数和总置信度
    if (context.continuous_frames < config_.min_continuous_frames) {
        return false;
    }
    
    if (context.accumulated_confidence < config_.min_total_confidence) {
        return false;
    }
    
    // 手势稳定性检查
    float stability = calculate_gesture_stability(context);
    if (stability > config_.gesture_stability_threshold) {
        return false;
    }
    
    // 置信度趋势检查：确保置信度不是在下降
    float confidence_trend = calculate_confidence_trend(context);
    if (confidence_trend < -0.1f) {  // 置信度显著下降
        return false;
    }
    
    return true;
}

void BallRequestDetector::transition_to_potential_request(RequestContext& context, float confidence) {
    context.state = BallRequestState::POTENTIAL_REQUEST;
    context.continuous_frames = 1;
    context.total_frames = 1;
    context.interruption_frames = 0;
    context.cooldown_frames = 0;
    context.accumulated_confidence = confidence;
    context.max_confidence = confidence;
    context.request_start_time = std::chrono::steady_clock::now();
    context.last_detected_time = context.request_start_time;
    
    // 重置滤波器
    if (context.confidence_filter) {
        context.confidence_filter->reset();
    }
    if (context.position_filter) {
        context.position_filter->reset();
    }
}

void BallRequestDetector::transition_to_confirmed_request(RequestContext& context) {
    context.state = BallRequestState::CONFIRMED_REQUEST;
    context.last_confirmed_time = std::chrono::steady_clock::now();
    // 保持其他计数器继续累积
}

void BallRequestDetector::transition_to_ending_request(RequestContext& context) {
    context.state = BallRequestState::ENDING_REQUEST;
    context.cooldown_frames = 0;
}

void BallRequestDetector::transition_to_no_request(RequestContext& context) {
    context.state = BallRequestState::NO_REQUEST;
    context.continuous_frames = 0;
    context.total_frames = 0;
    context.interruption_frames = 0;
    context.cooldown_frames = 0;
    context.accumulated_confidence = 0.0f;
    context.max_confidence = 0.0f;
    
    // 清空缓冲区
    context.hands_distance_buffer.clear();
    context.left_wrist_buffer.clear();
    context.right_wrist_buffer.clear();
    context.confidence_buffer.clear();
}

BallRequestDetector::RequestContext& BallRequestDetector::get_or_create_context(int person_id) {
    auto it = person_contexts_.find(person_id);
    if (it == person_contexts_.end()) {
        person_contexts_[person_id] = std::make_unique<RequestContext>(person_id, config_);
    }
    return *person_contexts_[person_id];
}

cv::Point2f BallRequestDetector::pixel_to_world(const cv::Point2f& pixel_point) const {
    if (!has_homography_) {
        return pixel_point;  // 直接返回像素坐标
    }
    
    std::vector<cv::Point2f> src_points = {pixel_point};
    std::vector<cv::Point2f> dst_points;
    cv::perspectiveTransform(src_points, dst_points, homography_matrix_);
    
    return dst_points[0];
}

float BallRequestDetector::calculate_world_distance(const cv::Point2f& p1, const cv::Point2f& p2) const {
    cv::Point2f world_p1 = pixel_to_world(p1);
    cv::Point2f world_p2 = pixel_to_world(p2);
    
    float dx = world_p1.x - world_p2.x;
    float dy = world_p1.y - world_p2.y;
    
    return std::sqrt(dx * dx + dy * dy);
}

bool BallRequestDetector::is_keypoint_valid(const PoseResult& pose, COCOKeypoint keypoint) const {
    int idx = static_cast<int>(keypoint);
    if (idx >= pose.keypoint_confidences.size()) return false;
    
    return pose.keypoint_confidences[idx] > config_.min_keypoint_confidence;
}

cv::Point2f BallRequestDetector::get_keypoint_position(const PoseResult& pose, COCOKeypoint keypoint) const {
    int idx = static_cast<int>(keypoint);
    if (idx >= pose.keypoints.size()) return cv::Point2f(-1, -1);
    
    return pose.keypoints[idx];
}

cv::Point2f BallRequestDetector::calculate_chest_center(const PoseResult& pose) const {
    // 尝试使用肩膀关键点计算胸部中心
    bool left_shoulder_valid = is_keypoint_valid(pose, COCOKeypoint::LEFT_SHOULDER);
    bool right_shoulder_valid = is_keypoint_valid(pose, COCOKeypoint::RIGHT_SHOULDER);
    
    if (left_shoulder_valid && right_shoulder_valid) {
        cv::Point2f left_shoulder = get_keypoint_position(pose, COCOKeypoint::LEFT_SHOULDER);
        cv::Point2f right_shoulder = get_keypoint_position(pose, COCOKeypoint::RIGHT_SHOULDER);
        cv::Point2f shoulder_center = (left_shoulder + right_shoulder) * 0.5f;
        
        // 胸部中心稍微低于肩膀中心
        shoulder_center.y += pose.bbox.height * 0.15f;  // 向下偏移15%的身体高度
        return shoulder_center;
    } else if (left_shoulder_valid) {
        cv::Point2f left_shoulder = get_keypoint_position(pose, COCOKeypoint::LEFT_SHOULDER);
        left_shoulder.y += pose.bbox.height * 0.15f;
        return left_shoulder;
    } else if (right_shoulder_valid) {
        cv::Point2f right_shoulder = get_keypoint_position(pose, COCOKeypoint::RIGHT_SHOULDER);
        right_shoulder.y += pose.bbox.height * 0.15f;
        return right_shoulder;
    } else {
        // 使用bbox中心估算
        return cv::Point2f(pose.bbox.x + pose.bbox.width * 0.5f, 
                          pose.bbox.y + pose.bbox.height * 0.4f);
    }
}

bool BallRequestDetector::is_point_in_front_of_chest(const cv::Point2f& hand_pos,
                                                   const cv::Point2f& chest_center,
                                                   const PoseResult& pose) const {
    // 简单的前方判定：手的x坐标在胸部中心附近，y坐标在合理范围内
    float x_distance = std::abs(hand_pos.x - chest_center.x);
    float y_distance = hand_pos.y - chest_center.y;
    
    // 手应该在胸部水平范围内，且稍微在前方（y方向稍微偏移）
    float max_x_offset = pose.bbox.width * 0.6f;  // 允许60%的身体宽度偏移
    float max_y_offset_above = pose.bbox.height * 0.1f;  // 允许向上10%偏移
    float max_y_offset_below = pose.bbox.height * 0.3f;  // 允许向下30%偏移
    
    return (x_distance <= max_x_offset) &&
           (y_distance >= -max_y_offset_above) &&
           (y_distance <= max_y_offset_below);
}

float BallRequestDetector::calculate_confidence_trend(const RequestContext& context) const {
    auto confidence_history = context.confidence_buffer.get_window(5);
    if (confidence_history.size() < 3) {
        return 0.0f;
    }
    
    // 计算线性趋势 (简单的斜率计算)
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    int n = confidence_history.size();
    
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i);
        float y = confidence_history[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

float BallRequestDetector::calculate_gesture_consistency(const RequestContext& context) const {
    auto distance_history = context.hands_distance_buffer.get_window(10);
    if (distance_history.size() < 3) {
        return 0.0f;
    }
    
    return 1.0f - BufferStatistics::coefficient_of_variation(distance_history);
}

void BallRequestDetector::update_config(const BallRequestConfig& config) {
    config_ = config;
    
    // 重新创建所有person的滤波器
    for (auto& pair : person_contexts_) {
        auto& context = *pair.second;
        
        if (config.smooth_confidence) {
            context.confidence_filter = FilterFactory::create_moving_average_filter(
                5, config.confidence_smoothing_alpha);
        }
        
        context.position_filter = FilterFactory::create_moving_average_filter(
            3, config.position_smoothing_alpha);
    }
}

void BallRequestDetector::reset() {
    person_contexts_.clear();
    frame_counter_ = 0;
}

void BallRequestDetector::reset_person(int person_id) {
    auto it = person_contexts_.find(person_id);
    if (it != person_contexts_.end()) {
        transition_to_no_request(*it->second);
    }
}

void BallRequestDetector::cleanup_stale_contexts(int max_age_ms) {
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = person_contexts_.begin(); it != person_contexts_.end();) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second->last_detected_time).count();
        
        if (elapsed > max_age_ms) {
            it = person_contexts_.erase(it);
        } else {
            ++it;
        }
    }
}

std::string BallRequestDetector::get_debug_info(int person_id) const {
    std::stringstream ss;
    
    if (person_id >= 0) {
        auto it = person_contexts_.find(person_id);
        if (it != person_contexts_.end()) {
            const auto& context = *it->second;
            ss << "Person " << person_id << ":\n";
            ss << "  State: " << static_cast<int>(context.state) << "\n";
            ss << "  Continuous frames: " << context.continuous_frames << "\n";
            ss << "  Total frames: " << context.total_frames << "\n";
            ss << "  Interruption frames: " << context.interruption_frames << "\n";
            ss << "  Accumulated confidence: " << context.accumulated_confidence << "\n";
            ss << "  Max confidence: " << context.max_confidence << "\n";
            ss << "  Gesture stability: " << calculate_gesture_stability(context) << "\n";
        }
    } else {
        ss << "Ball Request Detector Status:\n";
        ss << "  Active persons: " << person_contexts_.size() << "\n";
        ss << "  Frame counter: " << frame_counter_ << "\n";
        ss << "  Has homography: " << (has_homography_ ? "Yes" : "No") << "\n";
        
        for (const auto& pair : person_contexts_) {
            const auto& context = *pair.second;
            ss << "  Person " << pair.first << ": state=" << static_cast<int>(context.state) 
               << ", frames=" << context.continuous_frames << "\n";
        }
    }
    
    return ss.str();
}

std::vector<int> BallRequestDetector::get_active_person_ids() const {
    std::vector<int> ids;
    for (const auto& pair : person_contexts_) {
        ids.push_back(pair.first);
    }
    return ids;
}

std::vector<int> BallRequestDetector::get_requesting_person_ids() const {
    std::vector<int> ids;
    for (const auto& pair : person_contexts_) {
        if (pair.second->state == BallRequestState::CONFIRMED_REQUEST ||
            pair.second->state == BallRequestState::POTENTIAL_REQUEST) {
            ids.push_back(pair.first);
        }
    }
    return ids;
}

} // namespace pose_analysis