#include "id_priority_manager.h"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <unordered_set>

namespace pose_analysis {

IDPriorityManager::IDPriorityManager(const IDManagementConfig& config)
    : config_(config), next_available_priority_id_(1),
      last_priority_update_(std::chrono::steady_clock::now()) {
}

std::vector<IDPriorityResult> IDPriorityManager::update(const std::vector<BallRequestResult>& ball_requests) {
    auto now = std::chrono::steady_clock::now();
    std::vector<IDPriorityResult> results;
    
    // 更新每个person的跟踪信息
    for (const auto& request : ball_requests) {
        if (request.person_id >= 0) {
            auto& info = get_or_create_tracking_info(request.person_id);
            update_tracking_info(info, request);
        }
    }
    
    // 检查是否需要更新优先级
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_priority_update_).count();
    
    if (elapsed >= config_.priority_update_interval_ms) {
        process_swap_requests();
        last_priority_update_ = now;
    }
    
    // 生成结果
    for (const auto& request : ball_requests) {
        if (request.person_id >= 0) {
            IDPriorityResult result;
            result.person_id = request.person_id;
            
            auto it = bytetrack_to_priority_.find(request.person_id);
            if (it != bytetrack_to_priority_.end()) {
                result.priority_id = it->second;
            } else {
                result.priority_id = allocate_new_priority_id();
                bytetrack_to_priority_[request.person_id] = result.priority_id;
                priority_to_bytetrack_[result.priority_id] = request.person_id;
            }
            
            auto& info = get_or_create_tracking_info(request.person_id);
            result.priority_score = calculate_priority_score(info, request);
            result.can_swap_id = can_swap(request.person_id);
            result.last_swap_time = info.last_swap_time;
            
            results.push_back(result);
        }
    }
    
    // 清理过期的跟踪信息
    cleanup_stale_tracking_info();
    
    // 验证结果的合理性
    if (!validate_priority_rules(results)) {
        // 如果规则验证失败，重置所有映射
        reset_all_mappings();
        return update(ball_requests);  // 重新处理
    }
    
    return results;
}

void IDPriorityManager::register_new_person(int bytetrack_id) {
    if (bytetrack_to_priority_.find(bytetrack_id) != bytetrack_to_priority_.end()) {
        return;  // 已经注册过
    }
    
    int priority_id = allocate_new_priority_id();
    bytetrack_to_priority_[bytetrack_id] = priority_id;
    priority_to_bytetrack_[priority_id] = bytetrack_id;
    
    // 创建跟踪信息
    get_or_create_tracking_info(bytetrack_id);
}

void IDPriorityManager::unregister_person(int bytetrack_id) {
    auto it = bytetrack_to_priority_.find(bytetrack_id);
    if (it != bytetrack_to_priority_.end()) {
        int priority_id = it->second;
        
        // 移除映射
        bytetrack_to_priority_.erase(bytetrack_id);
        priority_to_bytetrack_.erase(priority_id);
        
        // 移除跟踪信息
        tracking_info_.erase(bytetrack_id);
        
        // 释放优先级ID以供重用 (如果配置允许)
        if (!config_.preserve_original_order) {
            release_priority_id(priority_id);
        }
    }
}

std::unordered_map<int, int> IDPriorityManager::get_bytetrack_to_priority_mapping() const {
    return bytetrack_to_priority_;
}

std::unordered_map<int, int> IDPriorityManager::get_priority_to_bytetrack_mapping() const {
    return priority_to_bytetrack_;
}

std::vector<int> IDPriorityManager::get_requesting_priority_ids() const {
    std::vector<int> requesting_ids;
    
    for (const auto& pair : tracking_info_) {
        const auto& info = *pair.second;
        if (info.is_confirmed_requesting) {
            auto it = bytetrack_to_priority_.find(pair.first);
            if (it != bytetrack_to_priority_.end()) {
                requesting_ids.push_back(it->second);
            }
        }
    }
    
    // 按优先级分数排序
    std::sort(requesting_ids.begin(), requesting_ids.end(),
              [this](int id1, int id2) {
                  auto bt1_it = priority_to_bytetrack_.find(id1);
                  auto bt2_it = priority_to_bytetrack_.find(id2);
                  
                  if (bt1_it == priority_to_bytetrack_.end() || 
                      bt2_it == priority_to_bytetrack_.end()) {
                      return id1 < id2;
                  }
                  
                  auto info1_it = tracking_info_.find(bt1_it->second);
                  auto info2_it = tracking_info_.find(bt2_it->second);
                  
                  if (info1_it == tracking_info_.end() || 
                      info2_it == tracking_info_.end()) {
                      return id1 < id2;
                  }
                  
                  return info1_it->second->request_priority > info2_it->second->request_priority;
              });
    
    return requesting_ids;
}

float IDPriorityManager::calculate_priority_score(const PersonTrackingInfo& info,
                                                 const BallRequestResult& request) const {
    if (!request.is_confirmed) {
        return 0.0f;  // 未确认的要球动作不参与优先级计算
    }
    
    // 基础分数来自请求的置信度
    float base_score = request.request_confidence;
    
    // 持续时间因子 (要球时间越长优先级越高)
    float duration_factor = 1.0f;
    if (request.request_duration_ms > 0) {
        float duration_seconds = request.request_duration_ms / 1000.0f;
        duration_factor = std::min(2.0f, 1.0f + duration_seconds / 5.0f);  // 最多2倍加成
    }
    
    // 稳定性因子 (手势越稳定优先级越高)
    float stability_factor = 1.0f;
    if (request.gesture_stability >= 0 && request.gesture_stability <= 1.0f) {
        stability_factor = 1.0f + (1.0f - request.gesture_stability);  // 稳定性越高因子越大
    }
    
    // 连续性因子 (连续帧数越多优先级越高)
    float continuity_factor = 1.0f;
    if (request.continuous_frames > 0) {
        continuity_factor = std::min(1.5f, 1.0f + request.continuous_frames / 20.0f);
    }
    
    // 历史表现因子 (考虑历史要球记录)
    float history_factor = calculate_request_stability(info);
    
    // 加权计算最终分数
    float final_score = base_score * 
                       (config_.confidence_weight * 1.0f +
                        config_.duration_weight * duration_factor +
                        config_.stability_weight * stability_factor) *
                       continuity_factor * history_factor;
    
    // 应用衰减 (防止长期占据高优先级)
    auto now = std::chrono::steady_clock::now();
    auto request_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - request.request_start_time).count();
    
    if (request_age_ms > 10000) {  // 10秒后开始衰减
        float decay_factor = std::pow(config_.decay_rate, (request_age_ms - 10000) / 1000.0f);
        final_score *= decay_factor;
    }
    
    return std::max(0.0f, final_score);
}

void IDPriorityManager::process_swap_requests() {
    if (!config_.allow_multiple_requesters && pending_swaps_.size() > 1) {
        // 只允许一个要球者，保留优先级最高的请求
        std::vector<IDSwapRequest> valid_requests;
        while (!pending_swaps_.empty()) {
            valid_requests.push_back(pending_swaps_.top());
            pending_swaps_.pop();
        }
        
        if (!valid_requests.empty()) {
            // 只保留分数最高的请求
            std::sort(valid_requests.begin(), valid_requests.end(),
                     [](const IDSwapRequest& a, const IDSwapRequest& b) {
                         return a.priority_score > b.priority_score;
                     });
            
            pending_swaps_.push(valid_requests[0]);
        }
    }
    
    // 处理队列中的交换请求
    std::vector<IDSwapRequest> failed_requests;
    
    while (!pending_swaps_.empty() && failed_requests.size() < 3) {  // 最多重试3次
        IDSwapRequest request = pending_swaps_.top();
        pending_swaps_.pop();
        
        if (validate_swap_conditions(request)) {
            if (execute_swap(request)) {
                // 交换成功，继续处理下一个
                continue;
            }
        }
        
        // 交换失败，加入失败列表
        failed_requests.push_back(request);
    }
    
    // 将失败的请求重新加入队列 (降低优先级)
    for (auto& failed : failed_requests) {
        failed.priority_score *= 0.9f;  // 降低10%优先级
        if (failed.priority_score > 0.1f) {
            pending_swaps_.push(failed);
        }
    }
}

bool IDPriorityManager::execute_swap(const IDSwapRequest& request) {
    auto requester_it = tracking_info_.find(request.requester_id);
    if (requester_it == tracking_info_.end()) {
        return false;
    }
    
    auto& requester_info = *requester_it->second;
    
    // 查找当前持有目标优先级ID的person
    auto target_bt_it = priority_to_bytetrack_.find(request.target_priority_id);
    if (target_bt_it == priority_to_bytetrack_.end()) {
        return false;
    }
    
    int target_bytetrack_id = target_bt_it->second;
    auto target_info_it = tracking_info_.find(target_bytetrack_id);
    if (target_info_it == tracking_info_.end()) {
        return false;
    }
    
    auto& target_info = *target_info_it->second;
    
    // 检查防抖动条件
    if (!should_trigger_swap(requester_info, target_info)) {
        return false;
    }
    
    // 获取请求者当前的优先级ID
    auto requester_priority_it = bytetrack_to_priority_.find(request.requester_id);
    if (requester_priority_it == bytetrack_to_priority_.end()) {
        return false;
    }
    
    int requester_current_priority = requester_priority_it->second;
    
    // 执行交换
    bytetrack_to_priority_[request.requester_id] = request.target_priority_id;
    bytetrack_to_priority_[target_bytetrack_id] = requester_current_priority;
    
    priority_to_bytetrack_[request.target_priority_id] = request.requester_id;
    priority_to_bytetrack_[requester_current_priority] = target_bytetrack_id;
    
    // 更新交换时间
    auto now = std::chrono::steady_clock::now();
    requester_info.last_swap_time = now;
    target_info.last_swap_time = now;
    
    // 更新跟踪信息中的当前优先级ID
    requester_info.current_priority_id = request.target_priority_id;
    target_info.current_priority_id = requester_current_priority;
    
    return true;
}

bool IDPriorityManager::validate_swap_conditions(const IDSwapRequest& request) const {
    // 检查请求者是否仍然有效
    auto requester_it = tracking_info_.find(request.requester_id);
    if (requester_it == tracking_info_.end()) {
        return false;
    }
    
    const auto& requester_info = *requester_it->second;
    
    // 检查请求者是否仍在要球
    if (!requester_info.is_confirmed_requesting) {
        return false;
    }
    
    // 检查冷却时间
    if (is_in_cooldown(requester_info)) {
        return false;
    }
    
    // 检查请求的稳定性
    if (!is_request_stable(requester_info)) {
        return false;
    }
    
    // 检查请求帧数是否足够
    if (requester_info.stable_request_frames < config_.min_request_frames_for_swap) {
        return false;
    }
    
    // 检查目标优先级ID是否存在
    auto target_it = priority_to_bytetrack_.find(request.target_priority_id);
    if (target_it == priority_to_bytetrack_.end()) {
        return false;
    }
    
    return true;
}

bool IDPriorityManager::can_swap(int bytetrack_id) const {
    auto it = tracking_info_.find(bytetrack_id);
    if (it == tracking_info_.end()) {
        return false;
    }
    
    const auto& info = *it->second;
    
    // 检查冷却时间
    return !is_in_cooldown(info) && is_request_stable(info);
}

void IDPriorityManager::update_tracking_info(PersonTrackingInfo& info, const BallRequestResult& request) {
    auto now = std::chrono::steady_clock::now();
    
    // 更新要球状态
    info.is_requesting = request.is_requesting;
    info.is_confirmed_requesting = request.is_confirmed;
    
    if (request.is_requesting) {
        info.last_request_time = now;
    }
    
    // 计算并平滑优先级分数
    float raw_priority = calculate_priority_score(info, request);
    info.request_priority = smooth_priority_score(info, raw_priority);
    
    // 更新稳定要球帧数
    if (request.is_confirmed) {
        info.stable_request_frames++;
    } else {
        info.stable_request_frames = 0;
    }
    
    // 更新历史记录
    info.priority_history.push(info.request_priority);
    info.request_history.push(request.is_confirmed);
    
    // 如果优先级足够高，添加交换请求
    if (request.is_confirmed && info.request_priority > 1.0f && can_swap(info.bytetrack_id)) {
        // 查找当前优先级最高的ID (通常是ID=1)
        int target_priority_id = 1;
        
        // 检查是否需要发起交换请求
        auto current_priority_it = bytetrack_to_priority_.find(info.bytetrack_id);
        if (current_priority_it != bytetrack_to_priority_.end() &&
            current_priority_it->second > target_priority_id) {
            
            IDSwapRequest swap_request(info.bytetrack_id, target_priority_id, info.request_priority);
            add_swap_request(swap_request);
        }
    }
}

void IDPriorityManager::add_swap_request(const IDSwapRequest& request) {
    // 避免重复的交换请求
    std::priority_queue<IDSwapRequest> temp_queue;
    bool request_exists = false;
    
    while (!pending_swaps_.empty()) {
        IDSwapRequest existing = pending_swaps_.top();
        pending_swaps_.pop();
        
        if (existing.requester_id == request.requester_id &&
            existing.target_priority_id == request.target_priority_id) {
            // 更新现有请求的分数
            existing.priority_score = std::max(existing.priority_score, request.priority_score);
            existing.last_update_time = std::chrono::steady_clock::now();
            temp_queue.push(existing);
            request_exists = true;
        } else {
            temp_queue.push(existing);
        }
    }
    
    // 恢复队列
    pending_swaps_ = std::move(temp_queue);
    
    // 如果是新请求，添加到队列
    if (!request_exists && pending_swaps_.size() < config_.max_pending_swaps) {
        pending_swaps_.push(request);
    }
}

IDPriorityManager::PersonTrackingInfo& IDPriorityManager::get_or_create_tracking_info(int bytetrack_id) {
    auto it = tracking_info_.find(bytetrack_id);
    if (it == tracking_info_.end()) {
        // 获取或分配优先级ID
        int priority_id;
        auto priority_it = bytetrack_to_priority_.find(bytetrack_id);
        if (priority_it != bytetrack_to_priority_.end()) {
            priority_id = priority_it->second;
        } else {
            priority_id = allocate_new_priority_id();
            bytetrack_to_priority_[bytetrack_id] = priority_id;
            priority_to_bytetrack_[priority_id] = bytetrack_id;
        }
        
        tracking_info_[bytetrack_id] = std::make_unique<PersonTrackingInfo>(bytetrack_id, priority_id, config_);
    }
    
    return *tracking_info_[bytetrack_id];
}

int IDPriorityManager::allocate_new_priority_id() {
    // 查找最小的可用优先级ID
    for (int id = 1; id <= config_.max_tracked_persons; ++id) {
        if (is_priority_id_available(id)) {
            return id;
        }
    }
    
    // 如果没有可用的ID，返回下一个序号
    return next_available_priority_id_++;
}

void IDPriorityManager::release_priority_id(int priority_id) {
    // 在preserve_original_order模式下不释放ID
    if (config_.preserve_original_order) {
        return;
    }
    
    // 确保优先级ID被释放
    auto it = priority_to_bytetrack_.find(priority_id);
    if (it != priority_to_bytetrack_.end()) {
        priority_to_bytetrack_.erase(it);
    }
}

bool IDPriorityManager::is_priority_id_available(int priority_id) const {
    return priority_to_bytetrack_.find(priority_id) == priority_to_bytetrack_.end();
}

bool IDPriorityManager::is_in_cooldown(const PersonTrackingInfo& info) const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - info.last_swap_time).count();
    
    return elapsed < config_.swap_cooldown_ms;
}

bool IDPriorityManager::is_request_stable(const PersonTrackingInfo& info) const {
    return info.stable_request_frames >= config_.min_request_frames_for_swap &&
           calculate_request_stability(info) > 0.7f;  // 稳定性阈值
}

float IDPriorityManager::smooth_priority_score(PersonTrackingInfo& info, float raw_score) const {
    auto recent_scores = info.priority_history.get_window(5);
    if (recent_scores.empty()) {
        return raw_score;
    }
    
    // 使用指数加权移动平均进行平滑
    float alpha = 0.7f;
    float smoothed = recent_scores.back();
    smoothed = alpha * raw_score + (1 - alpha) * smoothed;
    
    return smoothed;
}

bool IDPriorityManager::should_trigger_swap(const PersonTrackingInfo& requester, 
                                          const PersonTrackingInfo& current_holder) const {
    // 基础条件：请求者优先级必须明显高于当前持有者
    float priority_diff = requester.request_priority - current_holder.request_priority;
    if (priority_diff < 0.3f) {  // 至少要高0.3分
        return false;
    }
    
    // 滞后阈值：防止频繁切换
    if (current_holder.is_confirmed_requesting && priority_diff < 0.5f) {
        return false;  // 当前持有者还在要球，需要更大的差距
    }
    
    // 时间因子：要球时间越长，越难被替换
    auto now = std::chrono::steady_clock::now();
    auto current_holder_duration = std::chrono::duration_cast<std::chrono::seconds>(
        now - current_holder.last_request_time).count();
    
    if (current_holder_duration < 2) {  // 要球不到2秒的给予保护
        return priority_diff > 1.0f;  // 需要非常大的优先级差距
    }
    
    return true;
}

float IDPriorityManager::calculate_request_stability(const PersonTrackingInfo& info) const {
    auto recent_requests = info.request_history.get_recent_data();
    if (recent_requests.size() < 3) {
        return 0.0f;
    }
    
    // 计算最近请求的稳定性 (确认要球的比例)
    int confirmed_count = 0;
    for (bool confirmed : recent_requests) {
        if (confirmed) confirmed_count++;
    }
    
    return static_cast<float>(confirmed_count) / recent_requests.size();
}

float IDPriorityManager::calculate_priority_trend(const PersonTrackingInfo& info) const {
    auto recent_priorities = info.priority_history.get_window(5);
    if (recent_priorities.size() < 3) {
        return 0.0f;
    }
    
    // 计算线性趋势
    float sum_x = 0.0f, sum_y = 0.0f, sum_xy = 0.0f, sum_x2 = 0.0f;
    int n = recent_priorities.size();
    
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i);
        float y = recent_priorities[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    float slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return slope;
}

bool IDPriorityManager::force_swap_ids(int bytetrack_id1, int bytetrack_id2) {
    auto it1 = bytetrack_to_priority_.find(bytetrack_id1);
    auto it2 = bytetrack_to_priority_.find(bytetrack_id2);
    
    if (it1 == bytetrack_to_priority_.end() || it2 == bytetrack_to_priority_.end()) {
        return false;
    }
    
    int priority_id1 = it1->second;
    int priority_id2 = it2->second;
    
    // 交换映射
    bytetrack_to_priority_[bytetrack_id1] = priority_id2;
    bytetrack_to_priority_[bytetrack_id2] = priority_id1;
    priority_to_bytetrack_[priority_id1] = bytetrack_id2;
    priority_to_bytetrack_[priority_id2] = bytetrack_id1;
    
    // 更新跟踪信息
    auto info1_it = tracking_info_.find(bytetrack_id1);
    auto info2_it = tracking_info_.find(bytetrack_id2);
    
    if (info1_it != tracking_info_.end()) {
        info1_it->second->current_priority_id = priority_id2;
        info1_it->second->last_swap_time = std::chrono::steady_clock::now();
    }
    
    if (info2_it != tracking_info_.end()) {
        info2_it->second->current_priority_id = priority_id1;
        info2_it->second->last_swap_time = std::chrono::steady_clock::now();
    }
    
    return true;
}

void IDPriorityManager::reset_all_mappings() {
    bytetrack_to_priority_.clear();
    priority_to_bytetrack_.clear();
    tracking_info_.clear();
    
    // 清空交换请求队列
    while (!pending_swaps_.empty()) {
        pending_swaps_.pop();
    }
    
    next_available_priority_id_ = 1;
}

void IDPriorityManager::update_config(const IDManagementConfig& config) {
    config_ = config;
}

void IDPriorityManager::cleanup_stale_tracking_info(int max_age_ms) {
    auto now = std::chrono::steady_clock::now();
    
    for (auto it = tracking_info_.begin(); it != tracking_info_.end();) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - it->second->last_request_time).count();
        
        if (elapsed > max_age_ms && !it->second->is_requesting) {
            // 移除过期且不活跃的跟踪信息
            unregister_person(it->first);
            it = tracking_info_.begin();  // 重新开始迭代
        } else {
            ++it;
        }
    }
}

bool IDPriorityManager::validate_priority_rules(const std::vector<IDPriorityResult>& results) const {
    std::unordered_set<int> used_priority_ids;
    
    for (const auto& result : results) {
        // 检查优先级ID是否重复
        if (used_priority_ids.find(result.priority_id) != used_priority_ids.end()) {
            return false;  // 发现重复的优先级ID
        }
        used_priority_ids.insert(result.priority_id);
        
        // 检查映射的一致性
        auto bt_to_priority_it = bytetrack_to_priority_.find(result.person_id);
        if (bt_to_priority_it == bytetrack_to_priority_.end() ||
            bt_to_priority_it->second != result.priority_id) {
            return false;  // 映射不一致
        }
    }
    
    return true;
}

std::string IDPriorityManager::get_debug_info() const {
    std::stringstream ss;
    ss << "ID Priority Manager Status:\n";
    ss << "  Active persons: " << tracking_info_.size() << "\n";
    ss << "  Pending swaps: " << pending_swaps_.size() << "\n";
    ss << "  Next priority ID: " << next_available_priority_id_ << "\n";
    
    ss << "\nID Mappings (ByteTrack -> Priority):\n";
    for (const auto& pair : bytetrack_to_priority_) {
        ss << "  " << pair.first << " -> " << pair.second;
        
        auto info_it = tracking_info_.find(pair.first);
        if (info_it != tracking_info_.end()) {
            const auto& info = *info_it->second;
            ss << " (requesting: " << (info.is_confirmed_requesting ? "YES" : "NO")
               << ", priority: " << info.request_priority << ")";
        }
        ss << "\n";
    }
    
    ss << "\nRequesting Priority IDs: ";
    auto requesting_ids = get_requesting_priority_ids();
    for (size_t i = 0; i < requesting_ids.size(); ++i) {
        ss << requesting_ids[i];
        if (i < requesting_ids.size() - 1) ss << ", ";
    }
    ss << "\n";
    
    return ss.str();
}

std::vector<int> IDPriorityManager::get_active_bytetrack_ids() const {
    std::vector<int> ids;
    for (const auto& pair : tracking_info_) {
        ids.push_back(pair.first);
    }
    return ids;
}

} // namespace pose_analysis