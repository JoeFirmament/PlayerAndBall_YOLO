#pragma once

#include "pose_analysis_types.h"
#include "temporal_buffer.h"
#include <unordered_map>
#include <queue>
#include <vector>
#include <chrono>

namespace pose_analysis {

class IDPriorityManager {
private:
    // ID交换请求
    struct IDSwapRequest {
        int requester_id;           // 请求者的ByteTrack ID
        int target_priority_id;     // 目标的优先级ID
        float priority_score;       // 优先级分数
        int request_frames;         // 请求持续帧数
        std::chrono::steady_clock::time_point request_time;
        std::chrono::steady_clock::time_point last_update_time;
        
        IDSwapRequest(int req_id, int target_id, float score)
            : requester_id(req_id), target_priority_id(target_id), priority_score(score),
              request_frames(0), request_time(std::chrono::steady_clock::now()),
              last_update_time(std::chrono::steady_clock::now()) {}
        
        // 用于优先队列排序 (分数越高优先级越高)
        bool operator<(const IDSwapRequest& other) const {
            return priority_score < other.priority_score;
        }
    };
    
    // Person的跟踪信息
    struct PersonTrackingInfo {
        int bytetrack_id;           // ByteTrack分配的原始ID
        int current_priority_id;    // 当前的优先级ID
        int original_priority_id;   // 最初的优先级ID
        
        // 要球状态
        bool is_requesting;         // 是否正在要球
        bool is_confirmed_requesting; // 是否确认要球
        float request_priority;     // 要球优先级分数
        int stable_request_frames;  // 稳定要球的帧数
        
        // 时序信息
        std::chrono::steady_clock::time_point first_seen_time;
        std::chrono::steady_clock::time_point last_request_time;
        std::chrono::steady_clock::time_point last_swap_time;
        
        // 优先级历史
        TemporalBuffer<float> priority_history;
        TimestampedBuffer<bool> request_history;
        
        PersonTrackingInfo(int bt_id, int priority_id, const IDManagementConfig& config)
            : bytetrack_id(bt_id), current_priority_id(priority_id), original_priority_id(priority_id),
              is_requesting(false), is_confirmed_requesting(false), request_priority(0.0f),
              stable_request_frames(0), first_seen_time(std::chrono::steady_clock::now()),
              last_request_time(std::chrono::steady_clock::now()),
              last_swap_time(std::chrono::steady_clock::now()),
              priority_history(20), request_history(50, 5000) {  // 5秒的请求历史
        }
    };
    
    // 配置参数
    IDManagementConfig config_;
    
    // 跟踪信息映射
    std::unordered_map<int, std::unique_ptr<PersonTrackingInfo>> tracking_info_;
    
    // ID映射关系
    std::unordered_map<int, int> bytetrack_to_priority_;  // ByteTrack ID -> 优先级 ID
    std::unordered_map<int, int> priority_to_bytetrack_;  // 优先级 ID -> ByteTrack ID
    
    // 待处理的交换请求队列
    std::priority_queue<IDSwapRequest> pending_swaps_;
    
    // 状态管理
    int next_available_priority_id_;
    std::chrono::steady_clock::time_point last_priority_update_;
    
public:
    explicit IDPriorityManager(const IDManagementConfig& config);
    ~IDPriorityManager() = default;
    
    // 更新优先级管理 (主要接口)
    std::vector<IDPriorityResult> update(const std::vector<BallRequestResult>& ball_requests);
    
    // 处理新出现的person
    void register_new_person(int bytetrack_id);
    
    // 处理消失的person
    void unregister_person(int bytetrack_id);
    
    // 获取ID映射关系
    std::unordered_map<int, int> get_bytetrack_to_priority_mapping() const;
    std::unordered_map<int, int> get_priority_to_bytetrack_mapping() const;
    
    // 获取当前要球者的优先级ID列表
    std::vector<int> get_requesting_priority_ids() const;
    
    // 强制交换两个ID的优先级
    bool force_swap_ids(int bytetrack_id1, int bytetrack_id2);
    
    // 重置所有ID映射
    void reset_all_mappings();
    
    // 配置管理
    const IDManagementConfig& get_config() const { return config_; }
    void update_config(const IDManagementConfig& config);
    
    // 调试信息
    std::string get_debug_info() const;
    std::vector<int> get_active_bytetrack_ids() const;
    
    // 清理过期的跟踪信息
    void cleanup_stale_tracking_info(int max_age_ms = 10000);

private:
    // 核心优先级计算
    float calculate_priority_score(const PersonTrackingInfo& info,
                                  const BallRequestResult& request) const;
    
    // 处理交换请求
    void process_swap_requests();
    void add_swap_request(const IDSwapRequest& request);
    bool execute_swap(const IDSwapRequest& request);
    
    // 交换条件验证
    bool can_swap(int bytetrack_id) const;
    bool validate_swap_conditions(const IDSwapRequest& request) const;
    
    // ID分配和管理
    int allocate_new_priority_id();
    void release_priority_id(int priority_id);
    
    // 更新跟踪信息
    void update_tracking_info(PersonTrackingInfo& info, const BallRequestResult& request);
    
    // 优先级策略
    std::vector<int> calculate_new_priority_order(const std::vector<BallRequestResult>& requests);
    
    // 获取或创建跟踪信息
    PersonTrackingInfo& get_or_create_tracking_info(int bytetrack_id);
    
    // 冷却时间检查
    bool is_in_cooldown(const PersonTrackingInfo& info) const;
    
    // 稳定性检查
    bool is_request_stable(const PersonTrackingInfo& info) const;
    
    // 优先级分数的滤波和平滑
    float smooth_priority_score(PersonTrackingInfo& info, float raw_score) const;
    
    // 防抖动机制
    bool should_trigger_swap(const PersonTrackingInfo& requester, 
                           const PersonTrackingInfo& current_holder) const;
    
    // 时序分析工具
    float calculate_request_stability(const PersonTrackingInfo& info) const;
    float calculate_priority_trend(const PersonTrackingInfo& info) const;
    
    // ID映射维护
    void update_id_mappings();
    bool is_priority_id_available(int priority_id) const;
    
    // 规则验证
    bool validate_priority_rules(const std::vector<IDPriorityResult>& results) const;
};

} // namespace pose_analysis