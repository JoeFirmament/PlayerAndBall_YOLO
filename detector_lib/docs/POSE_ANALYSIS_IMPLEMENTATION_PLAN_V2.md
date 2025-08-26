# 姿态分析功能实施计划 V2.0
## 包含多帧滤波与时序判定机制

## 1. 功能需求（增强版）

### 1.1 身高检测功能
- **目标**: 检测person的真实身高（单位：毫米）
- **触发条件**: 仅当person的双手（手腕）不超过头部（头顶）时进行测量
- **多帧滤波机制**:
  - 连续N帧满足测量条件才开始测量
  - 使用滑动窗口记录M帧身高数据
  - 通过中值滤波或卡尔曼滤波获得稳定身高值
  - 支持异常值剔除（3σ原则）

### 1.2 篮球要球动作识别
- **目标**: 识别篮球比赛中的要球手势
- **判定条件**:
  - 双手手腕在胸部区域前方
  - 双手手腕距离小于设定阈值
  - 手臂呈现张开状态
- **多帧判定机制**:
  - 连续K帧满足要球条件才判定为要球
  - 支持动作中断容忍（允许L帧中断）
  - 动作置信度累积计算

### 1.3 要球者ID优先级管理
- **目标**: 当多人同时做要球动作时，动态调整跟踪ID
- **时序稳定性**:
  - ID切换需要持续要球T秒以上
  - 防止频繁切换的冷却机制
  - 支持优先级队列管理

## 2. 多帧滤波架构

### 2.1 时序数据管理器
```cpp
template<typename T>
class TemporalBuffer {
private:
    std::deque<T> buffer_;
    size_t max_size_;
    
public:
    TemporalBuffer(size_t max_size) : max_size_(max_size) {}
    
    void push(const T& data) {
        buffer_.push_back(data);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }
    
    std::vector<T> get_window(size_t n) const {
        size_t start = buffer_.size() > n ? buffer_.size() - n : 0;
        return std::vector<T>(buffer_.begin() + start, buffer_.end());
    }
    
    bool is_stable(size_t min_frames) const {
        return buffer_.size() >= min_frames;
    }
};
```

### 2.2 滤波器接口
```cpp
class IFilter {
public:
    virtual ~IFilter() = default;
    virtual float process(float input) = 0;
    virtual void reset() = 0;
    virtual bool is_stable() const = 0;
};

// 中值滤波器
class MedianFilter : public IFilter {
private:
    TemporalBuffer<float> buffer_;
    size_t window_size_;
    
public:
    MedianFilter(size_t window_size);
    float process(float input) override;
    bool is_stable() const override;
};

// 卡尔曼滤波器
class KalmanFilter1D : public IFilter {
private:
    float x_;  // 状态估计
    float P_;  // 估计误差协方差
    float Q_;  // 过程噪声协方差
    float R_;  // 测量噪声协方差
    
public:
    KalmanFilter1D(float Q, float R);
    float process(float measurement) override;
    bool is_stable() const override;
};

// 移动平均滤波器
class MovingAverageFilter : public IFilter {
private:
    TemporalBuffer<float> buffer_;
    size_t window_size_;
    float alpha_;  // 指数加权系数
    
public:
    MovingAverageFilter(size_t window_size, float alpha = 1.0);
    float process(float input) override;
};
```

## 3. 增强的实现细节

### 3.1 身高检测算法（带滤波）

#### 数据结构
```cpp
struct HeightMeasurement {
    float height_mm;
    int frame_id;
    float confidence;
    bool is_valid;
    std::chrono::steady_clock::time_point timestamp;
};

class HeightEstimator {
private:
    TemporalBuffer<HeightMeasurement> measurements_;
    std::unique_ptr<IFilter> height_filter_;
    std::unique_ptr<IFilter> confidence_filter_;
    
    // 状态机
    enum State {
        IDLE,           // 空闲
        MEASURING,      // 测量中
        STABLE,         // 稳定输出
        INVALID         // 无效状态
    };
    State current_state_;
    
    // 参数
    struct Params {
        size_t min_stable_frames = 10;      // 最小稳定帧数
        size_t measurement_window = 30;     // 测量窗口大小
        float outlier_threshold = 3.0;      // 异常值阈值（标准差倍数）
        float min_confidence = 0.7;         // 最小置信度
        float stability_threshold = 50.0;   // 稳定性阈值（mm）
    } params_;
    
public:
    HeightEstimator(const Params& params);
    
    // 处理新的测量
    void process_frame(const PoseResult& pose, const cv::Mat& homography);
    
    // 获取稳定的身高估计
    float get_stable_height() const;
    bool is_height_stable() const;
    float get_confidence() const;
    
private:
    // 异常值检测
    bool is_outlier(float height) const;
    
    // 计算稳定性指标
    float calculate_stability() const;
    
    // 状态转换
    void update_state();
};
```

#### 实现核心逻辑
```cpp
void HeightEstimator::process_frame(const PoseResult& pose, const cv::Mat& homography) {
    // 步骤1: 检查测量条件
    if (!can_measure_height(pose)) {
        // 如果不满足条件，可能需要重置状态
        if (current_state_ == MEASURING) {
            measurements_.clear();
            current_state_ = IDLE;
        }
        return;
    }
    
    // 步骤2: 计算原始身高
    float raw_height = calculate_raw_height(pose.bbox, homography);
    
    // 步骤3: 创建测量记录
    HeightMeasurement measurement;
    measurement.height_mm = raw_height;
    measurement.frame_id = frame_counter_++;
    measurement.confidence = calculate_measurement_confidence(pose);
    measurement.timestamp = std::chrono::steady_clock::now();
    
    // 步骤4: 异常值检测
    if (measurements_.size() >= 3 && is_outlier(raw_height)) {
        measurement.is_valid = false;
    } else {
        measurement.is_valid = true;
    }
    
    // 步骤5: 添加到缓冲区
    measurements_.push(measurement);
    
    // 步骤6: 滤波处理
    if (measurement.is_valid) {
        float filtered_height = height_filter_->process(raw_height);
        float filtered_confidence = confidence_filter_->process(measurement.confidence);
    }
    
    // 步骤7: 更新状态机
    update_state();
}

float HeightEstimator::get_stable_height() const {
    if (current_state_ != STABLE) {
        return -1.0f;
    }
    
    // 获取有效测量
    auto valid_measurements = filter_valid_measurements(measurements_.get_window(params_.measurement_window));
    
    // 方法1: 中值
    // return calculate_median(valid_measurements);
    
    // 方法2: 加权平均（基于置信度）
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    for (const auto& m : valid_measurements) {
        weighted_sum += m.height_mm * m.confidence;
        weight_sum += m.confidence;
    }
    
    return weight_sum > 0 ? weighted_sum / weight_sum : -1.0f;
}
```

### 3.2 要球动作识别（带时序判定）

#### 状态机设计
```cpp
class BallRequestDetector {
private:
    // 动作状态机
    enum RequestState {
        NO_REQUEST,         // 未要球
        POTENTIAL_REQUEST,  // 可能要球（检测到但未满足时间）
        CONFIRMED_REQUEST,  // 确认要球
        ENDING_REQUEST     // 结束要球（冷却期）
    };
    
    struct RequestContext {
        RequestState state = NO_REQUEST;
        int continuous_frames = 0;          // 连续检测帧数
        int total_frames = 0;              // 总检测帧数
        int interruption_frames = 0;        // 中断帧数
        float accumulated_confidence = 0.0f; // 累积置信度
        std::chrono::steady_clock::time_point start_time;
        std::chrono::steady_clock::time_point last_detected;
        
        // 手势特征的时序记录
        TemporalBuffer<float> hands_distance_buffer;
        TemporalBuffer<cv::Point2f> left_wrist_buffer;
        TemporalBuffer<cv::Point2f> right_wrist_buffer;
    };
    
    std::map<int, RequestContext> person_contexts_;  // 每个person的状态
    
    // 检测参数
    struct DetectionParams {
        int min_continuous_frames = 5;      // 最小连续帧数
        int max_interruption_frames = 2;    // 最大允许中断帧数
        float min_total_confidence = 3.5f;  // 最小累积置信度
        int cooldown_frames = 10;           // 冷却期帧数
        float gesture_stability_threshold = 0.2f; // 手势稳定性阈值
    } params_;
    
public:
    BallRequestDetector(const DetectionParams& params);
    
    // 处理一帧
    BallRequestResult process_frame(const PoseResult& pose);
    
    // 批量处理
    std::vector<BallRequestResult> process_batch(const std::vector<PoseResult>& poses);
    
private:
    // 检测原始要球动作
    bool detect_raw_request(const PoseResult& pose, float& confidence);
    
    // 计算手势稳定性
    float calculate_gesture_stability(const RequestContext& ctx);
    
    // 状态转换逻辑
    void update_state_machine(int person_id, bool is_requesting, float confidence);
    
    // 清理过期状态
    void cleanup_stale_contexts();
};
```

#### 核心实现
```cpp
BallRequestResult BallRequestDetector::process_frame(const PoseResult& pose) {
    BallRequestResult result;
    result.person_id = pose.person_id;
    
    // 获取或创建person的上下文
    auto& ctx = person_contexts_[pose.person_id];
    
    // 检测原始要球动作
    float confidence = 0.0f;
    bool is_requesting = detect_raw_request(pose, confidence);
    
    // 更新时序缓冲区
    if (is_requesting) {
        float hands_dist = calculate_hands_distance(pose);
        ctx.hands_distance_buffer.push(hands_dist);
        ctx.left_wrist_buffer.push(pose.keypoints[LEFT_WRIST]);
        ctx.right_wrist_buffer.push(pose.keypoints[RIGHT_WRIST]);
    }
    
    // 状态机更新
    switch (ctx.state) {
        case NO_REQUEST:
            if (is_requesting) {
                ctx.state = POTENTIAL_REQUEST;
                ctx.continuous_frames = 1;
                ctx.accumulated_confidence = confidence;
                ctx.start_time = std::chrono::steady_clock::now();
            }
            break;
            
        case POTENTIAL_REQUEST:
            if (is_requesting) {
                ctx.continuous_frames++;
                ctx.accumulated_confidence += confidence;
                
                // 检查是否满足确认条件
                if (ctx.continuous_frames >= params_.min_continuous_frames &&
                    ctx.accumulated_confidence >= params_.min_total_confidence) {
                    
                    // 额外检查：手势稳定性
                    float stability = calculate_gesture_stability(ctx);
                    if (stability < params_.gesture_stability_threshold) {
                        ctx.state = CONFIRMED_REQUEST;
                    }
                }
            } else {
                // 允许短暂中断
                ctx.interruption_frames++;
                if (ctx.interruption_frames > params_.max_interruption_frames) {
                    // 重置状态
                    ctx = RequestContext();
                }
            }
            break;
            
        case CONFIRMED_REQUEST:
            if (is_requesting) {
                ctx.continuous_frames++;
                ctx.accumulated_confidence += confidence;
                ctx.interruption_frames = 0;
            } else {
                ctx.interruption_frames++;
                if (ctx.interruption_frames > params_.max_interruption_frames) {
                    ctx.state = ENDING_REQUEST;
                    ctx.continuous_frames = 0;
                }
            }
            break;
            
        case ENDING_REQUEST:
            ctx.continuous_frames++;
            if (ctx.continuous_frames >= params_.cooldown_frames) {
                ctx = RequestContext();  // 重置
            }
            break;
    }
    
    // 设置返回结果
    result.is_requesting = (ctx.state == CONFIRMED_REQUEST);
    result.request_confidence = ctx.accumulated_confidence / std::max(1, ctx.continuous_frames);
    result.request_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - ctx.start_time).count();
    
    return result;
}

float BallRequestDetector::calculate_gesture_stability(const RequestContext& ctx) {
    // 计算手腕位置的方差
    auto left_wrist_window = ctx.left_wrist_buffer.get_window(5);
    auto right_wrist_window = ctx.right_wrist_buffer.get_window(5);
    
    if (left_wrist_window.size() < 3) {
        return 1.0f;  // 数据不足，返回不稳定
    }
    
    // 计算位置变化的标准差
    float variance_sum = 0.0f;
    cv::Point2f left_mean(0, 0), right_mean(0, 0);
    
    // 计算均值
    for (size_t i = 0; i < left_wrist_window.size(); i++) {
        left_mean += left_wrist_window[i];
        right_mean += right_wrist_window[i];
    }
    left_mean /= float(left_wrist_window.size());
    right_mean /= float(right_wrist_window.size());
    
    // 计算方差
    for (size_t i = 0; i < left_wrist_window.size(); i++) {
        float left_dist = cv::norm(left_wrist_window[i] - left_mean);
        float right_dist = cv::norm(right_wrist_window[i] - right_mean);
        variance_sum += left_dist * left_dist + right_dist * right_dist;
    }
    
    float std_dev = std::sqrt(variance_sum / (2 * left_wrist_window.size()));
    return std_dev / 100.0f;  // 归一化
}
```

### 3.3 增强的ID管理（带时序稳定性）

```cpp
class EnhancedIDManager {
private:
    struct IDSwapRequest {
        int requester_id;
        int target_id;
        float priority_score;
        int request_frames;
        std::chrono::steady_clock::time_point request_time;
    };
    
    struct PersonTrackingInfo {
        int current_id;
        int original_id;
        bool is_requesting;
        float request_priority;
        int stable_request_frames;
        std::chrono::steady_clock::time_point last_swap_time;
    };
    
    std::map<int, PersonTrackingInfo> tracking_info_;
    std::deque<IDSwapRequest> pending_swaps_;
    
    // 参数
    struct Params {
        int min_request_frames_for_swap = 15;   // 要球15帧才能申请换ID
        int swap_cooldown_ms = 2000;            // 2秒内不能再次换ID
        float priority_decay_rate = 0.95f;      // 优先级衰减率
        int max_pending_swaps = 5;              // 最多待处理交换请求
    } params_;
    
public:
    void update(const std::vector<BallRequestResult>& requests);
    std::map<int, int> get_id_mapping() const;
    
private:
    void process_swap_requests();
    bool can_swap(int person_id) const;
    float calculate_swap_priority(const BallRequestResult& request) const;
};
```

## 4. 配置文件格式（增强版）

### pose_analysis_config_v2.json
```json
{
    "height_detection": {
        "enabled": true,
        "measurement": {
            "min_keypoint_confidence": 0.5,
            "head_offset_pixels": 30,
            "height_correction_factor": 1.05,
            "min_roi_height_pixels": 100,
            "max_roi_height_pixels": 800
        },
        "filtering": {
            "type": "median",  // "median", "kalman", "moving_average"
            "window_size": 15,
            "min_stable_frames": 10,
            "outlier_threshold_sigma": 3.0,
            "stability_threshold_mm": 50.0,
            "kalman_process_noise": 0.01,
            "kalman_measurement_noise": 10.0
        },
        "temporal": {
            "measurement_window_frames": 30,
            "min_confidence": 0.7,
            "max_measurement_gap_ms": 500
        }
    },
    
    "ball_request_detection": {
        "enabled": true,
        "gesture": {
            "min_keypoint_confidence": 0.5,
            "max_hands_distance_mm": 400,
            "chest_region_scale": 1.5,
            "min_hand_height_ratio": 0.3,
            "max_hand_height_ratio": 0.7
        },
        "temporal": {
            "min_continuous_frames": 5,
            "max_interruption_frames": 2,
            "min_total_confidence": 3.5,
            "cooldown_frames": 10,
            "gesture_stability_threshold": 0.2,
            "detection_window_frames": 20
        },
        "filtering": {
            "smooth_confidence": true,
            "confidence_smoothing_alpha": 0.7,
            "position_smoothing_alpha": 0.8
        }
    },
    
    "id_management": {
        "enabled": true,
        "priority": {
            "confidence_weight": 0.3,
            "duration_weight": 0.4,
            "stability_weight": 0.3,
            "decay_rate": 0.95
        },
        "temporal": {
            "min_request_frames_for_swap": 15,
            "swap_cooldown_ms": 2000,
            "max_pending_swaps": 5,
            "priority_update_interval_ms": 100
        },
        "rules": {
            "allow_multiple_requesters": false,
            "preserve_original_order": true,
            "max_tracked_persons": 10
        }
    },
    
    "filtering_global": {
        "enable_temporal_smoothing": true,
        "enable_outlier_rejection": true,
        "enable_predictive_tracking": false,
        "frame_buffer_size": 60,
        "time_window_ms": 2000
    },
    
    "debug": {
        "show_filter_states": true,
        "show_temporal_buffers": true,
        "show_state_machines": true,
        "log_measurements": true,
        "save_debug_video": false,
        "debug_output_path": "./debug/"
    }
}
```

## 5. 滤波器选择指南

### 5.1 身高检测滤波器选择
| 滤波器类型 | 适用场景 | 优点 | 缺点 |
|----------|--------|------|------|
| **中值滤波** | 静止或缓慢移动 | 抗噪声能力强，能有效去除异常值 | 响应速度较慢 |
| **卡尔曼滤波** | 连续跟踪，有运动模型 | 响应快，预测能力强 | 需要调参，对模型依赖 |
| **移动平均** | 一般场景 | 简单高效，易于实现 | 对异常值敏感 |
| **自适应滤波** | 复杂动态场景 | 自动调整参数 | 计算复杂度高 |

### 5.2 动作检测滤波策略
- **预滤波**: 对关键点位置进行平滑
- **特征滤波**: 对手部距离等特征进行滤波
- **决策滤波**: 对最终判定结果进行时序滤波

## 6. 性能优化策略

### 6.1 计算优化
```cpp
// 使用SIMD加速
void batch_median_filter(float* data, size_t n, size_t window) {
    // AVX2/NEON优化的中值计算
}

// 查找表优化
class LookupTableFilter {
    std::vector<float> lut_;
    // 预计算常用值
};
```

### 6.2 内存优化
```cpp
// 环形缓冲区实现
template<typename T>
class CircularBuffer {
    std::vector<T> buffer_;
    size_t head_ = 0;
    size_t tail_ = 0;
    // 零拷贝访问
};
```

### 6.3 多线程处理
```cpp
class ParallelAnalyzer {
    // 身高检测线程
    std::thread height_thread_;
    // 动作检测线程
    std::thread gesture_thread_;
    // 结果融合
    std::mutex result_mutex_;
};
```

## 7. 测试验证方案

### 7.1 单元测试
```cpp
TEST(HeightFilter, MedianFilterStability) {
    MedianFilter filter(5);
    // 输入噪声数据
    std::vector<float> noisy_data = {1750, 1900, 1760, 1755, 2100, 1758};
    float result = 0;
    for (float d : noisy_data) {
        result = filter.process(d);
    }
    EXPECT_NEAR(result, 1758, 10);  // 期望去除异常值2100
}

TEST(BallRequestDetector, TemporalConsistency) {
    BallRequestDetector detector(params);
    // 模拟间断的要球动作
    // 验证允许的中断不影响检测
}
```

### 7.2 集成测试场景
1. **身高稳定性测试**: 人员静止30秒，验证身高输出稳定性
2. **要球动作鲁棒性**: 快速重复要球动作，验证不会误判
3. **多人场景**: 5人同时要球，验证ID管理正确性
4. **长时间运行**: 连续运行1小时，检查内存泄漏和性能退化

## 8. 可视化调试工具

### 8.1 实时显示面板
```cpp
class DebugVisualizer {
    void draw_height_chart(cv::Mat& frame, const HeightEstimator& estimator);
    void draw_gesture_state_machine(cv::Mat& frame, const RequestContext& ctx);
    void draw_temporal_buffer(cv::Mat& frame, const TemporalBuffer<float>& buffer);
    void draw_filter_comparison(cv::Mat& frame, 
                               const std::vector<IFilter*>& filters);
};
```

### 8.2 数据记录与回放
```cpp
class DataRecorder {
    void record_frame(const PoseResult& pose, 
                     const AnalysisResult& result);
    void save_session(const std::string& filename);
    void load_and_replay(const std::string& filename);
};
```

## 9. 使用示例

### 9.1 完整的测试程序
```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "PoseDetectorLib.h"
#include "PoseAnalyzer.h"

int main() {
    // 初始化检测器
    PoseDetectorLib detector("../models/Q_yolov8_pose.rknn");
    detector.enable_tracking(true);
    detector.load_calibration("../data/2025_8_6_1280_720.json");
    
    // 初始化分析器（带滤波配置）
    PoseAnalyzer analyzer("../data/pose_analysis_config_v2.json");
    analyzer.set_homography(detector.get_homography_matrix());
    
    // 配置滤波器
    analyzer.set_height_filter_type(FilterType::MEDIAN, 15);
    analyzer.set_gesture_filter_type(FilterType::KALMAN);
    
    // 打开摄像头
    cv::VideoCapture cap(0);
    
    // 调试可视化
    DebugVisualizer visualizer;
    DataRecorder recorder;
    
    int frame_count = 0;
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        // 姿态检测
        auto pose_results = detector.detect(frame);
        
        // 姿态分析（带多帧滤波）
        auto analysis_results = analyzer.analyze(pose_results);
        
        // 输出稳定的结果
        for (const auto& result : analysis_results) {
            // 只输出稳定的身高
            if (result.height_stable) {
                std::cout << "Person " << result.priority_id 
                         << " Height: " << result.estimated_height_mm 
                         << "mm (confidence: " << result.height_confidence << ")" << std::endl;
            }
            
            // 只输出确认的要球动作
            if (result.is_requesting_ball && result.request_confirmed) {
                std::cout << "Person " << result.priority_id 
                         << " REQUESTING BALL (duration: " 
                         << result.request_duration_ms << "ms)" << std::endl;
            }
        }
        
        // 绘制调试信息
        if (frame_count % 5 == 0) {  // 每5帧更新一次可视化
            visualizer.draw_height_chart(frame, analyzer.get_height_estimator());
            visualizer.draw_filter_comparison(frame, analyzer.get_active_filters());
        }
        
        // 记录数据
        recorder.record_frame(pose_results, analysis_results);
        
        cv::imshow("Pose Analysis with Filtering", frame);
        if (cv::waitKey(1) == 27) break;
        
        frame_count++;
    }
    
    // 保存记录的数据
    recorder.save_session("session_" + get_timestamp() + ".dat");
    
    return 0;
}
```

## 10. 部署建议

### 10.1 参数调优流程
1. **数据采集**: 录制不同场景的视频
2. **离线分析**: 使用DataRecorder回放，调整参数
3. **A/B测试**: 对比不同滤波器效果
4. **在线微调**: 根据实际使用反馈优化

### 10.2 场景适配建议
| 场景 | 身高检测配置 | 要球检测配置 |
|-----|------------|------------|
| 训练场 | 窗口15帧，中值滤波 | 连续3帧，容忍1帧中断 |
| 比赛场 | 窗口30帧，卡尔曼滤波 | 连续5帧，容忍2帧中断 |
| 青少年 | 窗口10帧，移动平均 | 连续3帧，手距阈值放宽 |

### 10.3 性能基准
- 目标延迟: <10ms per frame（含滤波）
- 身高稳定时间: 1-2秒
- 要球响应时间: 200-300ms
- ID切换延迟: 500ms

## 11. 常见问题与解决方案

### 11.1 身高跳变问题
**问题**: 身高测量值突然变化
**原因**: 关键点检测不稳定，透视变换误差
**解决**: 
- 增加中值滤波窗口
- 提高异常值检测阈值
- 使用卡尔曼滤波预测

### 11.2 要球误判问题
**问题**: 普通手势被识别为要球
**原因**: 阈值设置不当，时序判定不足
**解决**:
- 增加连续帧要求
- 加入手势稳定性检查
- 结合上下文信息

### 11.3 ID频繁切换
**问题**: ID在多人要球时频繁变化
**原因**: 优先级计算不稳定
**解决**:
- 增加切换冷却时间
- 使用优先级衰减机制
- 加入滞后阈值