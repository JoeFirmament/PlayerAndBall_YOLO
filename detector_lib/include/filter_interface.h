#pragma once

#include "temporal_buffer.h"
#include <memory>
#include <string>
#include <cmath>

namespace pose_analysis {

// 滤波器接口
class IFilter {
public:
    virtual ~IFilter() = default;
    
    // 处理输入数据并返回滤波后的结果
    virtual float process(float input) = 0;
    
    // 重置滤波器状态
    virtual void reset() = 0;
    
    // 检查是否已稳定
    virtual bool is_stable() const = 0;
    
    // 获取滤波器类型名称
    virtual std::string get_type() const = 0;
    
    // 获取当前状态信息
    virtual std::string get_status() const { return ""; }
};

// 中值滤波器
class MedianFilter : public IFilter {
private:
    TemporalBuffer<float> buffer_;
    size_t window_size_;
    size_t min_stable_size_;
    
public:
    MedianFilter(size_t window_size, size_t min_stable_size = 5)
        : buffer_(window_size), window_size_(window_size), min_stable_size_(min_stable_size) {}
    
    float process(float input) override {
        buffer_.push(input);
        auto values = buffer_.get_all();
        return BufferStatistics::median(values);
    }
    
    void reset() override {
        buffer_.clear();
    }
    
    bool is_stable() const override {
        return buffer_.size() >= min_stable_size_;
    }
    
    std::string get_type() const override {
        return "MedianFilter";
    }
    
    std::string get_status() const override {
        return "size: " + std::to_string(buffer_.size()) + 
               "/" + std::to_string(window_size_);
    }
};

// 移动平均滤波器
class MovingAverageFilter : public IFilter {
private:
    TemporalBuffer<float> buffer_;
    size_t window_size_;
    size_t min_stable_size_;
    float alpha_;  // 指数加权系数 (1.0 = 简单平均)
    
public:
    MovingAverageFilter(size_t window_size, float alpha = 1.0f, size_t min_stable_size = 3)
        : buffer_(window_size), window_size_(window_size), 
          min_stable_size_(min_stable_size), alpha_(alpha) {}
    
    float process(float input) override {
        buffer_.push(input);
        auto values = buffer_.get_all();
        
        if (values.empty()) return 0.0f;
        
        if (alpha_ >= 1.0f) {
            // 简单移动平均
            return BufferStatistics::mean(values);
        } else {
            // 指数加权移动平均
            float result = values[0];
            for (size_t i = 1; i < values.size(); ++i) {
                result = alpha_ * values[i] + (1 - alpha_) * result;
            }
            return result;
        }
    }
    
    void reset() override {
        buffer_.clear();
    }
    
    bool is_stable() const override {
        return buffer_.size() >= min_stable_size_;
    }
    
    std::string get_type() const override {
        return "MovingAverageFilter";
    }
    
    std::string get_status() const override {
        return "size: " + std::to_string(buffer_.size()) + 
               "/" + std::to_string(window_size_) + 
               ", alpha: " + std::to_string(alpha_);
    }
};

// 1D卡尔曼滤波器
class KalmanFilter1D : public IFilter {
private:
    float x_;           // 状态估计
    float P_;           // 估计误差协方差
    float Q_;           // 过程噪声协方差
    float R_;           // 测量噪声协方差
    bool initialized_;
    int update_count_;
    int min_updates_for_stable_;
    
public:
    KalmanFilter1D(float process_noise = 0.01f, float measurement_noise = 10.0f,
                   int min_updates = 5)
        : x_(0.0f), P_(1000.0f), Q_(process_noise), R_(measurement_noise),
          initialized_(false), update_count_(0), min_updates_for_stable_(min_updates) {}
    
    float process(float measurement) override {
        if (!initialized_) {
            // 使用第一次测量初始化
            x_ = measurement;
            P_ = R_;
            initialized_ = true;
            update_count_ = 1;
            return x_;
        }
        
        // 预测步骤
        // x_ = x_ (假设状态转移矩阵为1，无控制输入)
        P_ = P_ + Q_;
        
        // 更新步骤
        float K = P_ / (P_ + R_);  // 卡尔曼增益
        x_ = x_ + K * (measurement - x_);
        P_ = (1 - K) * P_;
        
        update_count_++;
        return x_;
    }
    
    void reset() override {
        x_ = 0.0f;
        P_ = 1000.0f;
        initialized_ = false;
        update_count_ = 0;
    }
    
    bool is_stable() const override {
        return initialized_ && update_count_ >= min_updates_for_stable_ && P_ < 100.0f;
    }
    
    std::string get_type() const override {
        return "KalmanFilter1D";
    }
    
    std::string get_status() const override {
        return "updates: " + std::to_string(update_count_) + 
               ", P: " + std::to_string(P_) +
               ", Q: " + std::to_string(Q_) +
               ", R: " + std::to_string(R_);
    }
    
    // 设置噪声参数
    void set_noise(float process_noise, float measurement_noise) {
        Q_ = process_noise;
        R_ = measurement_noise;
    }
    
    // 获取估计协方差
    float get_covariance() const { return P_; }
};

// 自适应滤波器（根据数据稳定性自动调整参数）
class AdaptiveFilter : public IFilter {
private:
    std::unique_ptr<IFilter> primary_filter_;
    std::unique_ptr<IFilter> fallback_filter_;
    TemporalBuffer<float> stability_buffer_;
    float stability_threshold_;
    bool use_primary_;
    int switch_cooldown_frames_;
    int cooldown_counter_;
    
public:
    AdaptiveFilter(std::unique_ptr<IFilter> primary, 
                  std::unique_ptr<IFilter> fallback,
                  float stability_threshold = 0.1f,
                  int switch_cooldown = 10)
        : primary_filter_(std::move(primary)), 
          fallback_filter_(std::move(fallback)),
          stability_buffer_(20),
          stability_threshold_(stability_threshold),
          use_primary_(true),
          switch_cooldown_frames_(switch_cooldown),
          cooldown_counter_(0) {}
    
    float process(float input) override {
        // 计算数据稳定性
        stability_buffer_.push(input);
        auto recent_values = stability_buffer_.get_window(10);
        float cv = BufferStatistics::coefficient_of_variation(recent_values);
        
        // 决定使用哪个滤波器
        if (cooldown_counter_ <= 0) {
            bool should_use_primary = cv <= stability_threshold_;
            if (should_use_primary != use_primary_) {
                use_primary_ = should_use_primary;
                cooldown_counter_ = switch_cooldown_frames_;
            }
        } else {
            cooldown_counter_--;
        }
        
        // 使用选定的滤波器
        return use_primary_ ? primary_filter_->process(input) : 
                             fallback_filter_->process(input);
    }
    
    void reset() override {
        primary_filter_->reset();
        fallback_filter_->reset();
        stability_buffer_.clear();
        use_primary_ = true;
        cooldown_counter_ = 0;
    }
    
    bool is_stable() const override {
        return use_primary_ ? primary_filter_->is_stable() : 
                             fallback_filter_->is_stable();
    }
    
    std::string get_type() const override {
        return "AdaptiveFilter(" + 
               (use_primary_ ? primary_filter_->get_type() : fallback_filter_->get_type()) + ")";
    }
    
    std::string get_status() const override {
        auto recent_values = stability_buffer_.get_window(10);
        float cv = BufferStatistics::coefficient_of_variation(recent_values);
        return "CV: " + std::to_string(cv) + 
               ", using: " + (use_primary_ ? "primary" : "fallback") +
               ", cooldown: " + std::to_string(cooldown_counter_);
    }
};

// 组合滤波器（串联多个滤波器）
class CompositeFilter : public IFilter {
private:
    std::vector<std::unique_ptr<IFilter>> filters_;
    std::string type_name_;
    
public:
    CompositeFilter(const std::string& name = "CompositeFilter") : type_name_(name) {}
    
    void add_filter(std::unique_ptr<IFilter> filter) {
        filters_.push_back(std::move(filter));
    }
    
    float process(float input) override {
        float result = input;
        for (auto& filter : filters_) {
            result = filter->process(result);
        }
        return result;
    }
    
    void reset() override {
        for (auto& filter : filters_) {
            filter->reset();
        }
    }
    
    bool is_stable() const override {
        // 所有滤波器都稳定才算稳定
        for (const auto& filter : filters_) {
            if (!filter->is_stable()) {
                return false;
            }
        }
        return !filters_.empty();
    }
    
    std::string get_type() const override {
        return type_name_;
    }
    
    std::string get_status() const override {
        std::string status = "stages: " + std::to_string(filters_.size());
        for (size_t i = 0; i < filters_.size(); ++i) {
            status += "\n  " + std::to_string(i) + ": " + filters_[i]->get_status();
        }
        return status;
    }
};

// 滤波器工厂
class FilterFactory {
public:
    static std::unique_ptr<IFilter> create_median_filter(size_t window_size) {
        return std::make_unique<MedianFilter>(window_size);
    }
    
    static std::unique_ptr<IFilter> create_moving_average_filter(size_t window_size, float alpha = 1.0f) {
        return std::make_unique<MovingAverageFilter>(window_size, alpha);
    }
    
    static std::unique_ptr<IFilter> create_kalman_filter(float process_noise, float measurement_noise) {
        return std::make_unique<KalmanFilter1D>(process_noise, measurement_noise);
    }
    
    static std::unique_ptr<IFilter> create_adaptive_filter(const std::string& primary_type,
                                                          const std::string& fallback_type,
                                                          float stability_threshold = 0.1f) {
        auto primary = create_filter(primary_type);
        auto fallback = create_filter(fallback_type);
        return std::make_unique<AdaptiveFilter>(std::move(primary), std::move(fallback), stability_threshold);
    }
    
    static std::unique_ptr<IFilter> create_filter(const std::string& type,
                                                 size_t window_size = 15,
                                                 float param1 = 1.0f,
                                                 float param2 = 10.0f) {
        if (type == "median") {
            return create_median_filter(window_size);
        } else if (type == "moving_average") {
            return create_moving_average_filter(window_size, param1);
        } else if (type == "kalman") {
            return create_kalman_filter(param1, param2);
        } else if (type == "adaptive_median_ma") {
            return create_adaptive_filter("median", "moving_average");
        } else if (type == "adaptive_kalman_median") {
            return create_adaptive_filter("kalman", "median");
        } else {
            // 默认返回中值滤波器
            return create_median_filter(window_size);
        }
    }
};

} // namespace pose_analysis