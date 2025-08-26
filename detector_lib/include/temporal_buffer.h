#pragma once

#include <deque>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <mutex>

namespace pose_analysis {

// 通用时序缓冲区模板
template<typename T>
class TemporalBuffer {
private:
    std::deque<T> buffer_;
    size_t max_size_;
    mutable std::mutex mutex_;
    
public:
    explicit TemporalBuffer(size_t max_size) : max_size_(max_size) {}
    
    // 添加新数据
    void push(const T& data) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.push_back(data);
        if (buffer_.size() > max_size_) {
            buffer_.pop_front();
        }
    }
    
    // 获取最近N个元素
    std::vector<T> get_window(size_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_.empty()) return {};
        
        size_t start = buffer_.size() > n ? buffer_.size() - n : 0;
        return std::vector<T>(buffer_.begin() + start, buffer_.end());
    }
    
    // 获取所有元素
    std::vector<T> get_all() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::vector<T>(buffer_.begin(), buffer_.end());
    }
    
    // 获取最新元素
    T get_latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty() ? T{} : buffer_.back();
    }
    
    // 检查是否有足够的稳定数据
    bool is_stable(size_t min_frames) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size() >= min_frames;
    }
    
    // 获取当前大小
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }
    
    // 清空缓冲区
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.clear();
    }
    
    // 检查是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.empty();
    }
    
    // 获取指定位置的元素
    T at(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (index >= buffer_.size()) return T{};
        return buffer_[index];
    }
};

// 带时间戳的时序缓冲区
template<typename T>
class TimestampedBuffer {
private:
    struct TimestampedData {
        T data;
        std::chrono::steady_clock::time_point timestamp;
        
        TimestampedData() : timestamp(std::chrono::steady_clock::now()) {}
        TimestampedData(const T& d) : data(d), timestamp(std::chrono::steady_clock::now()) {}
    };
    
    TemporalBuffer<TimestampedData> buffer_;
    int time_window_ms_;
    
public:
    TimestampedBuffer(size_t max_size, int time_window_ms = 2000) 
        : buffer_(max_size), time_window_ms_(time_window_ms) {}
    
    // 添加数据
    void push(const T& data) {
        buffer_.push(TimestampedData(data));
    }
    
    // 获取时间窗口内的数据
    std::vector<T> get_recent_data() const {
        auto all_data = buffer_.get_all();
        auto now = std::chrono::steady_clock::now();
        std::vector<T> result;
        
        for (const auto& item : all_data) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - item.timestamp).count();
            if (elapsed <= time_window_ms_) {
                result.push_back(item.data);
            }
        }
        
        return result;
    }
    
    // 获取指定时间窗口内的数据
    std::vector<T> get_data_in_window(int window_ms) const {
        auto all_data = buffer_.get_all();
        auto now = std::chrono::steady_clock::now();
        std::vector<T> result;
        
        for (const auto& item : all_data) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - item.timestamp).count();
            if (elapsed <= window_ms) {
                result.push_back(item.data);
            }
        }
        
        return result;
    }
    
    // 清理过期数据
    void cleanup_expired() {
        auto all_data = buffer_.get_all();
        auto now = std::chrono::steady_clock::now();
        buffer_.clear();
        
        for (const auto& item : all_data) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - item.timestamp).count();
            if (elapsed <= time_window_ms_) {
                buffer_.push(item);
            }
        }
    }
    
    size_t size() const { return buffer_.size(); }
    bool empty() const { return buffer_.empty(); }
    void clear() { buffer_.clear(); }
};

// 统计工具类
class BufferStatistics {
public:
    // 计算中值
    template<typename T>
    static T median(std::vector<T> values) {
        if (values.empty()) return T{};
        
        std::sort(values.begin(), values.end());
        size_t n = values.size();
        if (n % 2 == 0) {
            return static_cast<T>((values[n/2 - 1] + values[n/2]) / 2.0);
        } else {
            return values[n/2];
        }
    }
    
    // 计算平均值
    template<typename T>
    static T mean(const std::vector<T>& values) {
        if (values.empty()) return T{};
        
        T sum = std::accumulate(values.begin(), values.end(), T{});
        return static_cast<T>(sum / values.size());
    }
    
    // 计算标准差
    template<typename T>
    static T standard_deviation(const std::vector<T>& values) {
        if (values.size() <= 1) return T{};
        
        T avg = mean(values);
        T sum_sq_diff = T{};
        for (const auto& val : values) {
            T diff = val - avg;
            sum_sq_diff += diff * diff;
        }
        
        return static_cast<T>(std::sqrt(sum_sq_diff / (values.size() - 1)));
    }
    
    // 计算中值绝对偏差 (MAD)
    template<typename T>
    static T median_absolute_deviation(std::vector<T> values) {
        if (values.empty()) return T{};
        
        T med = median(values);
        std::vector<T> deviations;
        for (const auto& val : values) {
            deviations.push_back(std::abs(val - med));
        }
        
        return median(deviations);
    }
    
    // 异常值检测 (使用3σ原则)
    template<typename T>
    static std::vector<bool> detect_outliers_3sigma(const std::vector<T>& values, T sigma_multiplier = 3.0) {
        std::vector<bool> outliers(values.size(), false);
        if (values.size() <= 2) return outliers;
        
        T avg = mean(values);
        T std_dev = standard_deviation(values);
        T threshold = sigma_multiplier * std_dev;
        
        for (size_t i = 0; i < values.size(); ++i) {
            outliers[i] = std::abs(values[i] - avg) > threshold;
        }
        
        return outliers;
    }
    
    // 异常值检测 (使用MAD方法)
    template<typename T>
    static std::vector<bool> detect_outliers_mad(std::vector<T> values, T mad_multiplier = 3.0) {
        std::vector<bool> outliers(values.size(), false);
        if (values.size() <= 2) return outliers;
        
        T med = median(values);
        T mad = median_absolute_deviation(values);
        T threshold = mad_multiplier * mad;
        
        for (size_t i = 0; i < values.size(); ++i) {
            outliers[i] = std::abs(values[i] - med) > threshold;
        }
        
        return outliers;
    }
    
    // 过滤异常值
    template<typename T>
    static std::vector<T> filter_outliers(const std::vector<T>& values, T sigma_multiplier = 3.0) {
        auto outliers = detect_outliers_3sigma(values, sigma_multiplier);
        std::vector<T> filtered;
        
        for (size_t i = 0; i < values.size(); ++i) {
            if (!outliers[i]) {
                filtered.push_back(values[i]);
            }
        }
        
        return filtered;
    }
    
    // 计算稳定性指标 (变异系数)
    template<typename T>
    static T coefficient_of_variation(const std::vector<T>& values) {
        if (values.empty()) return T{};
        
        T avg = mean(values);
        if (avg == T{}) return T{};
        
        T std_dev = standard_deviation(values);
        return std_dev / std::abs(avg);
    }
};

} // namespace pose_analysis