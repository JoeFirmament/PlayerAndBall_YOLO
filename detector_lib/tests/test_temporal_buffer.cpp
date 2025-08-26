#include <gtest/gtest.h>
#include "temporal_buffer.h"
#include <thread>
#include <chrono>

using namespace pose_analysis;

class TemporalBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试前准备
    }
    
    void TearDown() override {
        // 测试后清理
    }
};

// 测试基本的push和get功能
TEST_F(TemporalBufferTest, BasicPushAndGet) {
    TemporalBuffer<int> buffer(5);
    
    // 测试空缓冲区
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0);
    
    // 添加元素
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    EXPECT_FALSE(buffer.empty());
    EXPECT_EQ(buffer.size(), 3);
    EXPECT_EQ(buffer.get_latest(), 3);
    
    // 获取窗口
    auto window = buffer.get_window(2);
    ASSERT_EQ(window.size(), 2);
    EXPECT_EQ(window[0], 2);
    EXPECT_EQ(window[1], 3);
}

// 测试缓冲区大小限制
TEST_F(TemporalBufferTest, BufferSizeLimit) {
    TemporalBuffer<int> buffer(3);
    
    // 填满缓冲区
    for (int i = 1; i <= 5; ++i) {
        buffer.push(i);
    }
    
    // 应该只保留最后3个元素
    EXPECT_EQ(buffer.size(), 3);
    auto all = buffer.get_all();
    ASSERT_EQ(all.size(), 3);
    EXPECT_EQ(all[0], 3);
    EXPECT_EQ(all[1], 4);
    EXPECT_EQ(all[2], 5);
}

// 测试稳定性检查
TEST_F(TemporalBufferTest, StabilityCheck) {
    TemporalBuffer<float> buffer(10);
    
    EXPECT_FALSE(buffer.is_stable(5));  // 不够元素
    
    for (int i = 0; i < 6; ++i) {
        buffer.push(1.0f);
    }
    
    EXPECT_TRUE(buffer.is_stable(5));   // 足够元素
    EXPECT_FALSE(buffer.is_stable(10)); // 不够10个元素
}

// 测试线程安全
TEST_F(TemporalBufferTest, ThreadSafety) {
    TemporalBuffer<int> buffer(100);
    const int num_threads = 4;
    const int elements_per_thread = 25;
    
    std::vector<std::thread> threads;
    
    // 启动多个线程同时写入
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&buffer, t, elements_per_thread]() {
            for (int i = 0; i < elements_per_thread; ++i) {
                buffer.push(t * elements_per_thread + i);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 验证总元素数量
    EXPECT_EQ(buffer.size(), num_threads * elements_per_thread);
}

// 测试时间戳缓冲区
TEST_F(TemporalBufferTest, TimestampedBuffer) {
    TimestampedBuffer<int> ts_buffer(10, 100);  // 100ms时间窗口
    
    ts_buffer.push(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    ts_buffer.push(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    ts_buffer.push(3);
    
    // 应该只有最后两个元素在时间窗口内
    auto recent = ts_buffer.get_recent_data();
    EXPECT_LE(recent.size(), 2);  // 可能由于时序差异有1或2个元素
    
    // 清理过期数据
    ts_buffer.cleanup_expired();
    EXPECT_LE(ts_buffer.size(), 2);
}

// 测试统计函数
TEST_F(TemporalBufferTest, BufferStatistics) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // 测试中值
    float median = BufferStatistics::median(values);
    EXPECT_FLOAT_EQ(median, 3.0f);
    
    // 测试平均值
    float mean = BufferStatistics::mean(values);
    EXPECT_FLOAT_EQ(mean, 3.0f);
    
    // 测试标准差
    float std_dev = BufferStatistics::standard_deviation(values);
    EXPECT_GT(std_dev, 0.0f);
    
    // 测试异常值检测
    std::vector<float> with_outlier = {1.0f, 2.0f, 3.0f, 4.0f, 100.0f};
    auto outliers = BufferStatistics::detect_outliers_3sigma(with_outlier);
    EXPECT_TRUE(outliers[4]);  // 最后一个元素应该是异常值
}

// 测试过滤异常值
TEST_F(TemporalBufferTest, OutlierFiltering) {
    std::vector<float> noisy_data = {10.0f, 11.0f, 9.0f, 10.5f, 50.0f, 10.2f, 9.8f};
    
    auto filtered = BufferStatistics::filter_outliers(noisy_data, 2.0f);
    
    // 异常值50.0应该被过滤掉
    EXPECT_LT(filtered.size(), noisy_data.size());
    
    // 验证过滤后的数据不包含明显异常值
    for (float val : filtered) {
        EXPECT_LT(val, 20.0f);  // 应该都小于20
    }
}

// 测试变异系数计算
TEST_F(TemporalBufferTest, CoefficientOfVariation) {
    // 稳定的数据
    std::vector<float> stable_data = {10.0f, 10.1f, 9.9f, 10.05f, 9.95f};
    float stable_cv = BufferStatistics::coefficient_of_variation(stable_data);
    
    // 不稳定的数据
    std::vector<float> unstable_data = {5.0f, 15.0f, 8.0f, 20.0f, 2.0f};
    float unstable_cv = BufferStatistics::coefficient_of_variation(unstable_data);
    
    // 不稳定数据的变异系数应该更大
    EXPECT_GT(unstable_cv, stable_cv);
}

// 测试边界条件
TEST_F(TemporalBufferTest, EdgeCases) {
    TemporalBuffer<int> buffer(5);
    
    // 测试空缓冲区的操作
    EXPECT_EQ(buffer.get_latest(), int{});  // 应该返回默认构造的值
    auto empty_window = buffer.get_window(3);
    EXPECT_TRUE(empty_window.empty());
    
    // 测试请求超过缓冲区大小的窗口
    buffer.push(1);
    buffer.push(2);
    auto large_window = buffer.get_window(10);
    EXPECT_EQ(large_window.size(), 2);  // 应该返回所有可用元素
    
    // 测试单个元素的统计
    std::vector<float> single_element = {5.0f};
    EXPECT_FLOAT_EQ(BufferStatistics::median(single_element), 5.0f);
    EXPECT_FLOAT_EQ(BufferStatistics::mean(single_element), 5.0f);
    EXPECT_FLOAT_EQ(BufferStatistics::standard_deviation(single_element), 0.0f);
}

// 性能测试
TEST_F(TemporalBufferTest, Performance) {
    const int buffer_size = 1000;
    const int num_operations = 10000;
    
    TemporalBuffer<int> buffer(buffer_size);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 大量push操作
    for (int i = 0; i < num_operations; ++i) {
        buffer.push(i);
        
        // 间歇性地读取数据
        if (i % 100 == 0) {
            auto window = buffer.get_window(50);
            (void)window;  // 避免编译器优化掉
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 性能检查：10000次操作应该在合理时间内完成
    EXPECT_LT(duration.count(), 1000);  // 应该小于1秒
    
    // 验证最终状态
    EXPECT_EQ(buffer.size(), buffer_size);
    EXPECT_EQ(buffer.get_latest(), num_operations - 1);
}