#include <gtest/gtest.h>
#include "filter_interface.h"
#include <vector>
#include <cmath>

using namespace pose_analysis;

class FilterInterfaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 生成测试数据
        clean_data_ = {10.0f, 10.1f, 9.9f, 10.05f, 9.95f, 10.02f, 9.98f, 10.03f};
        noisy_data_ = {10.0f, 15.0f, 9.9f, 25.0f, 9.95f, 10.02f, 30.0f, 10.03f};
    }
    
    std::vector<float> clean_data_;
    std::vector<float> noisy_data_;
};

// 测试中值滤波器
TEST_F(FilterInterfaceTest, MedianFilter) {
    MedianFilter filter(5);  // 窗口大小为5
    
    EXPECT_EQ(filter.get_type(), "MedianFilter");
    EXPECT_FALSE(filter.is_stable());  // 初始时不稳定
    
    std::vector<float> results;
    for (float input : noisy_data_) {
        float output = filter.process(input);
        results.push_back(output);
    }
    
    // 中值滤波器应该在有足够数据后变稳定
    EXPECT_TRUE(filter.is_stable());
    
    // 最后的输出应该比较稳定（接近10）
    EXPECT_LT(std::abs(results.back() - 10.0f), 2.0f);
    
    // 测试重置功能
    filter.reset();
    EXPECT_FALSE(filter.is_stable());
}

// 测试移动平均滤波器
TEST_F(FilterInterfaceTest, MovingAverageFilter) {
    MovingAverageFilter filter(4, 1.0f);  // 简单移动平均，窗口大小为4
    
    EXPECT_EQ(filter.get_type(), "MovingAverageFilter");
    
    std::vector<float> results;
    for (float input : clean_data_) {
        float output = filter.process(input);
        results.push_back(output);
    }
    
    // 对于平滑数据，移动平均应该产生稳定的输出
    float last_result = results.back();
    EXPECT_LT(std::abs(last_result - 10.0f), 0.5f);
}

// 测试指数加权移动平均
TEST_F(FilterInterfaceTest, ExponentialMovingAverage) {
    MovingAverageFilter filter(10, 0.7f);  // α = 0.7的指数加权
    
    std::vector<float> step_input = {0.0f, 0.0f, 0.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    std::vector<float> results;
    
    for (float input : step_input) {
        float output = filter.process(input);
        results.push_back(output);
    }
    
    // 指数加权应该对阶跃输入有合理的响应
    EXPECT_GT(results[3], results[2]);  // 阶跃后应该增加
    EXPECT_LT(results.back(), 10.0f);   // 但不会立即达到10
}

// 测试卡尔曼滤波器
TEST_F(FilterInterfaceTest, KalmanFilter) {
    KalmanFilter1D filter(0.01f, 1.0f);  // 小的过程噪声，较大的测量噪声
    
    EXPECT_EQ(filter.get_type(), "KalmanFilter1D");
    EXPECT_FALSE(filter.is_stable());
    
    // 用恒定值初始化
    for (int i = 0; i < 10; ++i) {
        filter.process(10.0f);
    }
    
    EXPECT_TRUE(filter.is_stable());
    
    // 添加一个异常值
    float result_before = filter.process(10.0f);
    float result_outlier = filter.process(50.0f);  // 异常值
    float result_after = filter.process(10.0f);
    
    // 卡尔曼滤波器应该对异常值有一定的抑制作用
    EXPECT_LT(std::abs(result_outlier - result_before), 10.0f);  // 不应该跳变太大
    
    // 重置测试
    filter.reset();
    EXPECT_FALSE(filter.is_stable());
}

// 测试自适应滤波器
TEST_F(FilterInterfaceTest, AdaptiveFilter) {
    auto primary = std::make_unique<MedianFilter>(5);
    auto fallback = std::make_unique<MovingAverageFilter>(5);
    
    AdaptiveFilter filter(std::move(primary), std::move(fallback), 0.1f);
    
    // 测试稳定数据（应该使用主滤波器）
    for (float input : clean_data_) {
        filter.process(input);
    }
    
    std::string status = filter.get_status();
    EXPECT_NE(status.find("primary"), std::string::npos);  // 应该显示使用primary
    
    // 测试不稳定数据
    for (float input : noisy_data_) {
        filter.process(input);
    }
    
    // 可能会切换到fallback滤波器
    EXPECT_TRUE(filter.is_stable());
}

// 测试组合滤波器
TEST_F(FilterInterfaceTest, CompositeFilter) {
    CompositeFilter composite("TestComposite");
    
    // 添加多个滤波器阶段
    composite.add_filter(std::make_unique<MovingAverageFilter>(3));
    composite.add_filter(std::make_unique<MedianFilter>(3));
    
    EXPECT_EQ(composite.get_type(), "TestComposite");
    
    float result = 0.0f;
    
    // 处理数据
    for (float input : noisy_data_) {
        result = composite.process(input);
    }
    
    // 组合滤波器应该产生更平滑的结果
    EXPECT_TRUE(composite.is_stable());
    
    // 检查状态信息
    std::string status = composite.get_status();
    EXPECT_NE(status.find("stages"), std::string::npos);
}

// 测试滤波器工厂
TEST_F(FilterInterfaceTest, FilterFactory) {
    // 测试创建不同类型的滤波器
    auto median = FilterFactory::create_median_filter(10);
    EXPECT_EQ(median->get_type(), "MedianFilter");
    
    auto moving_avg = FilterFactory::create_moving_average_filter(8, 0.8f);
    EXPECT_EQ(moving_avg->get_type(), "MovingAverageFilter");
    
    auto kalman = FilterFactory::create_kalman_filter(0.1f, 5.0f);
    EXPECT_EQ(kalman->get_type(), "KalmanFilter1D");
    
    auto adaptive = FilterFactory::create_adaptive_filter("median", "moving_average");
    EXPECT_NE(adaptive->get_type().find("AdaptiveFilter"), std::string::npos);
    
    // 测试通用创建函数
    auto generic_median = FilterFactory::create_filter("median", 15);
    EXPECT_EQ(generic_median->get_type(), "MedianFilter");
    
    auto generic_kalman = FilterFactory::create_filter("kalman", 10, 0.05f, 2.0f);
    EXPECT_EQ(generic_kalman->get_type(), "KalmanFilter1D");
    
    // 测试无效类型（应该返回默认的中值滤波器）
    auto default_filter = FilterFactory::create_filter("invalid_type");
    EXPECT_EQ(default_filter->get_type(), "MedianFilter");
}

// 测试滤波器性能和收敛性
TEST_F(FilterInterfaceTest, FilterConvergence) {
    MedianFilter median_filter(10);
    KalmanFilter1D kalman_filter(0.01f, 1.0f);
    
    // 使用恒定输入测试收敛
    const float target_value = 15.0f;
    const int num_iterations = 50;
    
    std::vector<float> median_results, kalman_results;
    
    for (int i = 0; i < num_iterations; ++i) {
        median_results.push_back(median_filter.process(target_value));
        kalman_results.push_back(kalman_filter.process(target_value));
    }
    
    // 两个滤波器最终都应该收敛到目标值附近
    EXPECT_LT(std::abs(median_results.back() - target_value), 0.1f);
    EXPECT_LT(std::abs(kalman_results.back() - target_value), 0.5f);
    
    // 卡尔曼滤波器应该收敛更快
    float median_early = median_results[10];  // 第10次迭代
    float kalman_early = kalman_results[10];
    
    EXPECT_LT(std::abs(kalman_early - target_value), std::abs(median_early - target_value));
}

// 测试滤波器在噪声环境下的表现
TEST_F(FilterInterfaceTest, NoiseResistance) {
    MedianFilter median_filter(7);
    MovingAverageFilter ma_filter(7);
    
    // 创建带有尖峰噪声的信号
    std::vector<float> signal_with_spikes;
    for (int i = 0; i < 20; ++i) {
        float base_signal = 10.0f + 2.0f * std::sin(i * 0.3f);  // 基础信号
        
        // 随机添加尖峰噪声
        if (i % 7 == 0) {
            signal_with_spikes.push_back(base_signal + 20.0f);  // 正尖峰
        } else if (i % 11 == 0) {
            signal_with_spikes.push_back(base_signal - 15.0f);  // 负尖峰
        } else {
            signal_with_spikes.push_back(base_signal);
        }
    }
    
    std::vector<float> median_output, ma_output;
    
    for (float input : signal_with_spikes) {
        median_output.push_back(median_filter.process(input));
        ma_output.push_back(ma_filter.process(input));
    }
    
    // 计算输出的方差
    float median_variance = BufferStatistics::standard_deviation(median_output);
    float ma_variance = BufferStatistics::standard_deviation(ma_output);
    
    // 中值滤波器在尖峰噪声环境下应该表现更好（方差更小）
    EXPECT_LT(median_variance, ma_variance);
}

// 测试滤波器状态信息
TEST_F(FilterInterfaceTest, FilterStatus) {
    KalmanFilter1D filter(0.01f, 1.0f);
    
    // 初始状态信息
    std::string initial_status = filter.get_status();
    EXPECT_NE(initial_status.find("updates"), std::string::npos);
    EXPECT_NE(initial_status.find("P:"), std::string::npos);  // 协方差信息
    
    // 处理一些数据
    for (int i = 0; i < 10; ++i) {
        filter.process(10.0f);
    }
    
    std::string updated_status = filter.get_status();
    EXPECT_NE(initial_status, updated_status);  // 状态应该有变化
    
    // 测试协方差获取
    float covariance = filter.get_covariance();
    EXPECT_GT(covariance, 0.0f);  // 协方差应该大于0
    EXPECT_LT(covariance, 1000.0f);  // 但不应该过大
}

// 边界条件测试
TEST_F(FilterInterfaceTest, EdgeCases) {
    MedianFilter filter(5);
    
    // 测试极值输入
    EXPECT_NO_THROW(filter.process(std::numeric_limits<float>::max()));
    EXPECT_NO_THROW(filter.process(std::numeric_limits<float>::lowest()));
    EXPECT_NO_THROW(filter.process(0.0f));
    
    // 测试NaN和无穷大（如果支持的话）
    // 注意：实际实现可能需要特殊处理这些情况
    
    // 测试快速重置和重新初始化
    for (int i = 0; i < 3; ++i) {
        filter.process(10.0f);
    }
    filter.reset();
    EXPECT_FALSE(filter.is_stable());
    
    filter.process(5.0f);
    // 重置后应该可以正常工作
}

// 测试滤波器参数设置
TEST_F(FilterInterfaceTest, FilterParameterSetting) {
    KalmanFilter1D filter(0.1f, 10.0f);
    
    // 获取初始协方差
    filter.process(10.0f);  // 处理一个值以初始化
    float initial_covariance = filter.get_covariance();
    
    // 改变噪声参数
    filter.set_noise(0.001f, 1.0f);  // 更小的过程噪声，更小的测量噪声
    
    // 重新初始化
    filter.reset();
    for (int i = 0; i < 10; ++i) {
        filter.process(10.0f);
    }
    
    float new_covariance = filter.get_covariance();
    
    // 较小的噪声设置应该导致更小的最终协方差
    EXPECT_LT(new_covariance, initial_covariance);
}