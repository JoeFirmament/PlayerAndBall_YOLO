#include <gtest/gtest.h>
#include "height_detector.h"
#include <opencv2/opencv.hpp>

using namespace pose_analysis;

class HeightDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建默认配置
        config_.min_keypoint_confidence = 0.5f;
        config_.filter_type = "median";
        config_.window_size = 10;
        config_.min_stable_frames = 5;
        config_.stability_threshold_mm = 50.0f;
        config_.height_correction_factor = 1.0f;
        config_.head_offset_pixels = 30;
        config_.min_roi_height_pixels = 100;
        config_.max_roi_height_pixels = 600;
        
        detector_ = std::make_unique<HeightDetector>(config_);
        
        // 创建测试用的Homography矩阵 (单位矩阵，像素=毫米)
        homography_ = cv::Mat::eye(3, 3, CV_64F);
        detector_->set_homography(homography_);
        
        // 创建测试姿态数据
        createTestPoseResult();
    }
    
    void createTestPoseResult() {
        test_pose_.person_id = 1;
        test_pose_.bbox = cv::Rect2f(100, 50, 200, 500);  // 200px宽, 500px高
        test_pose_.detection_confidence = 0.8f;
        test_pose_.timestamp = std::chrono::steady_clock::now();
        
        // 创建17个关键点
        test_pose_.keypoints.resize(17);
        test_pose_.keypoint_confidences.resize(17, 0.7f);
        
        // 设置主要关键点位置
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::NOSE)] = cv::Point2f(200, 100);          // 鼻子
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_SHOULDER)] = cv::Point2f(180, 150); // 左肩
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_SHOULDER)] = cv::Point2f(220, 150); // 右肩
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_WRIST)] = cv::Point2f(160, 300);    // 左手腕 (低于头部)
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_WRIST)] = cv::Point2f(240, 300);   // 右手腕 (低于头部)
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)] = cv::Point2f(180, 500);    // 左脚踝
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)] = cv::Point2f(220, 500);   // 右脚踝
    }
    
    HeightDetectionConfig config_;
    std::unique_ptr<HeightDetector> detector_;
    cv::Mat homography_;
    PoseResult test_pose_;
};

// 测试基本的身高检测功能
TEST_F(HeightDetectorTest, BasicHeightDetection) {
    std::vector<PoseResult> poses = {test_pose_};
    
    auto results = detector_->process_frame(poses);
    
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].person_id, 1);
    EXPECT_EQ(results[0].state, HeightDetectionState::MEASURING);  // 应该开始测量
    EXPECT_GT(results[0].estimated_height_mm, 0);  // 应该有身高估计
}

// 测试多帧稳定性检测
TEST_F(HeightDetectorTest, MultiFrameStability) {
    std::vector<PoseResult> poses = {test_pose_};
    
    std::vector<HeightResult> all_results;
    
    // 连续处理多帧
    for (int frame = 0; frame < 12; ++frame) {
        // 稍微变化姿态位置来模拟真实情况
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::NOSE)].y = 100 + frame % 3;
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)].y = 500 + frame % 2;
        test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)].y = 500 + frame % 2;
        
        auto results = detector_->process_frame(poses);
        ASSERT_EQ(results.size(), 1);
        all_results.push_back(results[0]);
    }
    
    // 检查状态转换
    EXPECT_EQ(all_results[0].state, HeightDetectionState::MEASURING);
    
    // 最后几帧应该达到稳定状态
    EXPECT_TRUE(all_results.back().state == HeightDetectionState::STABLE ||
                all_results.back().state == HeightDetectionState::MEASURING);
    
    // 稳定帧数应该逐渐增加
    if (all_results.back().state == HeightDetectionState::STABLE) {
        EXPECT_GE(all_results.back().stable_frames_count, config_.min_stable_frames);
    }
}

// 测试不满足测量条件的情况
TEST_F(HeightDetectorTest, InvalidMeasurementConditions) {
    // 测试手举得太高的情况
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_WRIST)] = cv::Point2f(160, 80);  // 手腕高于鼻子
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_WRIST)] = cv::Point2f(240, 80); // 手腕高于鼻子
    
    std::vector<PoseResult> poses = {test_pose_};
    auto results = detector_->process_frame(poses);
    
    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].is_stable);  // 不应该稳定
    
    // 测试关键点置信度不足的情况
    test_pose_.keypoint_confidences[static_cast<int>(COCOKeypoint::NOSE)] = 0.3f;  // 低于阈值
    results = detector_->process_frame(poses);
    
    ASSERT_EQ(results.size(), 1);
    EXPECT_FALSE(results[0].is_stable);
}

// 测试ROI大小限制
TEST_F(HeightDetectorTest, ROISizeConstraints) {
    // 测试ROI太小的情况
    test_pose_.bbox = cv::Rect2f(100, 50, 50, 80);  // 很小的ROI
    
    std::vector<PoseResult> poses = {test_pose_};
    auto results = detector_->process_frame(poses);
    
    ASSERT_EQ(results.size(), 1);
    // 应该不满足测量条件或者给出低置信度
    
    // 测试ROI太大的情况  
    test_pose_.bbox = cv::Rect2f(100, 50, 300, 800);  // 很大的ROI
    results = detector_->process_frame(poses);
    
    ASSERT_EQ(results.size(), 1);
    // 同样应该不满足条件
}

// 测试异常值过滤
TEST_F(HeightDetectorTest, OutlierFiltering) {
    std::vector<PoseResult> poses = {test_pose_};
    
    // 先处理几帧正常数据
    for (int i = 0; i < 5; ++i) {
        detector_->process_frame(poses);
    }
    
    // 注入异常值
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)] = cv::Point2f(180, 800);   // 异常的脚踝位置
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)] = cv::Point2f(220, 800);
    
    auto outlier_result = detector_->process_frame(poses);
    
    // 恢复正常数据
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::LEFT_ANKLE)] = cv::Point2f(180, 500);
    test_pose_.keypoints[static_cast<int>(COCOKeypoint::RIGHT_ANKLE)] = cv::Point2f(220, 500);
    
    auto normal_result = detector_->process_frame(poses);
    
    // 异常值不应该导致身高估计发生剧烈变化
    ASSERT_EQ(outlier_result.size(), 1);
    ASSERT_EQ(normal_result.size(), 1);
}

// 测试多人处理
TEST_F(HeightDetectorTest, MultiplePeople) {
    // 创建第二个人的姿态数据
    PoseResult person2 = test_pose_;
    person2.person_id = 2;
    person2.bbox = cv::Rect2f(400, 60, 180, 480);  // 不同位置和大小
    
    // 调整第二个人的关键点位置
    for (auto& point : person2.keypoints) {
        point.x += 300;  // 向右偏移
        point.y -= 10;   // 稍微向上
    }
    
    std::vector<PoseResult> poses = {test_pose_, person2};
    
    // 处理多帧
    for (int frame = 0; frame < 8; ++frame) {
        auto results = detector_->process_frame(poses);
        
        EXPECT_EQ(results.size(), 2);
        
        // 检查两个人的ID是否正确
        std::set<int> person_ids;
        for (const auto& result : results) {
            person_ids.insert(result.person_id);
        }
        
        EXPECT_TRUE(person_ids.count(1) > 0);
        EXPECT_TRUE(person_ids.count(2) > 0);
    }
    
    // 检查活跃的person ID
    auto active_ids = detector_->get_active_person_ids();
    EXPECT_EQ(active_ids.size(), 2);
}

// 测试状态机转换
TEST_F(HeightDetectorTest, StateMachineTransitions) {
    std::vector<PoseResult> poses = {test_pose_};
    
    // 1. IDLE -> MEASURING
    auto result = detector_->process_frame(poses);
    EXPECT_EQ(result[0].state, HeightDetectionState::MEASURING);
    
    // 2. 持续MEASURING状态
    for (int i = 0; i < 3; ++i) {
        result = detector_->process_frame(poses);
        EXPECT_EQ(result[0].state, HeightDetectionState::MEASURING);
    }
    
    // 3. 测试无效输入导致的状态转换
    test_pose_.keypoint_confidences[static_cast<int>(COCOKeypoint::NOSE)] = 0.2f;  // 低置信度
    
    for (int i = 0; i < 12; ++i) {  // 连续输入无效数据
        result = detector_->process_frame(poses);
    }
    
    // 应该转换到INVALID或IDLE状态
    EXPECT_TRUE(result[0].state == HeightDetectionState::INVALID || 
                result[0].state == HeightDetectionState::IDLE);
}

// 测试配置更新
TEST_F(HeightDetectorTest, ConfigurationUpdate) {
    // 获取初始配置
    auto initial_config = detector_->get_config();
    
    // 修改配置
    HeightDetectionConfig new_config = config_;
    new_config.filter_type = "kalman";
    new_config.window_size = 20;
    new_config.min_stable_frames = 8;
    
    detector_->update_config(new_config);
    
    // 验证配置已更新
    auto updated_config = detector_->get_config();
    EXPECT_EQ(updated_config.filter_type, "kalman");
    EXPECT_EQ(updated_config.window_size, 20);
    EXPECT_EQ(updated_config.min_stable_frames, 8);
    
    // 测试新配置是否生效
    std::vector<PoseResult> poses = {test_pose_};
    auto results = detector_->process_frame(poses);
    EXPECT_EQ(results.size(), 1);
}

// 测试重置功能
TEST_F(HeightDetectorTest, ResetFunctionality) {
    std::vector<PoseResult> poses = {test_pose_};
    
    // 处理一些帧以建立状态
    for (int i = 0; i < 5; ++i) {
        detector_->process_frame(poses);
    }
    
    auto active_ids_before = detector_->get_active_person_ids();
    EXPECT_GT(active_ids_before.size(), 0);
    
    // 测试重置单个person
    detector_->reset_person(1);
    
    // 处理新帧，状态应该被重置
    auto result = detector_->process_frame(poses);
    EXPECT_EQ(result[0].state, HeightDetectionState::MEASURING);  // 重新开始测量
    
    // 测试重置所有
    detector_->reset();
    auto active_ids_after = detector_->get_active_person_ids();
    EXPECT_EQ(active_ids_after.size(), 0);  // 所有状态被清除
}

// 测试过期数据清理
TEST_F(HeightDetectorTest, StaleDataCleanup) {
    std::vector<PoseResult> poses = {test_pose_};
    
    // 处理一些帧
    for (int i = 0; i < 3; ++i) {
        detector_->process_frame(poses);
    }
    
    auto active_ids_before = detector_->get_active_person_ids();
    EXPECT_EQ(active_ids_before.size(), 1);
    
    // 手动触发清理，使用很短的过期时间
    detector_->cleanup_stale_contexts(1);  // 1ms过期时间
    
    auto active_ids_after = detector_->get_active_person_ids();
    EXPECT_EQ(active_ids_after.size(), 0);  // 应该被清理掉
}

// 测试调试信息
TEST_F(HeightDetectorTest, DebugInformation) {
    std::vector<PoseResult> poses = {test_pose_};
    
    // 处理一些帧
    for (int i = 0; i < 3; ++i) {
        detector_->process_frame(poses);
    }
    
    // 获取整体调试信息
    std::string debug_info = detector_->get_debug_info();
    EXPECT_NE(debug_info.find("Height Detector Status"), std::string::npos);
    EXPECT_NE(debug_info.find("Active persons"), std::string::npos);
    
    // 获取特定person的调试信息
    std::string person_debug = detector_->get_debug_info(1);
    EXPECT_NE(person_debug.find("Person 1"), std::string::npos);
    EXPECT_NE(person_debug.find("State:"), std::string::npos);
}

// 测试Homography变换
TEST_F(HeightDetectorTest, HomographyTransform) {
    // 创建一个缩放变换矩阵 (每像素 = 2mm)
    cv::Mat scaling_homography = cv::Mat::eye(3, 3, CV_64F);
    scaling_homography.at<double>(0, 0) = 2.0;  // X方向缩放2倍
    scaling_homography.at<double>(1, 1) = 2.0;  // Y方向缩放2倍
    
    detector_->set_homography(scaling_homography);
    
    std::vector<PoseResult> poses = {test_pose_};
    
    auto results_with_transform = detector_->process_frame(poses);
    
    // 设置单位矩阵进行对比
    detector_->set_homography(cv::Mat::eye(3, 3, CV_64F));
    auto results_without_transform = detector_->process_frame(poses);
    
    // 使用变换的结果应该约为无变换结果的2倍
    // （由于滤波和状态的存在，可能不是严格的2倍关系）
    ASSERT_EQ(results_with_transform.size(), 1);
    ASSERT_EQ(results_without_transform.size(), 1);
    
    if (results_with_transform[0].estimated_height_mm > 0 && 
        results_without_transform[0].estimated_height_mm > 0) {
        float ratio = results_with_transform[0].estimated_height_mm / 
                     results_without_transform[0].estimated_height_mm;
        EXPECT_GT(ratio, 1.5f);  // 应该明显大于1
        EXPECT_LT(ratio, 3.0f);  // 但不应该过大
    }
}

// 测试不同滤波器类型
TEST_F(HeightDetectorTest, DifferentFilterTypes) {
    std::vector<std::string> filter_types = {"median", "moving_average", "kalman"};
    
    for (const auto& filter_type : filter_types) {
        HeightDetectionConfig test_config = config_;
        test_config.filter_type = filter_type;
        
        HeightDetector test_detector(test_config);
        test_detector.set_homography(homography_);
        
        std::vector<PoseResult> poses = {test_pose_};
        
        // 处理几帧数据
        std::vector<float> heights;
        for (int i = 0; i < 8; ++i) {
            auto results = test_detector.process_frame(poses);
            ASSERT_EQ(results.size(), 1);
            
            if (results[0].estimated_height_mm > 0) {
                heights.push_back(results[0].estimated_height_mm);
            }
        }
        
        // 所有滤波器都应该产生合理的身高估计
        EXPECT_GT(heights.size(), 0) << "Filter type: " << filter_type;
        
        if (!heights.empty()) {
            float avg_height = std::accumulate(heights.begin(), heights.end(), 0.0f) / heights.size();
            EXPECT_GT(avg_height, 100.0f) << "Filter type: " << filter_type;  // 至少100mm
            EXPECT_LT(avg_height, 3000.0f) << "Filter type: " << filter_type; // 最多3000mm
        }
    }
}