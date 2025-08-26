# 姿态分析功能实施计划

## 1. 功能需求

### 1.1 身高检测功能
- **目标**: 检测person的真实身高（单位：毫米）
- **触发条件**: 仅当person的双手（手腕）不超过头部（头顶）时进行测量
- **实现原理**: 
  - 使用ROI边界框高度（像素）
  - 通过Homography矩阵转换为真实世界高度
  - 考虑相机透视畸变的影响

### 1.2 篮球要球动作识别
- **目标**: 识别篮球比赛中的要球手势
- **判定条件**:
  - 双手手腕在胸部区域前方
  - 双手手腕距离小于设定阈值
  - 手臂呈现张开状态
- **胸部区域定义**: 肩部关键点和髋部关键点之间的区域

### 1.3 要球者ID优先级管理
- **目标**: 当多人同时做要球动作时，动态调整跟踪ID
- **规则**:
  - 检测到要球动作的person自动获得ID 0
  - 原ID 0的person如果不在要球，则分配新的ID
  - 支持多人要球时的优先级排序（如基于置信度或持续时间）

## 2. 技术架构

### 2.1 基于detector_lib的扩展
```
detector_lib/
├── include/
│   ├── PoseDetectorLib.h      (已存在)
│   ├── detector_types.h       (已存在) 
│   └── PoseAnalyzer.h         (新增)
├── src/
│   ├── PoseDetectorLib.cpp    (已存在)
│   └── PoseAnalyzer.cpp       (新增)
├── data/
│   └── pose_analysis_config.json (新增)
└── examples/
    └── test_pose_analyzer.cpp (新增)
```

### 2.2 关键数据结构

#### 扩展PoseResult结构体
```cpp
// 在detector_types.h中扩展
struct PoseAnalysisResult {
    // 身高检测
    float estimated_height_mm;      // 估算身高（毫米）
    bool height_valid;              // 身高测量是否有效
    
    // 要球动作检测
    bool is_requesting_ball;        // 是否在做要球动作
    float request_confidence;       // 要球动作置信度 [0-1]
    float hands_distance_mm;        // 双手距离（毫米）
    
    // ID管理
    int original_id;                // 原始跟踪ID
    int priority_id;                // 优先级调整后的ID
    int request_duration_frames;    // 要球动作持续帧数
};
```

## 3. 实现细节

### 3.1 身高检测算法

#### 步骤1: 判断测量条件
```cpp
bool can_measure_height(const PoseResult& pose) {
    // 获取关键点
    auto left_wrist = pose.keypoints[LEFT_WRIST];
    auto right_wrist = pose.keypoints[RIGHT_WRIST];
    auto nose = pose.keypoints[NOSE];
    
    // 检查关键点置信度
    if (pose.keypoint_scores[LEFT_WRIST] < min_confidence ||
        pose.keypoint_scores[RIGHT_WRIST] < min_confidence ||
        pose.keypoint_scores[NOSE] < min_confidence) {
        return false;
    }
    
    // 判断手腕是否低于头部
    float head_y = nose.y - head_offset_pixels;  // 头顶位置估算
    return (left_wrist.y > head_y && right_wrist.y > head_y);
}
```

#### 步骤2: 计算真实身高
```cpp
float calculate_real_height(const cv::Rect& bbox, 
                           const cv::Mat& homography_matrix,
                           const CalibrationParams& calib) {
    // ROI框的顶部和底部点
    cv::Point2f top_point(bbox.x + bbox.width/2, bbox.y);
    cv::Point2f bottom_point(bbox.x + bbox.width/2, bbox.y + bbox.height);
    
    // 转换到世界坐标
    cv::Point2f world_top = apply_homography(top_point, homography_matrix);
    cv::Point2f world_bottom = apply_homography(bottom_point, homography_matrix);
    
    // 计算垂直高度（考虑地面倾斜）
    float height_mm = calculate_vertical_distance(world_top, world_bottom, calib);
    
    // 应用校正因子
    height_mm *= height_correction_factor;
    
    return height_mm;
}
```

### 3.2 要球动作识别算法

#### 步骤1: 定义胸部区域
```cpp
cv::Rect get_chest_region(const PoseResult& pose) {
    // 获取肩部和髋部关键点
    auto left_shoulder = pose.keypoints[LEFT_SHOULDER];
    auto right_shoulder = pose.keypoints[RIGHT_SHOULDER];
    auto left_hip = pose.keypoints[LEFT_HIP];
    auto right_hip = pose.keypoints[RIGHT_HIP];
    
    // 计算胸部中心和范围
    float chest_center_x = (left_shoulder.x + right_shoulder.x) / 2;
    float chest_center_y = (left_shoulder.y + left_hip.y) / 2;
    float chest_width = abs(right_shoulder.x - left_shoulder.x) * 1.5;
    float chest_height = abs(left_hip.y - left_shoulder.y);
    
    return cv::Rect(chest_center_x - chest_width/2,
                    chest_center_y - chest_height/2,
                    chest_width, chest_height);
}
```

#### 步骤2: 判断要球动作
```cpp
bool detect_ball_request(const PoseResult& pose, const AnalysisConfig& config) {
    // 获取手腕位置
    auto left_wrist = pose.keypoints[LEFT_WRIST];
    auto right_wrist = pose.keypoints[RIGHT_WRIST];
    
    // 检查置信度
    if (pose.keypoint_scores[LEFT_WRIST] < config.min_keypoint_confidence ||
        pose.keypoint_scores[RIGHT_WRIST] < config.min_keypoint_confidence) {
        return false;
    }
    
    // 获取胸部区域
    cv::Rect chest_region = get_chest_region(pose);
    
    // 检查手腕是否在胸部区域
    bool left_in_chest = chest_region.contains(left_wrist);
    bool right_in_chest = chest_region.contains(right_wrist);
    
    if (!left_in_chest || !right_in_chest) {
        return false;
    }
    
    // 计算双手距离
    float hands_distance = cv::norm(left_wrist - right_wrist);
    
    // 转换为世界坐标距离（如果有homography）
    if (pose.has_ground_position) {
        hands_distance = convert_to_world_distance(hands_distance, homography);
    }
    
    // 判断是否满足要球条件
    return hands_distance < config.max_hands_distance_mm;
}
```

### 3.3 ID优先级管理算法

#### 步骤1: 收集要球状态
```cpp
struct RequestInfo {
    int person_id;
    bool is_requesting;
    float confidence;
    int duration_frames;
    float priority_score;
};

std::vector<RequestInfo> collect_request_states(
    const std::vector<PoseAnalysisResult>& results,
    const std::map<int, int>& history) {
    
    std::vector<RequestInfo> requests;
    
    for (const auto& result : results) {
        RequestInfo info;
        info.person_id = result.original_id;
        info.is_requesting = result.is_requesting_ball;
        info.confidence = result.request_confidence;
        info.duration_frames = history.count(info.person_id) ? 
                              history.at(info.person_id) : 0;
        
        // 计算优先级分数
        info.priority_score = calculate_priority(info);
        requests.push_back(info);
    }
    
    return requests;
}
```

#### 步骤2: 重新分配ID
```cpp
void reassign_ids(std::vector<PoseAnalysisResult>& results,
                  const std::vector<RequestInfo>& requests) {
    
    // 按优先级排序要球者
    auto requesting = filter_requesting(requests);
    std::sort(requesting.begin(), requesting.end(),
              [](const RequestInfo& a, const RequestInfo& b) {
                  return a.priority_score > b.priority_score;
              });
    
    // 分配ID 0给最高优先级的要球者
    if (!requesting.empty()) {
        int top_requester_id = requesting[0].person_id;
        
        // 找到原来的ID 0
        int old_id_0 = find_person_with_id(results, 0);
        
        // 交换ID
        if (old_id_0 != top_requester_id) {
            swap_ids(results, old_id_0, top_requester_id);
        }
    }
}
```

## 4. 配置文件格式

### pose_analysis_config.json
```json
{
    "height_detection": {
        "enabled": true,
        "min_keypoint_confidence": 0.5,
        "head_offset_pixels": 30,
        "height_correction_factor": 1.05,
        "min_roi_height_pixels": 100,
        "max_roi_height_pixels": 800
    },
    
    "ball_request_detection": {
        "enabled": true,
        "min_keypoint_confidence": 0.5,
        "max_hands_distance_mm": 400,
        "chest_region_scale": 1.5,
        "min_duration_frames": 3,
        "confidence_threshold": 0.7
    },
    
    "id_management": {
        "enabled": true,
        "priority_weights": {
            "confidence": 0.4,
            "duration": 0.3,
            "distance_to_center": 0.3
        },
        "id_swap_cooldown_frames": 10,
        "max_tracked_persons": 10
    },
    
    "debug": {
        "show_chest_region": true,
        "show_height_lines": true,
        "show_hands_distance": true,
        "log_id_changes": true
    }
}
```

## 5. 测试程序示例

### test_pose_analyzer.cpp
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
    
    // 初始化分析器
    PoseAnalyzer analyzer("../data/pose_analysis_config.json");
    analyzer.set_homography(detector.get_homography_matrix());
    
    // 打开摄像头
    cv::VideoCapture cap(0);
    
    while (true) {
        cv::Mat frame;
        cap >> frame;
        
        // 姿态检测
        auto pose_results = detector.detect(frame);
        
        // 姿态分析
        auto analysis_results = analyzer.analyze(pose_results);
        
        // 显示结果
        for (const auto& result : analysis_results) {
            std::cout << "Person " << result.priority_id << ": ";
            
            if (result.height_valid) {
                std::cout << "Height=" << result.estimated_height_mm << "mm ";
            }
            
            if (result.is_requesting_ball) {
                std::cout << "REQUESTING_BALL ";
            }
            
            if (result.priority_id != result.original_id) {
                std::cout << "(ID changed from " << result.original_id << ")";
            }
            
            std::cout << std::endl;
        }
        
        // 绘制可视化
        analyzer.draw_analysis(frame, analysis_results);
        
        cv::imshow("Pose Analysis", frame);
        if (cv::waitKey(1) == 27) break;
    }
    
    return 0;
}
```

## 6. API设计

### PoseAnalyzer类接口
```cpp
class PoseAnalyzer {
public:
    // 构造函数
    PoseAnalyzer(const std::string& config_file);
    
    // 配置方法
    void set_homography(const cv::Mat& homography_matrix);
    void set_calibration(const CalibrationParams& params);
    void update_config(const std::string& config_file);
    
    // 分析方法
    std::vector<PoseAnalysisResult> analyze(
        const std::vector<PoseResult>& pose_results);
    
    // 单人分析
    PoseAnalysisResult analyze_single(const PoseResult& pose);
    
    // 功能开关
    void enable_height_detection(bool enable);
    void enable_ball_request_detection(bool enable);
    void enable_id_management(bool enable);
    
    // 参数设置
    void set_height_params(const HeightDetectionParams& params);
    void set_request_params(const BallRequestParams& params);
    void set_id_management_params(const IDManagementParams& params);
    
    // 可视化
    void draw_analysis(cv::Mat& frame, 
                      const std::vector<PoseAnalysisResult>& results);
    
    // 状态查询
    bool is_initialized() const;
    std::string get_last_error() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
```

## 7. 关键技术点

### 7.1 坐标系转换
- **像素坐标 → 世界坐标**: 使用Homography矩阵
- **世界坐标单位**: 毫米（mm）
- **坐标原点**: 可配置，通常为场地中心或摄像头正下方

### 7.2 时序分析
- **动作持续性判断**: 使用滑动窗口记录历史状态
- **防抖动处理**: 设置最小持续帧数阈值
- **ID切换冷却**: 防止频繁ID交换

### 7.3 多人场景处理
- **优先级计算**: 综合置信度、持续时间、位置等因素
- **ID连续性**: 保持非要球者的ID稳定性
- **冲突解决**: 多人同时要球时的仲裁机制

## 8. 性能优化

### 8.1 计算优化
- 仅对有效关键点进行计算
- 使用查找表加速距离计算
- 缓存Homography变换结果

### 8.2 内存优化
- 使用对象池管理分析结果
- 限制历史记录队列长度
- 按需加载配置参数

### 8.3 实时性保证
- 目标处理时间: <5ms per frame
- 支持多线程并行分析
- 异步ID管理更新

## 9. 测试计划

### 9.1 单元测试
- 身高检测准确性测试
- 要球动作识别率测试
- ID管理逻辑测试

### 9.2 集成测试
- 与PoseDetectorLib的集成
- 多人场景测试
- 长时间运行稳定性测试

### 9.3 性能测试
- 处理延迟测试
- CPU/内存占用测试
- 并发处理能力测试

## 10. 部署建议

### 10.1 参数调优
- 根据实际场地调整身高校正因子
- 根据球员习惯调整要球动作阈值
- 根据比赛规则调整ID优先级权重

### 10.2 场景适配
- **训练场**: 降低要球动作阈值，提高灵敏度
- **比赛场**: 提高阈值，减少误判
- **青少年**: 调整身高检测范围

### 10.3 调试工具
- 实时参数调整界面
- 可视化调试输出
- 日志记录和回放

## 11. 扩展功能规划

### 11.1 更多动作识别
- 投篮动作
- 传球动作
- 防守姿态

### 11.2 团队战术分析
- 球员站位分析
- 跑动路线追踪
- 配合默契度评估

### 11.3 数据统计
- 要球次数统计
- 身高变化趋势（疲劳检测）
- 动作模式学习

## 12. 注意事项

### 12.1 隐私保护
- 不存储个人生物特征
- 数据本地处理
- 可选的匿名化输出

### 12.2 准确性声明
- 身高测量仅供参考，受相机角度影响
- 动作识别基于关键点，可能存在误判
- ID管理为辅助功能，不影响基础检测

### 12.3 使用限制
- 需要正确的相机标定
- 最佳检测距离: 3-10米
- 建议帧率: ≥25 FPS