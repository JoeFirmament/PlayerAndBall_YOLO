#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// 直接使用detector_lib的pose检测
#include "PoseDetectorLib.h"
#include "detector_path_utils.h"

using namespace cv;
using namespace std;
using namespace detector;

class DirectHeightMeasurement {
private:
    unique_ptr<PoseDetectorLib> detector_;
    float camera_height_mm_;
    
public:
    DirectHeightMeasurement() : camera_height_mm_(1170.0f) {} // 117cm = 1170mm
    
    // 初始化检测器
    bool initialize(const string& model_path, const string& homography_file) {
        cout << "初始化姿态检测器..." << endl;
        cout << "模型文件: " << model_path << endl;
        cout << "标定文件: " << homography_file << endl;
        
        // 1. 创建检测器
        detector_ = make_unique<PoseDetectorLib>(model_path);
        
        // 2. 配置检测器
        detector_->enable_tracking(true);  // 启用跟踪
        detector_->set_confidence_threshold(0.3f);  // 设置置信度阈值
        
        // 3. 加载homography标定
        bool calib_loaded = detector_->load_calibration(homography_file);
        if (!calib_loaded) {
            cerr << "❌ 无法加载标定文件: " << homography_file << endl;
            return false;
        }
        
        cout << "✓ 检测器初始化成功" << endl;
        return true;
    }
    
    // 获取头部位置（使用nose关键点）
    Point2f get_head_position(const PoseResult& pose) {
        // COCO格式：0=nose
        cout << "🔍 头部关键点检查: nose置信度=" << pose.keypoint_scores[0] << ", 坐标=(" 
             << pose.keypoints[0].x << ", " << pose.keypoints[0].y << ")" << endl;
             
        if (pose.keypoint_scores[0] > 0.5f) {
            cout << "    → 使用nose关键点" << endl;
            return pose.keypoints[0];
        }
        // 如果nose不可见，使用ROI顶部中心
        Point2f roi_head = Point2f(pose.bbox.x + pose.bbox.width * 0.5f, pose.bbox.y);
        cout << "    → nose不可见，使用ROI顶部中心: (" << roi_head.x << ", " << roi_head.y << ")" << endl;
        return roi_head;
    }
    
    // 获取脚部位置（使用ankle关键点的平均）
    Point2f get_foot_position(const PoseResult& pose) {
        // COCO格式：15=left_ankle, 16=right_ankle
        Point2f left_ankle = pose.keypoints[15];
        Point2f right_ankle = pose.keypoints[16];
        
        bool left_valid = pose.keypoint_scores[15] > 0.5f;
        bool right_valid = pose.keypoint_scores[16] > 0.5f;
        
        cout << "🔍 脚部关键点检查:" << endl;
        cout << "    left_ankle: 置信度=" << pose.keypoint_scores[15] << ", 坐标=(" 
             << left_ankle.x << ", " << left_ankle.y << "), 有效=" << (left_valid ? "是" : "否") << endl;
        cout << "    right_ankle: 置信度=" << pose.keypoint_scores[16] << ", 坐标=(" 
             << right_ankle.x << ", " << right_ankle.y << "), 有效=" << (right_valid ? "是" : "否") << endl;
        
        if (left_valid && right_valid) {
            // 两个脚踝都可见，取平均
            Point2f avg_ankle = Point2f((left_ankle.x + right_ankle.x) * 0.5f, 
                                      (left_ankle.y + right_ankle.y) * 0.5f);
            cout << "    → 使用双脚踝平均: (" << avg_ankle.x << ", " << avg_ankle.y << ")" << endl;
            return avg_ankle;
        } else if (left_valid) {
            cout << "    → 使用左脚踝" << endl;
            return left_ankle;
        } else if (right_valid) {
            cout << "    → 使用右脚踝" << endl;
            return right_ankle;
        } else {
            // 脚踝都不可见，使用ROI底部中心
            Point2f roi_foot = Point2f(pose.bbox.x + pose.bbox.width * 0.5f, 
                                     pose.bbox.y + pose.bbox.height);
            cout << "    → 脚踝都不可见，使用ROI底部中心: (" << roi_foot.x << ", " << roi_foot.y << ")" << endl;
            return roi_foot;
        }
    }
    
    // 计算身高（基于homography和相机高度）
    float calculate_height_homography(const PoseResult& pose) {
        // 这里使用detector_lib已经计算好的ground_position
        if (!pose.has_ground_position) {
            cout << "⚠️ 此人员没有地面坐标信息" << endl;
            return -1.0f;
        }
        
        Point2f head_pixel = get_head_position(pose);
        Point2f foot_pixel = get_foot_position(pose);
        
        cout << "\n=== 身高计算过程 DEBUG ===" << endl;
        cout << "🔍 1. ROI框信息: (" << pose.bbox.x << ", " << pose.bbox.y << ", " 
             << pose.bbox.width << ", " << pose.bbox.height << ")" << endl;
        cout << "🔍 2. ROI框高度: " << pose.bbox.height << " pixels" << endl;
        cout << "🔍 3. 头部像素: (" << head_pixel.x << ", " << head_pixel.y << ")" << endl;
        cout << "🔍 4. 脚部像素: (" << foot_pixel.x << ", " << foot_pixel.y << ")" << endl;
        cout << "🔍 5. 头脚像素高度差: " << abs(head_pixel.y - foot_pixel.y) << " pixels" << endl;
        
        // 🔥 关键：脚部通过homography映射到Z=0地面
        Point2f foot_world = pose.ground_position;
        cout << "🔍 6. 脚部地面坐标 (Z=0): (" << foot_world.x << ", " << foot_world.y << ") mm" << endl;
        
        // 🔥 关键：头部也投影到Z=0地面（这只是投影，不是真实位置）
        vector<Point2f> head_pixels = {head_pixel};
        vector<Point2f> head_world_projected;
        
        // 需要获取homography矩阵来手动转换头部坐标
        // 由于detector_lib已经有了foot的地面坐标，我们可以通过像素差异来计算
        
        // 计算头部投影到地面的位置
        Point2f head_world_projected_pos = foot_world; // 先假设在同一位置
        
        // 方法：利用脚部已知的像素->世界坐标转换比例
        // 计算脚部位置的像素尺度
        Point2f foot_pixel_offset = Point2f(foot_pixel.x + 1, foot_pixel.y); // 脚部位置向右1像素
        
        // 使用简化方法：假设在脚部附近像素尺度相对均匀
        float pixel_to_mm_ratio = 1.0f; // 默认值，需要通过实际标定确定
        
        // 头部相对于脚部的像素偏移
        float head_foot_pixel_offset_x = head_pixel.x - foot_pixel.x;
        float head_foot_pixel_offset_y = head_pixel.y - foot_pixel.y;
        
        cout << "🔍 7. 头部相对脚部像素偏移: (" << head_foot_pixel_offset_x << ", " << head_foot_pixel_offset_y << ")" << endl;
        
        // 估算像素到毫米的比例（基于距离）
        float distance_to_camera = sqrt(foot_world.x * foot_world.x + foot_world.y * foot_world.y);
        cout << "🔍 8. 脚部到相机初步距离: " << distance_to_camera << " mm" << endl;
        
        // 相机视野角度假设（需要根据实际相机参数调整）
        float fov_horizontal = 60.0f; // 度
        float fov_vertical = 40.0f;   // 度
        
        cout << "🔍 9. 相机视野角度: 水平" << fov_horizontal << "°, 垂直" << fov_vertical << "°" << endl;
        
        // 图像尺寸假设（需要从实际图像获取）
        float image_width = 1280.0f;  
        float image_height = 720.0f;
        
        cout << "🔍 10. 图像尺寸: " << image_width << "x" << image_height << endl;
        
        // 在脚部距离处的实际视野宽度和高度
        float real_width_at_distance = 2 * distance_to_camera * tan(fov_horizontal * M_PI / 360.0f);
        float real_height_at_distance = 2 * distance_to_camera * tan(fov_vertical * M_PI / 360.0f);
        
        cout << "🔍 11. 脚部距离处实际视野: 宽度" << real_width_at_distance << "mm, 高度" << real_height_at_distance << "mm" << endl;
        
        // 像素到毫米的转换比例
        float pixel_to_mm_x = real_width_at_distance / image_width;
        float pixel_to_mm_y = real_height_at_distance / image_height;
        
        cout << "🔍 12. 像素到毫米比例: X轴=" << pixel_to_mm_x << "mm/pixel, Y轴=" << pixel_to_mm_y << "mm/pixel" << endl;
        
        // 头部在地面的投影位置
        head_world_projected_pos.x = foot_world.x + head_foot_pixel_offset_x * pixel_to_mm_x;
        head_world_projected_pos.y = foot_world.y + head_foot_pixel_offset_y * pixel_to_mm_y;
        
        cout << "🔍 13. 头部地面投影计算:" << endl;
        cout << "    头部X = " << foot_world.x << " + " << head_foot_pixel_offset_x << " * " << pixel_to_mm_x << " = " << head_world_projected_pos.x << endl;
        cout << "    头部Y = " << foot_world.y << " + " << head_foot_pixel_offset_y << " * " << pixel_to_mm_y << " = " << head_world_projected_pos.y << endl;
        cout << "🔍 14. 头部地面投影坐标 (Z=0): (" << head_world_projected_pos.x << ", " << head_world_projected_pos.y << ") mm" << endl;
        
        // 🔥 核心：相似三角形计算身高
        // 头部投影与脚部在地面的距离
        float ground_distance = sqrt(pow(head_world_projected_pos.x - foot_world.x, 2) + 
                                   pow(head_world_projected_pos.y - foot_world.y, 2));
        
        // 🔥 修正：相机在世界坐标系中的位置是 (0, -400mm, 1170mm)
        float camera_world_x = 0.0f;
        float camera_world_y = -400.0f;  // 相机Y坐标：-40cm
        
        // 脚部到相机的水平距离（在地面投影）
        float horizontal_distance = sqrt(pow(foot_world.x - camera_world_x, 2) + 
                                       pow(foot_world.y - camera_world_y, 2));
        
        // 脚部到相机的真实3D距离
        float foot_distance_to_camera = sqrt(horizontal_distance * horizontal_distance + 
                                           camera_height_mm_ * camera_height_mm_);
        
        cout << "🔍 15. 头部投影与脚部地面距离: " << ground_distance << " mm" << endl;
        cout << "🔍 16. 脚部到相机水平距离: " << horizontal_distance << " mm" << endl;
        cout << "🔍 17. 脚部到相机3D距离: " << foot_distance_to_camera << " mm" << endl;
        cout << "🔍 18. 相机高度: " << camera_height_mm_ << " mm" << endl;
        
        // 🔥 正确的相似三角形计算：
        // 相机高度 / 脚部到相机水平距离 = 人的身高 / 头部投影与脚部的地面距离
        // person_height = camera_height * ground_distance / horizontal_distance
        cout << "🔍 19. 相似三角形计算过程:" << endl;
        cout << "    公式: 人的身高 = 相机高度 × 地面距离 / 水平距离" << endl;
        cout << "    计算: " << camera_height_mm_ << " × " << ground_distance << " / " << horizontal_distance << endl;
        
        float estimated_height = camera_height_mm_ * ground_distance / horizontal_distance;
        
        cout << "🔍 20. 最终计算身高: " << estimated_height << " mm = " << estimated_height/10.0f << " cm" << endl;
        
        return estimated_height;
    }
    
    // 绘制骨架连接线
    void draw_skeleton(Mat& image, const PoseResult& pose) {
        // COCO 17点骨架连接定义
        vector<pair<int, int>> connections = {
            {5, 6},   // 左肩-右肩
            {5, 7}, {7, 9},   // 左肩-左肘-左腕
            {6, 8}, {8, 10},  // 右肩-右肘-右腕
            {5, 11}, {6, 12}, {11, 12}, // 肩膀到臀部
            {11, 13}, {13, 15}, // 左侧腿部
            {12, 14}, {14, 16}  // 右侧腿部
        };
        
        for (const auto& conn : connections) {
            int p1 = conn.first, p2 = conn.second;
            if (p1 < pose.keypoint_scores.size() && p2 < pose.keypoint_scores.size() &&
                pose.keypoint_scores[p1] > 0.5f && pose.keypoint_scores[p2] > 0.5f) {
                line(image, pose.keypoints[p1], pose.keypoints[p2], 
                     Scalar(0, 255, 255), 2);  // 黄色骨架
            }
        }
    }
    
    // 处理图片并生成结果
    Mat process_image(const Mat& image, const string& output_path = "") {
        cout << "\n=== 直接使用detector_lib的姿态检测和身高测量 ===" << endl;
        cout << "图片尺寸: " << image.cols << "x" << image.rows << endl;
        
        // 1. 进行姿态检测（detector_lib自动处理homography转换）
        auto pose_results = detector_->detect(image);
        cout << "✓ 检测到 " << pose_results.size() << " 个人" << endl;
        cout << "✓ 推理时间: " << detector_->get_last_inference_time_ms() << "ms" << endl;
        
        if (pose_results.empty()) {
            cout << "  无姿态检测结果" << endl;
            return image.clone();
        }
        
        // 2. 创建结果图片
        Mat result_image = image.clone();
        
        for (size_t i = 0; i < pose_results.size(); ++i) {
            const auto& pose = pose_results[i];
            
            cout << "\n--- Person " << (i+1) << " ---" << endl;
            cout << "ID: " << pose.person_id << ", 置信度: " << pose.confidence << endl;
            
            // 绘制ROI框（紫色）
            Scalar roi_color(255, 0, 255);
            rectangle(result_image, pose.bbox, roi_color, 3);
            
            // 绘制关键点（带调试信息）
            cout << "🔍 绘制关键点调试信息:" << endl;
            for (size_t j = 0; j < pose.keypoints.size(); ++j) {
                Point2f kp = pose.keypoints[j];
                float score = pose.keypoint_scores[j];
                
                // 检查关键点是否在图像范围内
                bool in_image = (kp.x >= 0 && kp.x < result_image.cols && 
                               kp.y >= 0 && kp.y < result_image.rows);
                
                cout << "    关键点[" << j << "]: 坐标(" << kp.x << ", " << kp.y 
                     << "), 置信度=" << score << ", 在图像内=" << (in_image ? "是" : "否") << endl;
                
                // 只绘制在图像范围内且置信度高的关键点
                if (score > 0.5f && in_image) {
                    // 根据关键点类型使用不同颜色
                    Scalar color;
                    if (j == 0) color = Scalar(255, 0, 255);      // nose - 紫色
                    else if (j >= 1 && j <= 4) color = Scalar(0, 255, 255);  // 眼睛耳朵 - 黄色
                    else if (j >= 5 && j <= 10) color = Scalar(0, 255, 0);   // 上身 - 绿色
                    else if (j >= 11 && j <= 16) color = Scalar(255, 0, 0);  // 下身 - 蓝色
                    
                    circle(result_image, kp, 6, color, -1);
                    // 标注关键点编号
                    putText(result_image, to_string(j), Point(kp.x + 8, kp.y - 8),
                           FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
                }
            }
            
            // 特别标记我们用于身高计算的头脚点
            Point2f head_used = get_head_position(pose);
            Point2f foot_used = get_foot_position(pose);
            
            // 检查计算用的头脚点是否在图像内
            bool head_in_image = (head_used.x >= 0 && head_used.x < result_image.cols && 
                                head_used.y >= 0 && head_used.y < result_image.rows);
            bool foot_in_image = (foot_used.x >= 0 && foot_used.x < result_image.cols && 
                                foot_used.y >= 0 && foot_used.y < result_image.rows);
            
            cout << "🔍 用于计算的点:" << endl;
            cout << "    头部点: (" << head_used.x << ", " << head_used.y << "), 在图像内=" << (head_in_image ? "是" : "否") << endl;
            cout << "    脚部点: (" << foot_used.x << ", " << foot_used.y << "), 在图像内=" << (foot_in_image ? "是" : "否") << endl;
            
            // 用特殊标记绘制计算用的头脚点
            if (head_in_image) {
                circle(result_image, head_used, 10, Scalar(0, 0, 255), 3);  // 红色圆圈 - 头部
                putText(result_image, "HEAD", Point(head_used.x + 15, head_used.y - 15),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
            }
            if (foot_in_image) {
                circle(result_image, foot_used, 10, Scalar(255, 255, 0), 3);  // 青色圆圈 - 脚部
                putText(result_image, "FOOT", Point(foot_used.x + 15, foot_used.y + 25),
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            }
            
            // 绘制骨架
            draw_skeleton(result_image, pose);
            
            // 计算身高
            float height_mm = calculate_height_homography(pose);
            
            // 准备显示文本
            string roi_text = cv::format("Person%d (ID:%d, %.2f)", 
                                        (int)(i+1), pose.person_id, pose.confidence);
            
            string height_text;
            Scalar text_color;
            if (height_mm > 0) {
                float height_cm = height_mm / 10.0f;
                height_text = cv::format("Height: %.1fcm", height_cm);
                text_color = Scalar(0, 255, 0);  // 绿色
                cout << "✓ 身高: " << height_cm << "cm" << endl;
            } else {
                height_text = "Height: Cannot calculate";
                text_color = Scalar(0, 0, 255);  // 红色
                cout << "❌ 身高: 无法计算" << endl;
            }
            
            // 绘制文字标签
            Point text_org(pose.bbox.x, pose.bbox.y - 10);
            
            // ROI标签背景
            Size roi_size = getTextSize(roi_text, FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            rectangle(result_image, 
                     text_org + Point(-3, 3),
                     text_org + Point(roi_size.width + 3, -roi_size.height - 3),
                     Scalar(0, 0, 0), -1);
            putText(result_image, roi_text, text_org, 
                   FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2);
            
            // 身高标签背景  
            Point height_org(pose.bbox.x, pose.bbox.y - 35);
            Size height_size = getTextSize(height_text, FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            rectangle(result_image, 
                     height_org + Point(-3, 3),
                     height_org + Point(height_size.width + 3, -height_size.height - 3),
                     Scalar(0, 0, 0), -1);
            putText(result_image, height_text, height_org, 
                   FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
            
            // 显示世界坐标（如果有）
            if (pose.has_ground_position) {
                cout << "世界坐标: (" << pose.ground_position.x << ", " 
                     << pose.ground_position.y << ")mm" << endl;
                     
                string world_text = cv::format("World: (%.0f,%.0f)mm", 
                                              pose.ground_position.x, pose.ground_position.y);
                Point world_org(pose.bbox.x, pose.bbox.y + pose.bbox.height + 20);
                Size world_size = getTextSize(world_text, FONT_HERSHEY_SIMPLEX, 0.4, 1, nullptr);
                rectangle(result_image, 
                         world_org + Point(-2, 2),
                         world_org + Point(world_size.width + 2, -world_size.height - 2),
                         Scalar(0, 0, 0), -1);
                putText(result_image, world_text, world_org, 
                       FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255, 255, 255), 1);
            }
        }
        
        // 添加标题
        putText(result_image, "Direct Pose Detection + Height Measurement", 
                Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        putText(result_image, "Using detector_lib + Homography transformation", 
                Point(50, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 1);
        
        // 保存结果
        if (!output_path.empty()) {
            bool saved = imwrite(output_path, result_image);
            if (saved) {
                cout << "\n✅ 结果图片已保存到: " << output_path << endl;
            } else {
                cout << "\n❌ 保存结果图片失败！" << endl;
            }
        }
        
        return result_image;
    }
};

int main() {
    cout << "=== 基于detector_lib的直接姿态检测和身高测量 ===" << endl;
    
    try {
        DirectHeightMeasurement height_tester;
        
        // 1. 初始化
        string model_path = "../../models/Q_yolov8_pose.rknn";
        string homography_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/data/2025_8_6_1280_720.json";
        
        if (!height_tester.initialize(model_path, homography_file)) {
            cerr << "❌ 初始化失败" << endl;
            return -1;
        }
        
        // 2. 测试图片
        vector<string> image_files = {
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose.jpg",
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose1.jpg", 
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose2.jpg",
            "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose3.jpg"
        };
        
        for (size_t i = 0; i < image_files.size(); ++i) {
            cout << "\n" << string(60, '=') << endl;
            cout << "测试图片 " << (i+1) << ": " << image_files[i] << endl;
            cout << string(60, '=') << endl;
            
            // 加载图片
            Mat image = imread(image_files[i]);
            if (image.empty()) {
                cout << "❌ 无法加载图片: " << image_files[i] << endl;
                continue;
            }
            
            cout << "✓ 图片加载成功 (" << image.cols << "x" << image.rows << ")" << endl;
            
            // 处理并保存结果
            string output_filename = cv::format("direct_height_result_%d.jpg", (int)(i+1));
            height_tester.process_image(image, output_filename);
        }
        
        cout << "\n" << string(60, '=') << endl;
        cout << "✅ 所有图片测试完成" << endl;
        cout << "🎉 成功使用detector_lib进行：Pose检测 → ROI可视化 → 身高测量" << endl;
        
    } catch (const std::exception& e) {
        cerr << "错误: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}