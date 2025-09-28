#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

// 简化的姿态结果结构（模拟真实检测结果）
struct SimplePoseResult {
    int person_id;
    Rect2f bbox;              // ROI框
    vector<Point2f> keypoints; // 17个关键点 
    vector<float> keypoint_scores;
    float confidence;
    Point2f ground_position;  // 世界坐标
    bool has_ground_position;
    
    SimplePoseResult() : person_id(-1), confidence(0.0f), has_ground_position(false) {
        keypoints.resize(17);
        keypoint_scores.resize(17, 0.0f);
    }
};

class SimpleHeightMeasurementTester {
private:
    Mat homography_matrix_;
    float camera_height_mm_;
    
public:
    SimpleHeightMeasurementTester() : camera_height_mm_(1170.0f) {} // 117cm = 1170mm
    
    // 加载Homography矩阵
    bool load_homography(const string& json_file) {
        ifstream file(json_file);
        if (!file.is_open()) {
            cerr << "无法打开文件: " << json_file << endl;
            return false;
        }
        
        Json::Value root;
        file >> root;
        
        // 提取3x3矩阵
        homography_matrix_ = Mat::zeros(3, 3, CV_64F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                homography_matrix_.at<double>(i, j) = root["matrix"][i][j].asDouble();
            }
        }
        
        cout << "Homography矩阵加载成功:" << endl << homography_matrix_ << endl;
        return true;
    }
    
    // 像素坐标转换为地面世界坐标
    Point2f pixel_to_world(const Point2f& pixel) {
        vector<Point2f> src_points = {pixel};
        vector<Point2f> dst_points;
        
        perspectiveTransform(src_points, dst_points, homography_matrix_);
        return dst_points[0];
    }
    
    // 生成模拟的姿态检测结果（基于图片内容推测）
    vector<SimplePoseResult> simulate_pose_detection(const Mat& image) {
        vector<SimplePoseResult> results;
        
        // 模拟检测到的人员（基于图片分析的典型位置）
        vector<Rect2f> estimated_bboxes;
        
        if (image.cols >= 1280 && image.rows >= 720) {
            // 假设这是1280x720的图片，人员通常在这些位置
            estimated_bboxes.push_back(Rect2f(300, 100, 200, 500));   // Person 1
            estimated_bboxes.push_back(Rect2f(600, 80, 180, 520));    // Person 2  
            estimated_bboxes.push_back(Rect2f(900, 90, 190, 510));    // Person 3
        } else {
            // 其他分辨率的适应性位置
            float scale_x = image.cols / 1280.0f;
            float scale_y = image.rows / 720.0f;
            
            estimated_bboxes.push_back(Rect2f(300*scale_x, 100*scale_y, 200*scale_x, 500*scale_y));
            estimated_bboxes.push_back(Rect2f(600*scale_x, 80*scale_y, 180*scale_x, 520*scale_y));
        }
        
        // 为每个bbox生成姿态结果
        for (size_t i = 0; i < estimated_bboxes.size(); ++i) {
            const auto& bbox = estimated_bboxes[i];
            
            // 检查bbox是否在图片范围内
            if (bbox.x + bbox.width > image.cols || bbox.y + bbox.height > image.rows) {
                continue;
            }
            
            SimplePoseResult pose;
            pose.person_id = (int)i + 1;
            pose.bbox = bbox;
            pose.confidence = 0.85f + i * 0.05f;  // 模拟不同的置信度
            
            // 生成关键点（基于ROI的相对位置）
            generate_keypoints_in_roi(pose, bbox);
            
            // 计算脚部的世界坐标
            Point2f foot_center = get_foot_position(pose);
            pose.ground_position = pixel_to_world(foot_center);
            pose.has_ground_position = true;
            
            results.push_back(pose);
        }
        
        cout << "✓ 模拟姿态检测完成，检测到 " << results.size() << " 个人" << endl;
        return results;
    }
    
    // 在ROI内生成合理的关键点
    void generate_keypoints_in_roi(SimplePoseResult& pose, const Rect2f& bbox) {
        // COCO 17点索引
        // 0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear
        // 5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow
        // 9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip
        // 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle
        
        float cx = bbox.x + bbox.width * 0.5f;   // 中心X
        float head_y = bbox.y + bbox.height * 0.1f;  // 头部Y
        float shoulder_y = bbox.y + bbox.height * 0.25f; // 肩膀Y
        float hip_y = bbox.y + bbox.height * 0.6f;    // 髋部Y
        float knee_y = bbox.y + bbox.height * 0.8f;   // 膝盖Y
        float ankle_y = bbox.y + bbox.height * 0.95f; // 脚踝Y
        
        float shoulder_width = bbox.width * 0.3f;
        float hip_width = bbox.width * 0.25f;
        
        // 头部关键点
        pose.keypoints[0] = Point2f(cx, head_y);                    // nose
        pose.keypoints[1] = Point2f(cx - 15, head_y - 10);         // left_eye
        pose.keypoints[2] = Point2f(cx + 15, head_y - 10);         // right_eye
        pose.keypoints[3] = Point2f(cx - 25, head_y);              // left_ear
        pose.keypoints[4] = Point2f(cx + 25, head_y);              // right_ear
        
        // 上身关键点
        pose.keypoints[5] = Point2f(cx - shoulder_width, shoulder_y);  // left_shoulder
        pose.keypoints[6] = Point2f(cx + shoulder_width, shoulder_y);  // right_shoulder
        pose.keypoints[7] = Point2f(cx - shoulder_width - 20, shoulder_y + 80); // left_elbow
        pose.keypoints[8] = Point2f(cx + shoulder_width + 20, shoulder_y + 80); // right_elbow
        pose.keypoints[9] = Point2f(cx - shoulder_width - 30, shoulder_y + 160); // left_wrist
        pose.keypoints[10] = Point2f(cx + shoulder_width + 30, shoulder_y + 160); // right_wrist
        
        // 下身关键点
        pose.keypoints[11] = Point2f(cx - hip_width, hip_y);        // left_hip
        pose.keypoints[12] = Point2f(cx + hip_width, hip_y);        // right_hip
        pose.keypoints[13] = Point2f(cx - hip_width, knee_y);       // left_knee
        pose.keypoints[14] = Point2f(cx + hip_width, knee_y);       // right_knee
        pose.keypoints[15] = Point2f(cx - hip_width, ankle_y);      // left_ankle
        pose.keypoints[16] = Point2f(cx + hip_width, ankle_y);      // right_ankle
        
        // 设置所有关键点的置信度
        for (size_t i = 0; i < pose.keypoint_scores.size(); ++i) {
            pose.keypoint_scores[i] = 0.8f + (rand() % 20) * 0.01f;  // 0.8-1.0的随机置信度
        }
    }
    
    // 获取脚部中心位置
    Point2f get_foot_position(const SimplePoseResult& pose) {
        // 使用左右脚踝的中点作为脚部位置
        Point2f left_ankle = pose.keypoints[15];   // left_ankle
        Point2f right_ankle = pose.keypoints[16];  // right_ankle
        
        return Point2f((left_ankle.x + right_ankle.x) * 0.5f, 
                      (left_ankle.y + right_ankle.y) * 0.5f);
    }
    
    // 获取头部位置
    Point2f get_head_position(const SimplePoseResult& pose) {
        return pose.keypoints[0];  // nose
    }
    
    // 计算身高
    float calculate_height(const SimplePoseResult& pose) {
        Point2f head_pixel = get_head_position(pose);
        Point2f foot_pixel = get_foot_position(pose);
        
        cout << "\n=== 身高计算过程 ===" << endl;
        cout << "头部像素: (" << head_pixel.x << ", " << head_pixel.y << ")" << endl;
        cout << "脚部像素: (" << foot_pixel.x << ", " << foot_pixel.y << ")" << endl;
        
        // 1. 脚部投影到地面，获取地面坐标
        Point2f foot_world = pixel_to_world(foot_pixel);
        cout << "脚部地面坐标: (" << foot_world.x << ", " << foot_world.y << ") mm" << endl;
        
        // 2. 头部投影到地面
        Point2f head_world = pixel_to_world(head_pixel);
        cout << "头部地面投影: (" << head_world.x << ", " << head_world.y << ") mm" << endl;
        
        // 3. 计算脚部到相机中心的距离
        float foot_distance_to_camera = sqrt(foot_world.x * foot_world.x + foot_world.y * foot_world.y);
        cout << "脚部到相机距离: " << foot_distance_to_camera << " mm" << endl;
        
        // 4. 计算像素高度
        float pixel_height = abs(head_pixel.y - foot_pixel.y);
        cout << "像素高度差: " << pixel_height << " pixels" << endl;
        
        // 5. 利用Homography尺度计算真实身高
        Point2f foot_plus_one = Point2f(foot_pixel.x, foot_pixel.y - 1); // 向上1像素
        Point2f foot_plus_one_world = pixel_to_world(foot_plus_one);
        
        float pixel_scale_at_foot = sqrt(pow(foot_plus_one_world.x - foot_world.x, 2) + 
                                        pow(foot_plus_one_world.y - foot_world.y, 2));
        cout << "脚部位置像素尺度: " << pixel_scale_at_foot << " mm/pixel" << endl;
        
        // 6. 估算身高（考虑透视效应）
        float estimated_height = pixel_height * pixel_scale_at_foot;
        
        // 7. 透视修正
        float perspective_correction = 1.0f + (foot_distance_to_camera / camera_height_mm_) * 0.1f;
        estimated_height *= perspective_correction;
        
        cout << "透视修正因子: " << perspective_correction << endl;
        cout << "最终估算身高: " << estimated_height << " mm = " << estimated_height/10.0f << " cm" << endl;
        
        return estimated_height;
    }
    
    // 绘制骨架连接线
    void draw_skeleton_lines(Mat& image, const SimplePoseResult& pose) {
        // COCO 17点骨架连接定义
        vector<pair<int, int>> skeleton_connections = {
            {5, 6},   // 左肩-右肩
            {5, 7},   // 左肩-左肘
            {7, 9},   // 左肘-左腕
            {6, 8},   // 右肩-右肘
            {8, 10},  // 右肘-右腕
            {5, 11},  // 左肩-左髋
            {6, 12},  // 右肩-右髋
            {11, 12}, // 左髋-右髋
            {11, 13}, // 左髋-左膝
            {13, 15}, // 左膝-左踝
            {12, 14}, // 右髋-右膝
            {14, 16}  // 右膝-右踝
        };
        
        for (const auto& connection : skeleton_connections) {
            int p1_idx = connection.first;
            int p2_idx = connection.second;
            
            if (p1_idx < pose.keypoint_scores.size() && 
                p2_idx < pose.keypoint_scores.size() &&
                pose.keypoint_scores[p1_idx] > 0.5f && 
                pose.keypoint_scores[p2_idx] > 0.5f) {
                
                line(image, pose.keypoints[p1_idx], pose.keypoints[p2_idx], 
                     Scalar(0, 255, 255), 2);  // 黄色骨架线
            }
        }
    }
    
    // 🔥 核心：处理图片并生成结果
    Mat process_image_with_simulated_pose(const Mat& image, const string& output_path = "") {
        cout << "\n=== 基于模拟Pose检测的身高测量 ===" << endl;
        cout << "图片尺寸: " << image.cols << "x" << image.rows << endl;
        
        // 1. 模拟姿态检测
        auto pose_results = simulate_pose_detection(image);
        
        if (pose_results.empty()) {
            cout << "  无姿态检测结果" << endl;
            return image.clone();
        }
        
        // 2. 创建结果图片
        Mat result_image = image.clone();
        
        for (size_t i = 0; i < pose_results.size(); ++i) {
            const auto& pose = pose_results[i];
            
            cout << "\n--- Person " << (i+1) << " ---" << endl;
            cout << "检测置信度: " << pose.confidence << endl;
            
            // 绘制ROI框（紫色）
            Scalar roi_color(255, 0, 255);  // 紫色
            rectangle(result_image, pose.bbox, roi_color, 3);
            
            // 绘制关键点
            for (size_t j = 0; j < pose.keypoints.size(); ++j) {
                if (pose.keypoint_scores[j] > 0.5f) {
                    circle(result_image, pose.keypoints[j], 4, Scalar(0, 255, 0), -1);
                }
            }
            
            // 绘制骨架连接线
            draw_skeleton_lines(result_image, pose);
            
            // 计算身高
            float height_mm = calculate_height(pose);
            float height_cm = height_mm / 10.0f;
            
            // 身高文本
            string height_text = cv::format("Height: %.1fcm", height_cm);
            Scalar text_color = Scalar(0, 255, 0);  // 绿色
            cout << "✓ 身高: " << height_cm << "cm" << endl;
            
            // ROI标签
            string roi_text = cv::format("Person%d (%.2f)", pose.person_id, pose.confidence);
            
            // 绘制文字背景和文字
            Point text_org(pose.bbox.x, pose.bbox.y - 10);
            
            // ROI标签
            Size roi_text_size = getTextSize(roi_text, FONT_HERSHEY_SIMPLEX, 0.7, 2, nullptr);
            rectangle(result_image, 
                     text_org + Point(-5, 5),
                     text_org + Point(roi_text_size.width + 5, -roi_text_size.height - 5),
                     Scalar(0, 0, 0), -1);
            putText(result_image, roi_text, text_org, 
                   FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2);
            
            // 身高标签
            Point height_text_org(pose.bbox.x, pose.bbox.y - 40);
            Size height_text_size = getTextSize(height_text, FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            rectangle(result_image, 
                     height_text_org + Point(-5, 5),
                     height_text_org + Point(height_text_size.width + 5, -height_text_size.height - 5),
                     Scalar(0, 0, 0), -1);
            putText(result_image, height_text, height_text_org, 
                   FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
            
            // 显示世界坐标
            if (pose.has_ground_position) {
                cout << "世界坐标: (" << pose.ground_position.x << ", " 
                     << pose.ground_position.y << ")mm" << endl;
                
                string world_coord_text = cv::format("World: (%.0f,%.0f)mm", 
                                                    pose.ground_position.x, 
                                                    pose.ground_position.y);
                Point coord_text_org(pose.bbox.x, pose.bbox.y + pose.bbox.height + 25);
                Size coord_text_size = getTextSize(world_coord_text, FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
                rectangle(result_image, 
                         coord_text_org + Point(-3, 3),
                         coord_text_org + Point(coord_text_size.width + 3, -coord_text_size.height - 3),
                         Scalar(0, 0, 0), -1);
                putText(result_image, world_coord_text, coord_text_org, 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
        }
        
        // 添加标题
        putText(result_image, "Simulated Pose Detection + Height Measurement", 
                Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        putText(result_image, "Camera Height: 117cm, Homography-based calculation", 
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
    cout << "=== 基于模拟Pose检测的身高测量测试 ===" << endl;
    cout << "注意：此程序使用模拟的姿态检测结果来演示ROI可视化和身高测量流程" << endl;
    
    try {
        SimpleHeightMeasurementTester tester;
        
        // 1. 加载Homography标定
        string homography_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/data/2025_8_6_1280_720.json";
        
        if (!tester.load_homography(homography_file)) {
            cerr << "❌ 无法加载Homography文件" << endl;
            return -1;
        }
        
        // 2. 测试多张图片
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
            
            // 处理图片并生成结果
            string output_filename = cv::format("simulated_pose_height_result_%d.jpg", (int)(i+1));
            tester.process_image_with_simulated_pose(image, output_filename);
        }
        
        cout << "\n" << string(60, '=') << endl;
        cout << "✅ 所有图片测试完成" << endl;
        cout << "🎉 成功演示了：模拟Pose检测 → ROI可视化 → 身高测量流程" << endl;
        cout << "📝 注意：这是模拟版本，实际应用中需要集成真实的YOLOv8 Pose检测器" << endl;
        
    } catch (const std::exception& e) {
        cerr << "错误: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}