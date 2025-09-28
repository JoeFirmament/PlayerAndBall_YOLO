#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

class HeightMeasurementTester {
private:
    Mat homography_matrix_;
    float camera_height_mm_;
    
public:
    HeightMeasurementTester() : camera_height_mm_(1170.0f) {} // 117cm = 1170mm
    
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
    
    // 🔥 核心：基于Homography和相机高度计算身高
    float calculate_height(const Point2f& head_pixel, const Point2f& foot_pixel) {
        cout << "\n=== 身高计算过程 ===" << endl;
        cout << "头部像素: (" << head_pixel.x << ", " << head_pixel.y << ")" << endl;
        cout << "脚部像素: (" << foot_pixel.x << ", " << foot_pixel.y << ")" << endl;
        
        // 1. 脚部投影到地面，获取地面坐标
        Point2f foot_world = pixel_to_world(foot_pixel);
        cout << "脚部地面坐标: (" << foot_world.x << ", " << foot_world.y << ") mm" << endl;
        
        // 2. 头部投影到地面
        Point2f head_world = pixel_to_world(head_pixel);
        cout << "头部地面投影: (" << head_world.x << ", " << head_world.y << ") mm" << endl;
        
        // 3. 计算地面距离差异（这代表身高在地面的投影）
        float ground_distance_diff = sqrt(pow(head_world.x - foot_world.x, 2) + 
                                         pow(head_world.y - foot_world.y, 2));
        cout << "地面投影差异: " << ground_distance_diff << " mm" << endl;
        
        // 4. 计算脚部到相机中心的距离
        float foot_distance_to_camera = sqrt(foot_world.x * foot_world.x + foot_world.y * foot_world.y);
        cout << "脚部到相机距离: " << foot_distance_to_camera << " mm" << endl;
        
        // 5. 使用相似三角形原理计算身高
        // 原理：相机高度/脚部距离 = 身高/像素高度对应的地面距离
        float pixel_height = abs(head_pixel.y - foot_pixel.y);
        cout << "像素高度差: " << pixel_height << " pixels" << endl;
        
        // 6. 利用Homography尺度计算真实身高
        // 方法：在脚部位置计算单位像素对应的真实距离
        Point2f foot_plus_one = Point2f(foot_pixel.x, foot_pixel.y - 1); // 向上1像素
        Point2f foot_plus_one_world = pixel_to_world(foot_plus_one);
        
        float pixel_scale_at_foot = sqrt(pow(foot_plus_one_world.x - foot_world.x, 2) + 
                                        pow(foot_plus_one_world.y - foot_world.y, 2));
        cout << "脚部位置像素尺度: " << pixel_scale_at_foot << " mm/pixel" << endl;
        
        // 7. 估算身高（考虑透视效应）
        float estimated_height = pixel_height * pixel_scale_at_foot;
        
        // 8. 透视修正：头部距离相机更近，需要修正
        float perspective_correction = 1.0f + (foot_distance_to_camera / camera_height_mm_) * 0.1f;
        estimated_height *= perspective_correction;
        
        cout << "透视修正因子: " << perspective_correction << endl;
        cout << "最终估算身高: " << estimated_height << " mm = " << estimated_height/10.0f << " cm" << endl;
        
        return estimated_height;
    }
};

// GUI代码已移除，使用预设测试点

int main() {
    HeightMeasurementTester tester;
    
    // 1. 加载Homography矩阵
    string homography_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/data/2025_8_6_1280_720.json";
    if (!tester.load_homography(homography_file)) {
        return -1;
    }
    
    // 2. 加载测试图片
    string image_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/imgs/pose.jpg";
    Mat image = imread(image_file);
    if (image.empty()) {
        cerr << "无法加载图片: " << image_file << endl;
        return -1;
    }
    
    cout << "\n图片加载成功！尺寸: " << image.cols << "x" << image.rows << endl;
    
    // 3. 使用预设测试点（可以根据实际图片调整）
    vector<pair<Point2f, Point2f>> test_poses = {
        {Point2f(400, 150), Point2f(400, 650)},  // 测试人员1：头部(400,150), 脚部(400,650)
        {Point2f(600, 120), Point2f(600, 680)},  // 测试人员2：头部(600,120), 脚部(600,680)
        {Point2f(800, 100), Point2f(800, 700)},  // 测试人员3：头部(800,100), 脚部(800,700)
    };
    
    Mat result_image = image.clone();
    
    // 4. 对每个测试点计算身高
    for (int i = 0; i < test_poses.size(); i++) {
        Point2f head = test_poses[i].first;
        Point2f foot = test_poses[i].second;
        
        float height_mm = tester.calculate_height(head, foot);
        float height_cm = height_mm / 10.0f;
        
        // 在图片上绘制测试点和结果
        string person_name = "Person" + to_string(i + 1);
        
        // 绘制头部（绿色圆圈）
        circle(result_image, head, 8, Scalar(0, 255, 0), 2);
        putText(result_image, "HEAD", Point(head.x + 15, head.y - 5), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        
        // 绘制脚部（红色圆圈）
        circle(result_image, foot, 8, Scalar(0, 0, 255), 2);
        putText(result_image, "FOOT", Point(foot.x + 15, foot.y + 20), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        
        // 绘制身高线（蓝色线条）
        line(result_image, head, foot, Scalar(255, 0, 0), 3);
        
        // 标注身高结果（黄色文字）
        string height_text = person_name + ": " + to_string((int)height_cm) + " cm";
        putText(result_image, height_text, 
                Point(head.x + 20, head.y - 20), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
        
        // 控制台输出
        cout << "\n================================" << endl;
        cout << "🎯 " << person_name << " 身高测量结果: " << height_cm << " cm" << endl;
        cout << "   头部坐标: (" << head.x << ", " << head.y << ")" << endl;
        cout << "   脚部坐标: (" << foot.x << ", " << foot.y << ")" << endl;
        cout << "================================" << endl;
    }
    
    // 5. 添加标题和说明
    putText(result_image, "Height Measurement Test Results", 
            Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 3);
    putText(result_image, "Camera Height: 117cm, Homography-based calculation", 
            Point(50, 90), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2);
    
    // 6. 保存结果图片
    string result_file = "/home/orangepi/Qworkspace/yolov8_pose_basketball/pose_analysis/height_result.jpg";
    bool saved = imwrite(result_file, result_image);
    
    if (saved) {
        cout << "\n✅ 结果图片已保存到: " << result_file << endl;
    } else {
        cout << "\n❌ 保存结果图片失败！" << endl;
    }
    
    cout << "\n🎉 身高测量测试完成！" << endl;
    
    return 0;
}