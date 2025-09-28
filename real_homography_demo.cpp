/**
 * 真实Homography系统演示
 * 基于实际项目中的标定数据展示身高测量
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

// 使用实际项目中的Homography矩阵
std::vector<std::vector<double>> homography_matrix = {
    {-3.2720398953723757, -0.006616969830473663, 2185.3722002814093},
    {-0.07920249932550606, 0.6201388621485532, -2183.270680916352},
    {2.0578777115434938e-05, -0.0027736686912052497, 1.0}
};

/**
 * 像素坐标转换为世界坐标
 * 使用实际项目中的Homography矩阵
 */
cv::Point2f pixel_to_world(cv::Point2f pixel_point) {
    // 转换为齐次坐标
    cv::Mat src_points = (cv::Mat_<float>(3, 1) << pixel_point.x, pixel_point.y, 1.0f);

    // Homography矩阵
    cv::Mat H = (cv::Mat_<float>(3, 3) <<
        homography_matrix[0][0], homography_matrix[0][1], homography_matrix[0][2],
        homography_matrix[1][0], homography_matrix[1][1], homography_matrix[1][2],
        homography_matrix[2][0], homography_matrix[2][1], homography_matrix[2][2]);

    // 矩阵乘法
    cv::Mat dst_points = H * src_points;

    // 归一化
    float x_world = dst_points.at<float>(0, 0) / dst_points.at<float>(2, 0);
    float y_world = dst_points.at<float>(1, 0) / dst_points.at<float>(2, 0);

    return cv::Point2f(x_world, y_world);
}

/**
 * 计算两个世界坐标点之间的距离
 */
float calculate_world_distance(cv::Point2f p1, cv::Point2f p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * 演示真实Homography系统中的身高测量
 */
void demonstrate_real_homography_height_measurement() {
    std::cout << "=======================================" << std::endl;
    std::cout << "      真实Homography身高测量演示" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n🏀 场景：篮球场上使用真实的标定系统" << std::endl;
    std::cout << "📊 使用项目中的实际Homography矩阵" << std::endl;
    std::cout << "🎯 测量运动员身高" << std::endl;

    // 从实际标定数据中选择有代表性的点
    // 标定点6: (713.32, 484.66) -> (455, 5915)
    // 标定点7: (857.48, 613.90) -> (910, 2730)

    std::cout << "\n📐 首先验证地面距离测量：" << std::endl;

    // 标定点6: 713.32, 484.66 -> 455, 5915
    cv::Point2f cal_point6_pixel(713.32f, 484.66f);
    cv::Point2f cal_point6_world = pixel_to_world(cal_point6_pixel);

    // 标定点7: 857.48, 613.90 -> 910, 2730
    cv::Point2f cal_point7_pixel(857.48f, 613.90f);
    cv::Point2f cal_point7_world = pixel_to_world(cal_point7_pixel);

    float ground_distance = calculate_world_distance(cal_point6_world, cal_point7_world);

    std::cout << "标定点6（像素）：" << cal_point6_pixel << std::endl;
    std::cout << "标定点6（世界）：" << cal_point6_world << " mm" << std::endl;
    std::cout << "标定点7（像素）：" << cal_point7_pixel << std::endl;
    std::cout << "标定点7（世界）：" << cal_point7_world << " mm" << std::endl;
    std::cout << "地面距离：" << ground_distance << " 毫米 = " << ground_distance/1000 << " 米" << std::endl;

    // 理论上的距离（从标定文件中提取）
    float theoretical_distance = std::sqrt(std::pow(910-455, 2) + std::pow(2730-5915, 2));
    std::cout << "理论距离：" << theoretical_distance << " 毫米" << std::endl;
    std::cout << "误差：" << std::abs(ground_distance - theoretical_distance) << " 毫米" << std::endl;

    // 现在测量运动员身高
    std::cout << "\n🎯 测量运动员身高：" << std::endl;

    // 假设运动员的头顶和脚踝在图像中的位置
    cv::Point2f athlete_head_pixel(500.0f, 350.0f);   // 头顶像素坐标
    cv::Point2f athlete_foot_pixel(500.0f, 480.0f);   // 脚踝像素坐标

    // 转换为世界坐标
    cv::Point2f athlete_head_world = pixel_to_world(athlete_head_pixel);
    cv::Point2f athlete_foot_world = pixel_to_world(athlete_foot_pixel);

    // 计算身高
    float athlete_height = calculate_world_distance(athlete_head_world, athlete_foot_world);

    std::cout << "运动员头顶（像素）：" << athlete_head_pixel << std::endl;
    std::cout << "运动员头顶（世界）：" << athlete_head_world << " mm" << std::endl;
    std::cout << "运动员脚踝（像素）：" << athlete_foot_pixel << std::endl;
    std::cout << "运动员脚踝（世界）：" << athlete_foot_world << " mm" << std::endl;
    std::cout << std::endl;
    std::cout << "📏 计算身高：" << std::endl;
    std::cout << "身高 = √((" << athlete_head_world.x << " - " << athlete_foot_world.x << ")² + ("
              << athlete_head_world.y << " - " << athlete_foot_world.y << ")²)" << std::endl;
    std::cout << "身高 = √((" << (athlete_head_world.x - athlete_foot_world.x) << ")² + ("
              << (athlete_head_world.y - athlete_foot_world.y) << ")²)" << std::endl;
    std::cout << "身高 = √(" << std::pow(athlete_head_world.x - athlete_foot_world.x, 2) << " + "
              << std::pow(athlete_head_world.y - athlete_foot_world.y, 2) << ")" << std::endl;
    std::cout << "身高 = √(" << (std::pow(athlete_head_world.x - athlete_foot_world.x, 2) +
                                std::pow(athlete_head_world.y - athlete_foot_world.y, 2)) << ")" << std::endl;
    std::cout << "身高 = " << athlete_height << " 毫米" << std::endl;
    std::cout << "身高 = " << athlete_height / 10.0f << " 厘米" << std::endl;
    std::cout << "身高 = " << athlete_height / 1000.0f << " 米" << std::endl;

    // 位置分析
    std::cout << "\n📍 运动员位置分析：" << std::endl;
    std::cout << "运动员站在篮球场上的位置：" << athlete_foot_world << " mm" << std::endl;
    std::cout << "距离篮筐的距离：" << calculate_world_distance(athlete_foot_world, cv::Point2f(0, 0))
              << " 毫米" << std::endl;

    // 验证Homography矩阵的准确性
    std::cout << "\n🎯 Homography矩阵验证：" << std::endl;

    // 测试几个标定点
    std::vector<std::pair<cv::Point2f, cv::Point2f>> test_points = {
        {cv::Point2f(713.32f, 484.66f), cv::Point2f(455.0f, 5915.0f)},  // 标定点6
        {cv::Point2f(857.48f, 613.90f), cv::Point2f(910.0f, 2730.0f)}   // 标定点7
    };

    for(size_t i = 0; i < test_points.size(); i++) {
        cv::Point2f pixel = test_points[i].first;
        cv::Point2f expected_world = test_points[i].second;
        cv::Point2f calculated_world = pixel_to_world(pixel);

        float error_x = std::abs(calculated_world.x - expected_world.x);
        float error_y = std::abs(calculated_world.y - expected_world.y);
        float error_distance = std::sqrt(error_x * error_x + error_y * error_y);

        std::cout << "标定点" << (i+6) << "：" << std::endl;
        std::cout << "  像素坐标：" << pixel << std::endl;
        std::cout << "  期望世界坐标：" << expected_world << " mm" << std::endl;
        std::cout << "  计算世界坐标：" << calculated_world << " mm" << std::endl;
        std::cout << "  误差：x=" << error_x << "mm, y=" << error_y << "mm, 距离=" << error_distance << "mm" << std::endl;
    }

    std::cout << "\n💡 关键结论：" << std::endl;
    std::cout << "1. Homography矩阵将2D像素坐标转换为3D世界坐标" << std::endl;
    std::cout << "2. 在世界坐标系中，可以直接计算距离（使用勾股定理）" << std::endl;
    std::cout << "3. 垂直方向的高度测量就是两个世界坐标点之间的距离" << std::endl;
    std::cout << "4. 篮球场上的Homography系统精度通常在±20毫米以内" << std::endl;

    std::cout << "\n🎉 这就是垂直方向如何转换为毫米的完整过程！" << std::endl;
    std::cout << "Homography + 勾股定理 = 从2D图像测量3D世界尺寸的神奇能力！🏀✨" << std::endl;
}

/**
 * 简单数学解释
 */
void mathematical_explanation() {
    std::cout << "\n\n=======================================" << std::endl;
    std::cout << "      数学原理：为什么能测量垂直方向？" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n📐 核心数学原理：" << std::endl;
    std::cout << "1. Homography矩阵：2D像素 → 3D世界坐标" << std::endl;
    std::cout << "2. 勾股定理：在3D世界中计算距离" << std::endl;
    std::cout << "3. 篮球场假设：所有人站在同一平面上" << std::endl;

    std::cout << "\n🔢 具体计算：" << std::endl;
    std::cout << "像素坐标：P_pixel = (x_pixel, y_pixel)" << std::endl;
    std::cout << "Homography：P_world = H × P_pixel" << std::endl;
    std::cout << "距离公式：distance = √((x2-x1)² + (y2-y1)²)" << std::endl;
    std::cout << "单位：毫米（或厘米、米）" << std::endl;

    std::cout << "\n🎯 关键洞察：" << std::endl;
    std::cout << "垂直方向的高度在Homography变换后，变成了世界坐标系中的距离差。" << std::endl;
    std::cout << "这就是为什么可以用2D图像测量3D世界中垂直方向尺寸的奥秘！" << std::endl;

    std::cout << "\n🏀 篮球场的特殊优势：" << std::endl;
    std::cout << "- 运动员站立时基本在同一高度" << std::endl;
    std::cout << "- 地面相对平整" << std::endl;
    std::cout << "- 相机角度固定" << std::endl;
    std::cout << "- 垂直方向压缩比例统一" << std::endl;
    std::cout << "- 完美适用于身高测量！" << std::endl;
}

int main() {
    demonstrate_real_homography_height_measurement();
    mathematical_explanation();

    return 0;
}

