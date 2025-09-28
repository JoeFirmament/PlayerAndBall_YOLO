/**
 * 简单身高测量演示程序 - 高中生也能看懂的版本！
 * 演示如何从照片像素坐标计算出真实世界中的身高
 */

#include <iostream>
#include <cmath>
#include <vector>

// 简化版本的Homography变换矩阵
// 这是从实际标定文件中提取的3x3矩阵
std::vector<std::vector<double>> homography_matrix = {
    {-3.2720398953723757, -0.006616969830473663, 2185.3722002814093},
    {-0.07920249932550606, 0.6201388621485532, -2183.270680916352},
    {2.0578777115434938e-05, -0.0027736686912052497, 1.0}
};

/**
 * 像素坐标转换为世界坐标
 * 输入：像素坐标 (x_pixel, y_pixel)
 * 输出：世界坐标 (x_world, y_world) 单位：毫米
 */
std::pair<double, double> pixel_to_world(double x_pixel, double y_pixel) {
    // 添加齐次坐标 (x, y, 1)
    double x_h = x_pixel;
    double y_h = y_pixel;
    double z_h = 1.0;

    // 矩阵乘法：world = homography_matrix * [x_pixel, y_pixel, 1]
    double x_world = homography_matrix[0][0] * x_h +
                     homography_matrix[0][1] * y_h +
                     homography_matrix[0][2] * z_h;

    double y_world = homography_matrix[1][0] * x_h +
                     homography_matrix[1][1] * y_h +
                     homography_matrix[1][2] * z_h;

    double z_world = homography_matrix[2][0] * x_h +
                     homography_matrix[2][1] * y_h +
                     homography_matrix[2][2] * z_h;

    // 归一化（除以z坐标）
    if (z_world != 0) {
        x_world /= z_world;
        y_world /= z_world;
    }

    return {x_world, y_world};
}

/**
 * 计算两个世界坐标点之间的距离
 */
double calculate_distance(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);  // 单位：毫米
}

/**
 * 演示身高测量过程
 */
void demonstrate_height_measurement() {
    std::cout << "=======================================" << std::endl;
    std::cout << "      高中生也能懂的身高测量演示" << std::endl;
    std::cout << "=======================================" << std::endl;

    // 场景：篮球场上的小明
    std::cout << "\n场景：篮球场上的小明" << std::endl;
    std::cout << "相机拍下一张照片，照片分辨率 1280x720 像素" << std::endl;

    // 1. 照片里的关键点坐标（像素单位）
    double nose_pixel_x = 400.0;    // 鼻子x坐标：400像素
    double nose_pixel_y = 300.0;    // 鼻子y坐标：300像素

    double ankle_pixel_x = 400.0;   // 脚踝x坐标：400像素
    double ankle_pixel_y = 450.0;   // 脚踝y坐标：450像素

    std::cout << "\n1️⃣ 照片里的测量（像素单位）" << std::endl;
    std::cout << "鼻子像素坐标：(" << nose_pixel_x << ", " << nose_pixel_y << ") 像素" << std::endl;
    std::cout << "脚踝像素坐标：(" << ankle_pixel_x << ", " << ankle_pixel_y << ") 像素" << std::endl;
    std::cout << "照片上垂直距离：" << (ankle_pixel_y - nose_pixel_y) << " 像素" << std::endl;

    // 2. 转换为世界坐标
    auto [nose_world_x, nose_world_y] = pixel_to_world(nose_pixel_x, nose_pixel_y);
    auto [ankle_world_x, ankle_world_y] = pixel_to_world(ankle_pixel_x, ankle_pixel_y);

    std::cout << "\n2️⃣ 转换为篮球场坐标（毫米单位）" << std::endl;
    std::cout << "鼻子世界坐标：(" << nose_world_x << ", " << nose_world_y << ") 毫米" << std::endl;
    std::cout << "脚踝世界坐标：(" << ankle_world_x << ", " << ankle_world_y << ") 毫米" << std::endl;

    // 3. 计算真实身高
    double height_mm = calculate_distance(nose_world_x, nose_world_y, ankle_world_x, ankle_world_y);

    std::cout << "\n3️⃣ 计算真实身高" << std::endl;
    std::cout << "计算公式：身高 = √((x2-x1)² + (y2-y1)²)" << std::endl;
    std::cout << "水平距离：" << (ankle_world_x - nose_world_x) << " 毫米" << std::endl;
    std::cout << "垂直距离：" << (ankle_world_y - nose_world_y) << " 毫米" << std::endl;
    std::cout << "身高 = √((" << ankle_world_x << " - " << nose_world_x << ")² + ("
              << ankle_world_y << " - " << nose_world_y << ")²)" << std::endl;
    std::cout << "身高 = " << height_mm << " 毫米 = " << (height_mm / 10.0) << " 厘米" << std::endl;

    // 4. 实际应用中的校正
    std::cout << "\n4️⃣ 实际应用中的校正" << std::endl;
    std::cout << "鼻子到头顶还有大约30像素的距离" << std::endl;
    std::cout << "脚踝到脚底还有大约20像素的距离" << std::endl;

    // 校正头部位置（向上偏移30像素）
    double head_pixel_y = nose_pixel_y - 30;
    auto [head_world_x, head_world_y] = pixel_to_world(nose_pixel_x, head_pixel_y);

    // 校正脚部位置（向下偏移20像素）
    double foot_pixel_y = ankle_pixel_y + 20;
    auto [foot_world_x, foot_world_y] = pixel_to_world(ankle_pixel_x, foot_pixel_y);

    double corrected_height = calculate_distance(head_world_x, head_world_y, foot_world_x, foot_world_y);

    std::cout << "校正后身高：" << corrected_height << " 毫米 = " << (corrected_height / 10.0) << " 厘米" << std::endl;

    // 5. 精度说明
    std::cout << "\n5️⃣ 精度说明" << std::endl;
    std::cout << "理论精度：±5-20毫米（取决于标定质量和距离）" << std::endl;
    std::cout << "实际精度：±10-30毫米（考虑各种干扰因素）" << std::endl;
    std::cout << "多帧平均后精度更高！" << std::endl;

    std::cout << "\n=======================================" << std::endl;
    std::cout << "这就是从照片量出身高的神奇过程！" << std::endl;
    std::cout << "计算机视觉就是这么神奇！🏀✨" << std::endl;
}

/**
 * 演示距离计算的数学原理
 */
void demonstrate_math_principle() {
    std::cout << "\n\n=======================================" << std::endl;
    std::cout << "      距离计算的数学原理" << std::endl;
    std::cout << "=======================================" << std::endl;

    // 两个点在二维平面上的距离公式
    std::cout << "\n📐 距离公式：" << std::endl;
    std::cout << "如果有两个点：" << std::endl;
    std::cout << "点A：(x1, y1)" << std::endl;
    std::cout << "点B：(x2, y2)" << std::endl;
    std::cout << std::endl;
    std::cout << "距离 = √[(x2 - x1)² + (y2 - y1)²]" << std::endl;

    // 举例计算
    std::cout << "\n📝 举例计算：" << std::endl;
    std::cout << "点A：(0, 0)" << std::endl;
    std::cout << "点B：(3, 4)" << std::endl;
    std::cout << std::endl;
    std::cout << "距离 = √[(3-0)² + (4-0)²] = √[9 + 16] = √25 = 5" << std::endl;

    // 身高测量中的应用
    std::cout << "\n🏀 在身高测量中的应用：" << std::endl;
    std::cout << "头顶：(-2000, 3000) 毫米" << std::endl;
    std::cout << "脚底：(-2000, 4850) 毫米" << std::endl;
    std::cout << std::endl;
    std::cout << "水平距离：-2000 - (-2000) = 0 毫米" << std::endl;
    std::cout << "垂直距离：4850 - 3000 = 1850 毫米" << std::endl;
    std::cout << "身高 = √(0² + 1850²) = 1850 毫米" << std::endl;
    std::cout << "身高 = 185.0 厘米" << std::endl;

    std::cout << "\n这就是勾股定理在计算机视觉中的应用！" << std::endl;
    std::cout << "初中数学就能理解的高科技！😄" << std::endl;
}

int main() {
    demonstrate_height_measurement();
    demonstrate_math_principle();

    return 0;
}

