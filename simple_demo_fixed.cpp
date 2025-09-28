/**
 * 简单身高测量演示程序 - 高中生也能看懂的版本！
 * 演示如何从照片像素坐标计算出真实世界中的身高
 * 修复版本：避免C++17特性，使用传统语法
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <utility>

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

    return std::make_pair(x_world, y_world);
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
 * 演示身高测量过程 - 一步一步讲解
 */
void demonstrate_height_measurement_step_by_step() {
    std::cout << "=======================================" << std::endl;
    std::cout << "      高中生也能懂的身高测量演示" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n想象这个场景：" << std::endl;
    std::cout << "你在篮球场上，用手机拍了一张同学小明的照片" << std::endl;
    std::cout << "照片大小是 1280×720 像素" << std::endl;
    std::cout << "现在，你想知道小明到底有多高..." << std::endl;

    // ===============================
    // 第一步：照片里的测量
    // ===============================
    std::cout << "\n\n📱 第一步：照片里的测量（像素世界）" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 假设AI检测到的关键点位置
    double nose_pixel_x = 500.0;    // 鼻子x坐标：500像素
    double nose_pixel_y = 300.0;    // 鼻子y坐标：300像素

    double ankle_pixel_x = 500.0;   // 脚踝x坐标：500像素
    double ankle_pixel_y = 450.0;   // 脚踝y坐标：450像素

    std::cout << "AI检测到小明的关键点：" << std::endl;
    std::cout << "  🥺 鼻子：在照片的 (" << nose_pixel_x << ", " << nose_pixel_y << ") 像素位置" << std::endl;
    std::cout << "  🦵 脚踝：在照片的 (" << ankle_pixel_x << ", " << ankle_pixel_y << ") 像素位置" << std::endl;

    double pixel_distance = ankle_pixel_y - nose_pixel_y;
    std::cout << "  📏 在照片上，从鼻子到脚踝的垂直距离：" << pixel_distance << " 像素" << std::endl;

    std::cout << "\n❓ 问题：这 150 像素到底等于多少厘米？" << std::endl;
    std::cout << "❌ 直接说 '150厘米' 是错的！照片里的距离不是真实距离！" << std::endl;

    // ===============================
    // 第二步：坐标系转换
    // ===============================
    std::cout << "\n\n🔄 第二步：把照片坐标转换为篮球场坐标" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "我们需要一个'翻译官'：Homography矩阵" << std::endl;
    std::cout << "它知道怎么把照片里的像素坐标翻译成篮球场上的毫米坐标" << std::endl;

    // 转换鼻子坐标
    std::pair<double, double> nose_world = pixel_to_world(nose_pixel_x, nose_pixel_y);
    double nose_world_x = nose_world.first;
    double nose_world_y = nose_world.second;

    // 转换脚踝坐标
    std::pair<double, double> ankle_world = pixel_to_world(ankle_pixel_x, ankle_pixel_y);
    double ankle_world_x = ankle_world.first;
    double ankle_world_y = ankle_world.second;

    std::cout << "\n📐 转换结果：" << std::endl;
    std::cout << "  🥺 鼻子在篮球场上的位置：(" << nose_world_x << ", " << nose_world_y << ") 毫米" << std::endl;
    std::cout << "  🦵 脚踝在篮球场上的位置：(" << ankle_world_x << ", " << ankle_world_y << ") 毫米" << std::endl;

    std::cout << "\n💡 理解坐标的含义：" << std::endl;
    std::cout << "  - 篮筐在 (0, 0) 位置" << std::endl;
    std::cout << "  - x坐标：正数在篮筐右边，负数在篮筐左边" << std::endl;
    std::cout << "  - y坐标：正数在篮筐前方，负数在篮筐后方" << std::endl;

    // ===============================
    // 第三步：计算真实距离
    // ===============================
    std::cout << "\n\n📏 第三步：在篮球场上计算真实距离" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    double distance_x = ankle_world_x - nose_world_x;
    double distance_y = ankle_world_y - nose_world_y;
    double height_mm = calculate_distance(nose_world_x, nose_world_y, ankle_world_x, ankle_world_y);

    std::cout << "📊 计算过程：" << std::endl;
    std::cout << "  水平距离（左右）：" << distance_x << " 毫米" << std::endl;
    std::cout << "  垂直距离（前后）：" << distance_y << " 毫米" << std::endl;
    std::cout << "  📐 使用勾股定理：" << std::endl;
    std::cout << "     身高 = √(水平距离² + 垂直距离²)" << std::endl;
    std::cout << "     身高 = √(" << distance_x << "² + " << distance_y << "²)" << std::endl;
    std::cout << "     身高 = √(" << (distance_x * distance_x) << " + " << (distance_y * distance_y) << ")" << std::endl;
    std::cout << "     身高 = √(" << ((distance_x * distance_x) + (distance_y * distance_y)) << ")" << std::endl;
    std::cout << "     身高 = " << height_mm << " 毫米" << std::endl;
    std::cout << "     身高 = " << (height_mm / 10.0) << " 厘米" << std::endl;
    std::cout << "     身高 = " << (height_mm / 1000.0) << " 米" << std::endl;

    // ===============================
    // 第四步：实际应用中的优化
    // ===============================
    std::cout << "\n\n🎯 第四步：实际应用中的优化" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "📏 为什么不直接用鼻子到脚踝的距离？" << std::endl;
    std::cout << "   因为鼻子到头顶还有一段距离！" << std::endl;
    std::cout << "   脚踝到脚底也有距离！" << std::endl;

    // 校正头部位置（向上偏移30像素）
    double head_pixel_y = nose_pixel_y - 30;
    std::pair<double, double> head_world = pixel_to_world(nose_pixel_x, head_pixel_y);
    double head_world_x = head_world.first;
    double head_world_y = head_world.second;

    // 校正脚部位置（向下偏移20像素）
    double foot_pixel_y = ankle_pixel_y + 20;
    std::pair<double, double> foot_world = pixel_to_world(ankle_pixel_x, foot_pixel_y);
    double foot_world_x = foot_world.first;
    double foot_world_y = foot_world.second;

    double corrected_height = calculate_distance(head_world_x, head_world_y, foot_world_x, foot_world_y);

    std::cout << "\n🔧 校正过程：" << std::endl;
    std::cout << "  1. 头部校正：鼻子位置向上30像素（约头发高度）" << std::endl;
    std::cout << "  2. 脚部校正：脚踝位置向下20像素（约鞋底高度）" << std::endl;
    std::cout << "  3. 重新计算距离" << std::endl;

    std::cout << "\n✅ 校正后身高：" << corrected_height << " 毫米 = " << (corrected_height / 10.0) << " 厘米" << std::endl;

    // ===============================
    // 第五步：精度和稳定性
    // ===============================
    std::cout << "\n\n🎯 第五步：提高测量精度" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "📊 为什么需要多次测量？" << std::endl;
    std::cout << "   因为人会动，姿势会变，光线会影响检测" << std::endl;

    // 模拟多次测量
    std::vector<double> measurements;
    measurements.push_back(corrected_height);
    measurements.push_back(corrected_height + 15);  // 模拟轻微变化
    measurements.push_back(corrected_height - 10);  // 模拟轻微变化
    measurements.push_back(corrected_height + 5);   // 模拟轻微变化
    measurements.push_back(corrected_height - 8);   // 模拟轻微变化

    double sum = 0;
    for(double m : measurements) {
        sum += m;
    }
    double average = sum / measurements.size();

    std::cout << "\n📈 五次测量结果：" << std::endl;
    for(size_t i = 0; i < measurements.size(); i++) {
        std::cout << "   第" << (i+1) << "次： " << measurements[i] << " 毫米" << std::endl;
    }
    std::cout << "   平均值： " << average << " 毫米 = " << (average / 10.0) << " 厘米" << std::endl;
    std::cout << "   精度范围：±" << 15 << "毫米（取决于测量条件）" << std::endl;

    std::cout << "\n🎉 结论：" << std::endl;
    std::cout << "   我们从一张照片，量出了真实世界中的身高！" << std::endl;
    std::cout << "   这就是计算机视觉的魔力！" << std::endl;
    std::cout << "   初中数学 + 高科技 = 无限可能！🏀✨" << std::endl;
}

/**
 * 超简单版本：用直观的例子解释
 */
void super_simple_explanation() {
    std::cout << "\n\n=======================================" << std::endl;
    std::cout << "      超简单版本：用生活例子解释" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n想象你在操场上：" << std::endl;
    std::cout << "1. 📱 你用手机拍了一张照片" << std::endl;
    std::cout << "2. 🤖 AI找出同学的鼻子和脚踝在照片里的位置" << std::endl;
    std::cout << "3. 🔄 '翻译官'把照片里的位置转换为操场上的真实位置" << std::endl;
    std::cout << "4. 📏 在操场上测量鼻子到脚踝的直线距离" << std::endl;
    std::cout << "5. 🎯 就是同学的身高了！" << std::endl;

    std::cout << "\n关键在于第3步的'翻译'！" << std::endl;
    std::cout << "照片里的距离是假的，翻译成真实世界的距离才是真的！" << std::endl;

    std::cout << "\n这就是为什么：" << std::endl;
    std::cout << "- 照片里看起来很近的两个人，实际上可能相隔很远" << std::endl;
    std::cout << "- 照片里看起来很高的人，实际上可能很矮" << std::endl;
    std::cout << "- 需要数学公式来'翻译'这些视觉错觉" << std::endl;

    std::cout << "\n所以，下次当你看到AI量出身高时，记住：" << std::endl;
    std::cout << "这不是魔法，而是数学和计算机的完美结合！🎉" << std::endl;
}

int main() {
    demonstrate_height_measurement_step_by_step();
    super_simple_explanation();

    return 0;
}

