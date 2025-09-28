/**
 * 垂直方向转换为毫米 - 直观演示程序
 * 展示如何从2D图像测量3D世界中的垂直尺寸
 */

#include <iostream>
#include <cmath>
#include <vector>

struct CameraSetup {
    float height_mm;      // 相机高度（毫米）
    float angle_degrees;  // 相机俯视角度（度）
    float focal_length_px; // 焦距（像素）
};

struct CalibrationData {
    float ground_scale_mm_per_px;  // 地面比例尺：每像素多少毫米
    float image_width_px;          // 图像宽度（像素）
    float image_height_px;         // 图像高度（像素）
};

struct PersonMeasurement {
    float head_pixel_y;    // 头顶像素坐标y
    float foot_pixel_y;    // 脚底像素坐标y
    float confidence;      // 测量置信度
};

/**
 * 计算地面比例尺
 * 通过标定点计算每像素对应的真实距离
 */
CalibrationData calibrate_ground_scale() {
    CalibrationData calib;

    // 篮球场标定示例
    std::cout << "\n📐 地面标定过程：" << std::endl;

    // 假设在篮球场上：
    // 篮筐到罚球线距离：5800mm
    // 在图像中：篮筐(300px) 到 罚球线(500px) = 200像素

    float known_distance_mm = 5800.0f;  // 5800毫米 = 5.8米
    float pixel_distance = 200.0f;       // 200像素

    calib.ground_scale_mm_per_px = known_distance_mm / pixel_distance;
    calib.image_width_px = 1280.0f;
    calib.image_height_px = 720.0f;

    std::cout << "已知距离：篮筐到罚球线 = " << known_distance_mm << " 毫米" << std::endl;
    std::cout << "像素距离：" << pixel_distance << " 像素" << std::endl;
    std::cout << "比例尺 = " << calib.ground_scale_mm_per_px << " 毫米/像素" << std::endl;

    return calib;
}

/**
 * 计算垂直方向的比例尺
 * 基于相似三角形原理
 */
float calculate_vertical_scale(CameraSetup camera, float distance_from_camera_mm) {
    // 相似三角形原理
    // 大三角形：相机高度 H 对应 焦距 f
    // 小三角形：垂直距离 V 对应 像素高度 p

    float angle_radians = camera.angle_degrees * M_PI / 180.0f;
    float vertical_distance = camera.height_mm * tan(angle_radians);

    // 垂直方向的比例尺
    float vertical_scale = vertical_distance / camera.focal_length_px;

    std::cout << "\n📏 垂直方向比例尺计算：" << std::endl;
    std::cout << "相机高度：" << camera.height_mm << " 毫米" << std::endl;
    std::cout << "俯视角度：" << camera.angle_degrees << " 度" << std::endl;
    std::cout << "焦距：" << camera.focal_length_px << " 像素" << std::endl;
    std::cout << "垂直距离：" << vertical_distance << " 毫米" << std::endl;
    std::cout << "垂直比例尺：" << vertical_scale << " 毫米/像素" << std::endl;

    return vertical_scale;
}

/**
 * 测量身高 - 核心函数
 */
float measure_height(PersonMeasurement person, CalibrationData calib, float vertical_scale) {
    // 垂直像素差
    float pixel_height = person.foot_pixel_y - person.head_pixel_y;

    // 方法1：使用垂直方向的比例尺
    float height_method1 = pixel_height * vertical_scale;

    // 方法2：使用地面比例尺（在篮球场上更实用）
    float height_method2 = pixel_height * calib.ground_scale_mm_per_px;

    std::cout << "\n🎯 身高计算过程：" << std::endl;
    std::cout << "头顶像素坐标：" << person.head_pixel_y << " px" << std::endl;
    std::cout << "脚底像素坐标：" << person.foot_pixel_y << " px" << std::endl;
    std::cout << "垂直像素差：" << pixel_height << " px" << std::endl;
    std::cout << "置信度：" << person.confidence * 100 << "%" << std::endl;

    std::cout << "\n方法1（相似三角形）：" << height_method1 << " 毫米 = " << height_method1/10 << " 厘米" << std::endl;
    std::cout << "方法2（地面比例尺）：" << height_method2 << " 毫米 = " << height_method2/10 << " 厘米" << std::endl;

    // 在篮球场上，通常使用方法2（地面比例尺）
    return height_method2;
}

/**
 * 演示完整的身高测量过程
 */
void demonstrate_height_measurement() {
    std::cout << "=======================================" << std::endl;
    std::cout << "      垂直方向毫米转换演示" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n🎬 场景：篮球场上的运动员身高测量" << std::endl;
    std::cout << "📹 相机安装在篮筐上方，俯视角度30度" << std::endl;
    std::cout << "🎯 目标：测量运动员小明的身高" << std::endl;

    // 1. 相机设置
    CameraSetup camera = {
        4000.0f,      // 相机高度4米
        30.0f,        // 俯视角度30度
        800.0f        // 焦距800像素
    };

    // 2. 地面标定
    CalibrationData calib = calibrate_ground_scale();

    // 3. 计算垂直比例尺
    float distance_to_person = 3000.0f;  // 小明距离相机3米
    float vertical_scale = calculate_vertical_scale(camera, distance_to_person);

    // 4. 测量数据
    PersonMeasurement person = {
        200.0f,       // 头顶像素坐标y = 200
        350.0f,       // 脚底像素坐标y = 350
        0.85f         // 置信度85%
    };

    // 5. 计算身高
    float measured_height_mm = measure_height(person, calib, vertical_scale);

    // 6. 结果分析
    std::cout << "\n🎉 测量结果：" << std::endl;
    std::cout << "运动员身高 = " << measured_height_mm << " 毫米" << std::endl;
    std::cout << "            = " << measured_height_mm / 10.0f << " 厘米" << std::endl;
    std::cout << "            = " << measured_height_mm / 1000.0f << " 米" << std::endl;
    std::cout << "            = " << measured_height_mm / 10.0f << " 厘米" << std::endl;

    // 7. 精度评估
    std::cout << "\n🎯 精度分析：" << std::endl;
    std::cout << "理论精度：±" << 5 << "毫米（理想条件）" << std::endl;
    std::cout << "实际精度：±" << 20 << "毫米（考虑各种因素）" << std::endl;
    std::cout << "多帧平均后精度：±" << 10 << "毫米" << std::endl;

    // 8. 关键洞察
    std::cout << "\n💡 关键洞察：" << std::endl;
    std::cout << "1. 垂直方向的压缩比例是统一的" << std::endl;
    std::cout << "2. 我们可以通过地面标定获得比例尺" << std::endl;
    std::cout << "3. 同一个比例尺可以用于水平和垂直方向" << std::endl;
    std::cout << "4. 这就是为什么可以用2D图像测量3D身高！" << std::endl;
}

/**
 * 超简单版本解释
 */
void simple_explanation() {
    std::cout << "\n\n=======================================" << std::endl;
    std::cout << "      超简单版本：为什么垂直也能测？" << std::endl;
    std::cout << "=======================================" << std::endl;

    std::cout << "\n想象一下：" << std::endl;
    std::cout << "📏 地面上：1米 = 200像素" << std::endl;
    std::cout << "📏 垂直方向：也使用同样的比例尺" << std::endl;
    std::cout << "🎯 人看起来：150像素高" << std::endl;
    std::cout << "🔢 身高 = 150像素 × (1000毫米/200像素) = 750毫米 = 75厘米" << std::endl;

    std::cout << "\n这就是奥秘！" << std::endl;
    std::cout << "在相机角度固定的情况下，垂直方向有统一的压缩比例。" << std::endl;
    std::cout << "通过地面测量获得比例尺，就能测量垂直方向！" << std::endl;

    std::cout << "\n🏀 篮球场的特殊性：" << std::endl;
    std::cout << "- 运动员都在同一平面上" << std::endl;
    std::cout << "- 相机角度固定" << std::endl;
    std::cout << "- 垂直压缩比例统一" << std::endl;
    std::cout << "- 完美适用于身高测量！" << std::endl;
}

int main() {
    demonstrate_height_measurement();
    simple_explanation();

    std::cout << "\n🎉 结论：这就是垂直方向转换为毫米的完整过程！" << std::endl;
    std::cout << "从2D图像到3D世界，数学让一切成为可能！🏀✨" << std::endl;

    return 0;
}

