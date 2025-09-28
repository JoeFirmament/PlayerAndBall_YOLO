#include "cpp_calibration_loader.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>

/**
 * C++相机标定结果使用示例
 * 展示如何：
 * 1. 加载相机标定和地面标定结果
 * 2. 矫正图像畸变
 * 3. 坐标系转换
 * 4. 验证标定质量
 * 5. 计算地面距离
 */

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <camera_calibration.npz> <ground_calibration.npz> [test_images...]" << std::endl;
        std::cout << "Example: " << argv[0] << " camera_calib.npz ground_calib.npz test1.jpg test2.jpg" << std::endl;
        return -1;
    }

    std::string camera_npz_path = argv[1];
    std::string ground_npz_path = argv[2];

    // 收集测试图像路径
    std::vector<std::string> test_images;
    for (int i = 3; i < argc; ++i) {
        test_images.push_back(argv[i]);
    }

    // 创建标定加载器
    CalibrationLoader loader;

    std::cout << "=== Loading Calibration Files ===" << std::endl;

    // 1. 加载相机标定结果
    if (!loader.loadCameraCalibration(camera_npz_path)) {
        std::cerr << "Failed to load camera calibration from: " << camera_npz_path << std::endl;
        return -1;
    }

    // 2. 加载地面标定结果
    if (!loader.loadGroundCalibration(ground_npz_path)) {
        std::cerr << "Failed to load ground calibration from: " << ground_npz_path << std::endl;
        return -1;
    }

    // 3. 执行标定验证
    if (!test_images.empty()) {
        std::cout << "\n=== Calibration Validation ===" << std::endl;
        std::string validation_report = loader.validateCalibration(test_images);
        std::cout << validation_report << std::endl;
    }

    // 4. 演示图像矫正
    if (!test_images.empty()) {
        std::cout << "\n=== Image Undistortion Demo ===" << std::endl;

        cv::Mat input_image = cv::imread(test_images[0]);
        if (!input_image.empty()) {
            cv::Mat undistorted_image;

            if (loader.undistortImage(input_image, undistorted_image)) {
                std::cout << "Image undistortion successful!" << std::endl;
                std::cout << "Original size: " << input_image.cols << "x" << input_image.rows << std::endl;
                std::cout << "Undistorted size: " << undistorted_image.cols << "x" << undistorted_image.rows << std::endl;

                // 显示结果（如果有GUI支持）
                cv::imshow("Original Image", input_image);
                cv::imshow("Undistorted Image", undistorted_image);
                cv::waitKey(0);
                cv::destroyAllWindows();
            }
        }
    }

    // 5. 演示坐标转换
    std::cout << "\n=== Coordinate Transformation Demo ===" << std::endl;

    // 示例图像点（像素坐标）
    std::vector<cv::Point2f> image_points = {
        cv::Point2f(320, 240),   // 图像中心
        cv::Point2f(640, 360),   // 右下
        cv::Point2f(0, 0),       // 左上
        cv::Point2f(640, 0)      // 右上
    };

    // 转换为地面坐标
    std::vector<cv::Point3f> ground_points;
    if (loader.imageToGround(image_points, ground_points, 0.0)) {
        std::cout << "Image to Ground Coordinate Conversion:" << std::endl;
        for (size_t i = 0; i < image_points.size(); ++i) {
            std::cout << "Image (" << image_points[i].x << ", " << image_points[i].y << ") "
                      << "-> Ground (" << ground_points[i].x << ", " << ground_points[i].y
                      << ", " << ground_points[i].z << ") mm" << std::endl;
        }

        // 计算距离示例
        if (ground_points.size() >= 2) {
            double distance = loader.calculateGroundDistance(ground_points[0], ground_points[1]);
            std::cout << "\nDistance between first two points: " << distance << " mm" << std::endl;
        }
    }

    // 6. 演示反向转换
    if (!ground_points.empty()) {
        std::vector<cv::Point2f> back_to_image;
        if (loader.groundToImage(ground_points, back_to_image)) {
            std::cout << "\nGround to Image Coordinate Conversion (Verification):" << std::endl;
            for (size_t i = 0; i < ground_points.size(); ++i) {
                std::cout << "Ground (" << ground_points[i].x << ", " << ground_points[i].y << ") "
                          << "-> Image (" << back_to_image[i].x << ", " << back_to_image[i].y << ") pixels" << std::endl;
            }
        }
    }

    // 7. 设置地面原点示例
    std::cout << "\n=== Ground Origin Setup Demo ===" << std::endl;

    // 假设图像中的某个点作为原点
    cv::Point2f origin_in_image(320, 480);  // 图像下方的中心点
    cv::Point2f origin_in_ground(0, 0);     // 地面坐标系原点

    loader.setGroundOrigin(origin_in_image, origin_in_ground);

    // 重新转换坐标，验证原点设置
    std::vector<cv::Point3f> ground_points_with_origin;
    if (loader.imageToGround(image_points, ground_points_with_origin, 0.0)) {
        std::cout << "Coordinates with new origin:" << std::endl;
        for (size_t i = 0; i < image_points.size(); ++i) {
            std::cout << "Image (" << image_points[i].x << ", " << image_points[i].y << ") "
                      << "-> Ground (" << ground_points_with_origin[i].x << ", "
                      << ground_points_with_origin[i].y << ", "
                      << ground_points_with_origin[i].z << ") mm" << std::endl;
        }
    }

    std::cout << "\n=== Demo Complete ===" << std::endl;
    std::cout << "Calibration data has been successfully loaded and tested." << std::endl;
    std::cout << "You can now use the CalibrationLoader class in your own applications." << std::endl;

    return 0;
}

/**
 * 编译和使用说明：
 *
 * 1. 编译命令（需要OpenCV）：
 *    g++ -std=c++11 cpp_calibration_example.cpp cpp_calibration_loader.cpp \
 *        -o calibration_example `pkg-config --cflags --libs opencv4`
 *
 * 2. 如果使用Eigen：
 *    g++ -std=c++11 cpp_calibration_example.cpp cpp_calibration_loader.cpp \
 *        -o calibration_example `pkg-config --cflags --libs opencv4` -I/path/to/eigen
 *
 * 3. 运行示例：
 *    ./calibration_example camera_calibration.npz ground_calibration.npz test_image.jpg
 *
 * 注意事项：
 * - 确保npz文件路径正确
 * - 需要OpenCV 4.x或更高版本
 * - 如果使用Eigen，需要安装Eigen库
 * - 测试图像是可选的，但推荐用于验证
 */
