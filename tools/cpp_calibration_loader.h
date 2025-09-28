#ifndef CPP_CALIBRATION_LOADER_H
#define CPP_CALIBRATION_LOADER_H

/**
 * C++相机标定结果加载器
 * 用于读取Python工具生成的npz标定文件
 *
 * 使用方法:
 * 1. 包含此头文件
 * 2. 创建CalibrationLoader实例
 * 3. 调用loadCalibrationResults()加载npz文件
 * 4. 使用提供的矫正和转换函数
 */

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>  // 如果使用Eigen进行矩阵运算

class CalibrationLoader {
public:
    /**
     * 相机标定结果结构体
     */
    struct CameraCalibrationResults {
        cv::Mat camera_matrix;           // 相机内参矩阵 3x3
        cv::Mat dist_coeffs;             // 畸变系数
        cv::Mat rvecs;                   // 旋转向量
        cv::Mat tvecs;                   // 平移向量
        double reprojection_error;       // 重投影误差
        cv::Size image_size;             // 图像尺寸
        std::string calibration_date;    // 标定日期
    };

    /**
     * 地面标定结果结构体
     */
    struct GroundCalibrationResults {
        cv::Mat homography_matrix;       // 地面单应矩阵 3x3
        double reprojection_error;       // 重投影误差
        cv::Point2d origin_offset;       // 坐标系原点偏移
        double ground_height;            // 地面高度
        std::string calibration_date;    // 标定日期
    };

    CalibrationLoader();
    ~CalibrationLoader();

    /**
     * 从npz文件加载相机标定结果
     * @param npz_path npz文件路径
     * @return 是否加载成功
     */
    bool loadCameraCalibration(const std::string& npz_path);

    /**
     * 从npz文件加载地面标定结果
     * @param npz_path npz文件路径
     * @return 是否加载成功
     */
    bool loadGroundCalibration(const std::string& npz_path);

    /**
     * 矫正图像畸变
     * @param input_image 输入图像
     * @param output_image 输出图像
     * @return 是否矫正成功
     */
    bool undistortImage(const cv::Mat& input_image, cv::Mat& output_image);

    /**
     * 将图像点转换为地面坐标
     * @param image_points 图像坐标点
     * @param ground_points 输出地面坐标点
     * @param z_height 目标高度 (相对于地面)
     * @return 是否转换成功
     */
    bool imageToGround(const std::vector<cv::Point2f>& image_points,
                      std::vector<cv::Point3f>& ground_points,
                      double z_height = 0.0);

    /**
     * 将地面坐标转换为图像坐标
     * @param ground_points 地面坐标点
     * @param image_points 输出图像坐标点
     * @return 是否转换成功
     */
    bool groundToImage(const std::vector<cv::Point3f>& ground_points,
                      std::vector<cv::Point2f>& image_points);

    /**
     * 计算两点间的地面距离
     * @param point1 第一个地面点
     * @param point2 第二个地面点
     * @return 距离（单位：毫米）
     */
    double calculateGroundDistance(const cv::Point3f& point1, const cv::Point3f& point2);

    /**
     * 设置地面坐标系原点
     * @param origin_image_point 原点在图像中的位置
     * @param origin_ground_point 原点在地面坐标系中的位置
     */
    void setGroundOrigin(const cv::Point2f& origin_image_point,
                        const cv::Point2f& origin_ground_point = cv::Point2f(0, 0));

    /**
     * 获取相机标定结果
     */
    const CameraCalibrationResults& getCameraResults() const { return camera_results_; }

    /**
     * 获取地面标定结果
     */
    const GroundCalibrationResults& getGroundResults() const { return ground_results_; }

    /**
     * 验证标定结果质量
     * @param test_images 验证图像路径列表
     * @return 验证报告
     */
    std::string validateCalibration(const std::vector<std::string>& test_images);

private:
    CameraCalibrationResults camera_results_;
    GroundCalibrationResults ground_results_;
    bool camera_loaded_;
    bool ground_loaded_;

    /**
     * 从npz文件读取矩阵数据
     * @param npz_file FileStorage对象
     * @param key 键名
     * @param output 输出矩阵
     * @return 是否读取成功
     */
    bool readMatrixFromNpz(cv::FileStorage& npz_file, const std::string& key, cv::Mat& output);

    /**
     * 从npz文件读取标量数据
     * @param npz_file FileStorage对象
     * @param key 键名
     * @param output 输出值
     * @return 是否读取成功
     */
    bool readScalarFromNpz(cv::FileStorage& npz_file, const std::string& key, double& output);

    /**
     * 从npz文件读取字符串数据
     * @param npz_file FileStorage对象
     * @param key 键名
     * @param output 输出字符串
     * @return 是否读取成功
     */
    bool readStringFromNpz(cv::FileStorage& npz_file, const std::string& key, std::string& output);

    /**
     * 执行相机标定验证
     * @param test_image 测试图像
     * @return 验证结果
     */
    double validateCameraCalibration(const cv::Mat& test_image);

    /**
     * 执行地面标定验证
     * @param test_image 测试图像
     * @return 验证结果
     */
    double validateGroundCalibration(const cv::Mat& test_image);
};

#endif // CPP_CALIBRATION_LOADER_H
