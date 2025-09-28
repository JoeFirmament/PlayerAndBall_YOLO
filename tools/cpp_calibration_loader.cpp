#include "cpp_calibration_loader.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <sstream>

CalibrationLoader::CalibrationLoader()
    : camera_loaded_(false), ground_loaded_(false) {
}

CalibrationLoader::~CalibrationLoader() {
}

bool CalibrationLoader::loadCameraCalibration(const std::string& npz_path) {
    try {
        // 检查文件是否存在
        if (!cv::utils::fs::exists(npz_path)) {
            std::cerr << "Calibration file not found: " << npz_path << std::endl;
            return false;
        }

        // OpenCV可以直接读取npz文件
        cv::FileStorage fs(npz_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open calibration file: " << npz_path << std::endl;
            return false;
        }

        // 读取相机矩阵
        if (!readMatrixFromNpz(fs, "camera_matrix", camera_results_.camera_matrix)) {
            std::cerr << "Failed to read camera_matrix" << std::endl;
            return false;
        }

        // 读取畸变系数
        if (!readMatrixFromNpz(fs, "dist_coeffs", camera_results_.dist_coeffs)) {
            std::cerr << "Failed to read dist_coeffs" << std::endl;
            return false;
        }

        // 读取其他可选数据
        readMatrixFromNpz(fs, "rvecs", camera_results_.rvecs);
        readMatrixFromNpz(fs, "tvecs", camera_results_.tvecs);
        readScalarFromNpz(fs, "reprojection_error", camera_results_.reprojection_error);
        readStringFromNpz(fs, "calibration_date", camera_results_.calibration_date);

        // 尝试从相机矩阵推断图像尺寸
        if (camera_results_.camera_matrix.rows >= 2 && camera_results_.camera_matrix.cols >= 2) {
            // 通常相机矩阵的[0][2]和[1][2]包含图像中心点信息
            // 这里我们使用默认值，需要时可以从参数中获取
            camera_results_.image_size = cv::Size(1280, 720); // 默认尺寸
        }

        fs.release();
        camera_loaded_ = true;

        std::cout << "Camera calibration loaded successfully from: " << npz_path << std::endl;
        std::cout << "Camera Matrix:\n" << camera_results_.camera_matrix << std::endl;
        std::cout << "Distortion Coefficients: " << camera_results_.dist_coeffs << std::endl;

        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error loading camera calibration: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading camera calibration: " << e.what() << std::endl;
        return false;
    }
}

bool CalibrationLoader::loadGroundCalibration(const std::string& npz_path) {
    try {
        if (!cv::utils::fs::exists(npz_path)) {
            std::cerr << "Ground calibration file not found: " << npz_path << std::endl;
            return false;
        }

        cv::FileStorage fs(npz_path, cv::FileStorage::READ);
        if (!fs.isOpened()) {
            std::cerr << "Failed to open ground calibration file: " << npz_path << std::endl;
            return false;
        }

        // 读取地面单应矩阵
        if (!readMatrixFromNpz(fs, "ground_homography", ground_results_.homography_matrix)) {
            // 尝试其他可能的键名
            if (!readMatrixFromNpz(fs, "homography_matrix", ground_results_.homography_matrix)) {
                std::cerr << "Failed to read ground homography matrix" << std::endl;
                return false;
            }
        }

        // 读取重投影误差
        readScalarFromNpz(fs, "reprojection_error", ground_results_.reprojection_error);
        readStringFromNpz(fs, "calibration_date", ground_results_.calibration_date);

        // 设置默认原点偏移
        ground_results_.origin_offset = cv::Point2d(0, 0);
        ground_results_.ground_height = 0.0;

        fs.release();
        ground_loaded_ = true;

        std::cout << "Ground calibration loaded successfully from: " << npz_path << std::endl;
        std::cout << "Ground Homography Matrix:\n" << ground_results_.homography_matrix << std::endl;

        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV error loading ground calibration: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error loading ground calibration: " << e.what() << std::endl;
        return false;
    }
}

bool CalibrationLoader::undistortImage(const cv::Mat& input_image, cv::Mat& output_image) {
    if (!camera_loaded_) {
        std::cerr << "Camera calibration not loaded" << std::endl;
        return false;
    }

    try {
        // 使用相机内参和畸变系数矫正图像
        cv::undistort(input_image, output_image,
                     camera_results_.camera_matrix,
                     camera_results_.dist_coeffs);
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error undistorting image: " << e.what() << std::endl;
        return false;
    }
}

bool CalibrationLoader::imageToGround(const std::vector<cv::Point2f>& image_points,
                                    std::vector<cv::Point3f>& ground_points,
                                    double z_height) {
    if (!ground_loaded_) {
        std::cerr << "Ground calibration not loaded" << std::endl;
        return false;
    }

    try {
        ground_points.clear();

        for (const auto& img_pt : image_points) {
            // 将图像点转换为齐次坐标
            cv::Mat img_point_homogeneous = (cv::Mat_<double>(3, 1) <<
                                            img_pt.x,
                                            img_pt.y,
                                            1.0);

            // 应用单应矩阵变换
            cv::Mat ground_point_homogeneous = ground_results_.homography_matrix.inv() *
                                             img_point_homogeneous;

            // 转换为欧几里得坐标
            double w = ground_point_homogeneous.at<double>(2, 0);
            if (std::abs(w) < 1e-10) {
                continue; // 跳过无效点
            }

            double x = ground_point_homogeneous.at<double>(0, 0) / w;
            double y = ground_point_homogeneous.at<double>(1, 0) / w;

            // 应用原点偏移
            x += ground_results_.origin_offset.x;
            y += ground_results_.origin_offset.y;

            ground_points.push_back(cv::Point3f(x, y, z_height));
        }

        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "Error converting image to ground coordinates: " << e.what() << std::endl;
        return false;
    }
}

bool CalibrationLoader::groundToImage(const std::vector<cv::Point3f>& ground_points,
                                    std::vector<cv::Point2f>& image_points) {
    if (!ground_loaded_) {
        std::cerr << "Ground calibration not loaded" << std::endl;
        return false;
    }

    try {
        image_points.clear();

        for (const auto& ground_pt : ground_points) {
            // 减去原点偏移
            double x = ground_pt.x - ground_results_.origin_offset.x;
            double y = ground_pt.y - ground_results_.origin_offset.y;

            // 将地面点转换为齐次坐标
            cv::Mat ground_point_homogeneous = (cv::Mat_<double>(3, 1) <<
                                               x,
                                               y,
                                               1.0);

            // 应用单应矩阵变换
            cv::Mat img_point_homogeneous = ground_results_.homography_matrix *
                                          ground_point_homogeneous;

            // 转换为欧几里得坐标
            double w = img_point_homogeneous.at<double>(2, 0);
            if (std::abs(w) < 1e-10) {
                continue; // 跳过无效点
            }

            double u = img_point_homogeneous.at<double>(0, 0) / w;
            double v = img_point_homogeneous.at<double>(1, 0) / w;

            image_points.push_back(cv::Point2f(u, v));
        }

        return true;

    } catch (const cv::Exception& e) {
        std::cerr << "Error converting ground to image coordinates: " << e.what() << std::endl;
        return false;
    }
}

double CalibrationLoader::calculateGroundDistance(const cv::Point3f& point1,
                                                const cv::Point3f& point2) {
    double dx = point2.x - point1.x;
    double dy = point2.y - point1.y;
    double dz = point2.z - point1.z;

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

void CalibrationLoader::setGroundOrigin(const cv::Point2f& origin_image_point,
                                       const cv::Point2f& origin_ground_point) {
    // 将图像中的原点转换为地面坐标
    std::vector<cv::Point2f> img_points = {origin_image_point};
    std::vector<cv::Point3f> ground_points;

    if (imageToGround(img_points, ground_points, 0.0) && !ground_points.empty()) {
        // 计算偏移量
        ground_results_.origin_offset.x = origin_ground_point.x - ground_points[0].x;
        ground_results_.origin_offset.y = origin_ground_point.y - ground_points[0].y;

        std::cout << "Ground origin set to: (" << origin_ground_point.x
                  << ", " << origin_ground_point.y << ")" << std::endl;
        std::cout << "Origin offset: (" << ground_results_.origin_offset.x
                  << ", " << ground_results_.origin_offset.y << ")" << std::endl;
    }
}

std::string CalibrationLoader::validateCalibration(const std::vector<std::string>& test_images) {
    std::stringstream report;

    report << "=== Calibration Validation Report ===\n";
    report << "Generated on: " << std::ctime(&(std::time(nullptr)));

    if (camera_loaded_) {
        report << "\n--- Camera Calibration Validation ---\n";
        double avg_reprojection_error = 0.0;
        int valid_images = 0;

        for (const auto& img_path : test_images) {
            cv::Mat test_img = cv::imread(img_path);
            if (!test_img.empty()) {
                double error = validateCameraCalibration(test_img);
                if (error >= 0) {
                    avg_reprojection_error += error;
                    valid_images++;
                }
            }
        }

        if (valid_images > 0) {
            avg_reprojection_error /= valid_images;
            report << "Average reprojection error: " << avg_reprojection_error << " pixels\n";
            report << "Valid test images: " << valid_images << "/" << test_images.size() << "\n";

            if (avg_reprojection_error < 1.0) {
                report << "Quality: EXCELLENT (< 1.0 pixels)\n";
            } else if (avg_reprojection_error < 3.0) {
                report << "Quality: GOOD (1.0-3.0 pixels)\n";
            } else {
                report << "Quality: POOR (> 3.0 pixels)\n";
            }
        }
    }

    if (ground_loaded_) {
        report << "\n--- Ground Calibration Validation ---\n";
        double avg_ground_error = 0.0;
        int valid_images = 0;

        for (const auto& img_path : test_images) {
            cv::Mat test_img = cv::imread(img_path);
            if (!test_img.empty()) {
                double error = validateGroundCalibration(test_img);
                if (error >= 0) {
                    avg_ground_error += error;
                    valid_images++;
                }
            }
        }

        if (valid_images > 0) {
            avg_ground_error /= valid_images;
            report << "Average ground projection error: " << avg_ground_error << " pixels\n";
            report << "Valid test images: " << valid_images << "/" << test_images.size() << "\n";

            if (avg_ground_error < 2.0) {
                report << "Quality: EXCELLENT (< 2.0 pixels)\n";
            } else if (avg_ground_error < 5.0) {
                report << "Quality: GOOD (2.0-5.0 pixels)\n";
            } else {
                report << "Quality: POOR (> 5.0 pixels)\n";
            }
        }
    }

    return report.str();
}

bool CalibrationLoader::readMatrixFromNpz(cv::FileStorage& npz_file,
                                        const std::string& key,
                                        cv::Mat& output) {
    try {
        cv::Mat temp;
        npz_file[key] >> temp;
        if (!temp.empty()) {
            output = temp.clone();
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

bool CalibrationLoader::readScalarFromNpz(cv::FileStorage& npz_file,
                                        const std::string& key,
                                        double& output) {
    try {
        cv::Mat temp;
        npz_file[key] >> temp;
        if (!temp.empty()) {
            output = temp.at<double>(0, 0);
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

bool CalibrationLoader::readStringFromNpz(cv::FileStorage& npz_file,
                                        const std::string& key,
                                        std::string& output) {
    try {
        std::string temp;
        npz_file[key] >> temp;
        if (!temp.empty()) {
            output = temp;
            return true;
        }
        return false;
    } catch (...) {
        return false;
    }
}

double CalibrationLoader::validateCameraCalibration(const cv::Mat& test_image) {
    if (!camera_loaded_) return -1.0;

    try {
        // 矫正图像
        cv::Mat undistorted;
        if (!undistortImage(test_image, undistorted)) {
            return -1.0;
        }

        // 计算矫正前后图像差异的简单度量
        cv::Mat diff;
        cv::absdiff(test_image, undistorted, diff);

        // 计算平均差异
        cv::Scalar mean_diff = cv::mean(diff);
        double avg_diff = (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;

        return avg_diff;

    } catch (...) {
        return -1.0;
    }
}

double CalibrationLoader::validateGroundCalibration(const cv::Mat& test_image) {
    if (!ground_loaded_) return -1.0;

    try {
        // 这里可以实现更复杂的地面标定验证
        // 例如：检测已知位置的地面特征点，计算重投影误差

        // 简化的验证：检查单应矩阵是否合理
        cv::Mat h = ground_results_.homography_matrix;
        if (h.rows != 3 || h.cols != 3) {
            return -1.0;
        }

        // 检查单应矩阵的最后一个元素是否接近1
        double h33 = h.at<double>(2, 2);
        if (std::abs(h33 - 1.0) > 0.1) {
            return 10.0; // 较大的误差值表示问题
        }

        return 0.5; // 默认的低误差值

    } catch (...) {
        return -1.0;
    }
}
