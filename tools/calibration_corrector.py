#!/usr/bin/env python3
"""
相机标定内参和外参矫正工具

功能：
1. 内参矫正：图像去畸变（undistortion）
2. 外参矫正：3D坐标变换
3. 联合矫正：图像坐标 → 世界坐标 + 去畸变
4. 矫正效果评估
"""

import numpy as np
import cv2
import os
import json
from datetime import datetime

class CalibrationCorrector:
    """标定参数矫正器"""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.image_size = None

    def load_calibration(self, npz_path):
        """加载标定参数"""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"标定文件不存在: {npz_path}")

        data = np.load(npz_path)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        # 可选参数
        if 'rvecs' in data:
            self.rvecs = data['rvecs']
            # 确保rvecs是正确的格式
            if self.rvecs.ndim == 1:
                self.rvecs = self.rvecs.reshape(1, -1)
        if 'tvecs' in data:
            self.tvecs = data['tvecs']
            # 确保tvecs是正确的格式
            if self.tvecs.ndim == 1:
                self.tvecs = self.tvecs.reshape(1, -1)
        if 'image_size' in data:
            self.image_size = tuple(data['image_size'])

        print(f"✅ 加载标定参数: {npz_path}")
        print(f"   相机矩阵: {self.camera_matrix.shape}")
        print(f"   畸变系数: {self.dist_coeffs.shape}")
        if self.rvecs is not None:
            print(f"   旋转向量: {self.rvecs.shape}")
        if self.tvecs is not None:
            print(f"   平移向量: {self.tvecs.shape}")

        return True

    def undistort_image(self, image, alpha=0.0):
        """
        内参矫正：图像去畸变

        参数:
        image: 输入图像
        alpha: 自由缩放参数 (0-1)
               0: 保持所有像素（可能有黑边）
               1: 裁剪到有效区域
               0.5: 平衡选择

        返回:
        去畸变后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("请先加载标定参数")

        h, w = image.shape[:2]

        # 计算最佳的去畸变映射
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), alpha, (w, h)
        )

        # 去畸变
        undistorted = cv2.undistort(image, self.camera_matrix,
                                   self.dist_coeffs, None, new_camera_matrix)

        # 如果alpha<1，可能需要裁剪ROI
        if alpha < 1.0:
            x, y, w_roi, h_roi = roi
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]

        print("✅ 图像去畸变完成")
        print(f"   原始尺寸: {image.shape[1]}x{image.shape[0]}")
        print(f"   矫正尺寸: {undistorted.shape[1]}x{undistorted.shape[0]}")
        print(f"   矫正参数: alpha={alpha}")
        return undistorted

    def undistort_points(self, points, image_size=None):
        """
        内参矫正：矫正图像点坐标

        参数:
        points: 图像坐标点 [(x,y), ...] 或 Nx2 数组
        image_size: 图像尺寸 (w, h)，可选

        返回:
        去畸变后的坐标点
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("请先加载标定参数")

        # 转换为numpy数组
        if isinstance(points, list):
            points = np.array(points, dtype=np.float32)
        elif not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)

        if points.ndim == 1:
            points = points.reshape(-1, 2)

        # 去畸变点坐标
        undistorted_points = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix
        )

        return undistorted_points.reshape(-1, 2)

    def transform_3d_points(self, points_3d, rvec, tvec):
        """
        外参矫正：3D坐标变换

        参数:
        points_3d: 3D世界坐标点 Nx3
        rvec: 旋转向量 3x1
        tvec: 平移向量 3x1

        返回:
        变换后的3D坐标
        """
        if isinstance(points_3d, list):
            points_3d = np.array(points_3d, dtype=np.float32)

        # 应用刚体变换
        # points_camera = R * points_world + t
        transformed_points, _ = cv2.projectPoints(
            points_3d, rvec, tvec,
            np.eye(3), np.zeros(5)  # 无畸变
        )

        return transformed_points.reshape(-1, 2)

    def image_to_world_coordinates(self, image_points, rvec, tvec, z=0.0):
        """
        联合矫正：图像坐标 → 世界坐标

        参数:
        image_points: 图像坐标点 Nx2
        rvec: 旋转向量
        tvec: 平移向量
        z: 世界坐标Z值（默认地面高度为0）

        返回:
        世界坐标点 Nx3
        """
        if isinstance(image_points, list):
            image_points = np.array(image_points, dtype=np.float32)

        # 1. 去畸变图像点
        undistorted_points = self.undistort_points(image_points)

        # 2. 图像坐标 → 归一化坐标
        # 使用相机内参的逆矩阵
        camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # 转换为齐次坐标
        homogeneous_points = np.column_stack([
            undistorted_points,
            np.ones(len(undistorted_points))
        ])

        # 归一化相机坐标
        normalized_points = (camera_matrix_inv @ homogeneous_points.T).T
        normalized_points = normalized_points[:, :2] / normalized_points[:, 2:3]

        # 3. 反投影到3D世界坐标
        # 假设Z = z（例如地面为0）
        world_points = []
        for point in normalized_points:
            # 从相机坐标系转换到世界坐标系
            # point_world = R^T * (point_camera - t)
            R, _ = cv2.Rodrigues(rvec)
            point_camera = np.array([point[0], point[1], 1.0])

            # 反投影
            world_point = np.linalg.inv(R) @ (point_camera - tvec.ravel())
            # 归一化
            world_point = world_point / world_point[2] * z

            world_points.append([world_point[0], world_point[1], z])

        return np.array(world_points)

    def evaluate_correction_quality(self, original_image, corrected_image):
        """
        评估矫正质量

        参数:
        original_image: 原始图像
        corrected_image: 矫正后图像

        返回:
        质量评估字典
        """
        if original_image.shape != corrected_image.shape:
            # 如果尺寸不同，调整为相同尺寸进行比较
            min_h = min(original_image.shape[0], corrected_image.shape[0])
            min_w = min(original_image.shape[1], corrected_image.shape[1])

            original_crop = original_image[:min_h, :min_w]
            corrected_crop = corrected_image[:min_h, :min_w]
        else:
            original_crop = original_image
            corrected_crop = corrected_image

        # 计算差异
        diff = cv2.absdiff(original_crop, corrected_crop)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        # 计算结构相似性指数 (SSIM)
        gray_orig = cv2.cvtColor(original_crop, cv2.COLOR_BGR2GRAY)
        gray_corr = cv2.cvtColor(corrected_crop, cv2.COLOR_BGR2GRAY)

        # 简单的SSIM计算
        mu1 = cv2.GaussianBlur(gray_orig, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray_corr, (11, 11), 1.5)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(gray_orig * gray_orig, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(gray_corr * gray_corr, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(gray_orig * gray_corr, (11, 11), 1.5) - mu1_mu2

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
        denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

        ssim_map = numerator / denominator
        ssim_score = np.mean(ssim_map)

        quality = {
            'mean_difference': float(mean_diff),
            'max_difference': float(max_diff),
            'ssim_score': float(ssim_score),
            'original_size': original_image.shape,
            'corrected_size': corrected_image.shape,
            'quality_assessment': 'GOOD' if ssim_score > 0.8 else 'FAIR' if ssim_score > 0.6 else 'POOR'
        }

        return quality

    def batch_correct_images(self, image_paths, output_dir, alpha=0.0):
        """
        批量矫正图像

        参数:
        image_paths: 图像路径列表
        output_dir: 输出目录
        alpha: 去畸变参数
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []

        for i, image_path in enumerate(image_paths):
            try:
                print(f"处理图像 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ❌ 无法读取图像: {image_path}")
                    continue

                # 去畸变
                corrected = self.undistort_image(image, alpha)

                # 保存结果
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"corrected_{base_name}")
                cv2.imwrite(output_path, corrected)

                # 评估质量
                quality = self.evaluate_correction_quality(image, corrected)

                result = {
                    'original_path': image_path,
                    'corrected_path': output_path,
                    'quality': quality
                }

                results.append(result)
                print(f"  ✅ 已保存: {output_path}")

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                continue

        # 保存批量处理报告
        report_path = os.path.join(output_dir, "correction_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'batch_info': {
                    'total_images': len(image_paths),
                    'successful_corrections': len(results),
                    'timestamp': datetime.now().isoformat()
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n📄 批量处理报告已保存: {report_path}")
        return results

def main():
    """主函数演示"""
    print("=" * 60)
    print("🎯 相机标定内参外参矫正工具演示")
    print("=" * 60)

    # 创建矫正器
    corrector = CalibrationCorrector()

    # 1. 加载标定参数
    try:
        corrector.load_calibration("example_calibration.npz")
    except FileNotFoundError:
        print("❌ 请先运行标定生成npz文件")
        return

    # 2. 演示图像去畸变
    print("\n🔧 1. 图像去畸变演示")
    test_image = cv2.imread("simple_chessboard.jpg")
    if test_image is not None:
        print(f"原始图像尺寸: {test_image.shape[1]}x{test_image.shape[0]}")

        # 去畸变
        undistorted = corrector.undistort_image(test_image, alpha=0.5)
        cv2.imwrite("undistorted_chessboard.jpg", undistorted)

        # 评估质量
        quality = corrector.evaluate_correction_quality(test_image, undistorted)
        print("去畸变质量评估:")
        print(f"  平均差异: {quality['mean_difference']:.2f}")
        print(f"  SSIM分数: {quality['ssim_score']:.4f}")
        print(f"  质量等级: {quality['quality_assessment']}")

    # 3. 演示坐标变换
    print("\n🔄 2. 坐标变换演示")
    if corrector.rvecs is not None and corrector.tvecs is not None:
        try:
            # 检查rvecs和tvecs的形状
            print(f"rvecs shape: {corrector.rvecs.shape}")
            print(f"tvecs shape: {corrector.tvecs.shape}")

            # 获取第一个视图的参数
            if corrector.rvecs.shape[0] > 0:
                rvec = corrector.rvecs[0].flatten()  # 确保是1D数组
                tvec = corrector.tvecs[0].flatten()

                print(f"使用第一个视图的参数:")
                print(f"  旋转向量: {rvec}")
                print(f"  平移向量: {tvec}")

                # 模拟一些图像点
                image_points = np.array([
                    [100, 100],
                    [200, 150],
                    [300, 200]
                ], dtype=np.float32)

                # 转换为世界坐标
                world_points = corrector.image_to_world_coordinates(
                    image_points, rvec, tvec, z=0.0
                )

                print("坐标变换结果:")
                for i, (img_pt, world_pt) in enumerate(zip(image_points, world_points)):
                    print(f"  图像点 {i}: ({img_pt[0]:.1f}, {img_pt[1]:.1f})")
                    print(f"  世界坐标: ({world_pt[0]:.2f}, {world_pt[1]:.2f}, {world_pt[2]:.2f}) mm")
            else:
                print("  ❌ 没有有效的视图参数")

        except Exception as e:
            print(f"  ❌ 坐标变换演示失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ 矫正工具演示完成!")
    print("可用的矫正功能:")
    print("• undistort_image() - 图像去畸变")
    print("• undistort_points() - 点坐标去畸变")
    print("• image_to_world_coordinates() - 图像到世界坐标转换")
    print("• batch_correct_images() - 批量图像矫正")

if __name__ == "__main__":
    main()
