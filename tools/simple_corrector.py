#!/usr/bin/env python3
"""
简化的相机标定内参矫正工具

专注于内参矫正（图像去畸变），不依赖外参数据
"""

import numpy as np
import cv2
import os

class SimpleCorrector:
    """简化的标定矫正器"""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None

    def load_calibration(self, npz_path):
        """加载标定参数"""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"标定文件不存在: {npz_path}")

        data = np.load(npz_path)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']

        print(f"✅ 加载标定参数: {npz_path}")
        print(f"   相机矩阵: {self.camera_matrix.shape}")
        print(f"   畸变系数: {self.dist_coeffs.shape}")

        # 显示内参信息
        print("   内参信息:")
        print(f"     焦距 (fx, fy): {self.camera_matrix[0,0]:.2f}, {self.camera_matrix[1,1]:.2f}")
        print(f"     主点 (cx, cy): {self.camera_matrix[0,2]:.2f}, {self.camera_matrix[1,2]:.2f}")
        print(f"     畸变系数: {self.dist_coeffs.flatten()}")

        return True

    def undistort_image(self, image, alpha=0.5):
        """
        内参矫正：图像去畸变

        参数:
        image: 输入图像
        alpha: 自由缩放参数
               0: 保持所有像素（可能有黑边）
               0.5: 平衡选择（推荐）
               1: 裁剪到有效区域

        返回:
        去畸变后的图像
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("请先加载标定参数")

        h, w = image.shape[:2]

        print(f"🔧 开始图像去畸变...")
        print(f"   输入图像尺寸: {w}x{h}")

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
            print(f"   裁剪ROI: [{x},{y}] -> {w_roi}x{h_roi}")

        print("✅ 图像去畸变完成")
        print(f"   输出图像尺寸: {undistorted.shape[1]}x{undistorted.shape[0]}")
        print(f"   矫正参数: alpha={alpha}")

        return undistorted

    def undistort_points(self, points):
        """
        内参矫正：矫正图像点坐标

        参数:
        points: 图像坐标点 [(x,y), ...] 或 Nx2 数组

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

        print(f"🔧 矫正 {len(points)} 个点坐标...")

        # 去畸变点坐标
        undistorted_points = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix
        )

        result = undistorted_points.reshape(-1, 2)
        print("✅ 点坐标矫正完成")
        return result

    def evaluate_undistortion_quality(self, original_image, corrected_image):
        """
        评估去畸变质量

        参数:
        original_image: 原始图像
        corrected_image: 矫正后图像

        返回:
        质量评估字典
        """
        if original_image.shape != corrected_image.shape:
            # 调整为相同尺寸进行比较
            min_h = min(original_image.shape[0], corrected_image.shape[0])
            min_w = min(original_image.shape[1], corrected_image.shape[1])

            orig_crop = original_image[:min_h, :min_w]
            corr_crop = corrected_image[:min_h, :min_w]
        else:
            orig_crop = original_image
            corr_crop = corrected_image

        # 计算差异
        diff = cv2.absdiff(orig_crop, corr_crop)
        mean_diff = np.mean(diff)

        # 计算结构相似性指数 (SSIM)
        gray_orig = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY)
        gray_corr = cv2.cvtColor(corr_crop, cv2.COLOR_BGR2GRAY)

        # 简化的SSIM计算
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
            'ssim_score': float(ssim_score),
            'original_size': original_image.shape,
            'corrected_size': corrected_image.shape,
            'quality_assessment': 'EXCELLENT' if ssim_score > 0.95 else
                                'GOOD' if ssim_score > 0.85 else
                                'FAIR' if ssim_score > 0.7 else 'POOR'
        }

        return quality

    def batch_undistort(self, image_paths, output_dir, alpha=0.5):
        """
        批量去畸变图像

        参数:
        image_paths: 图像路径列表
        output_dir: 输出目录
        alpha: 去畸变参数
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []

        print(f"🔄 开始批量去畸变 {len(image_paths)} 张图像...")

        for i, image_path in enumerate(image_paths):
            try:
                print(f"处理 {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")

                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  ❌ 无法读取图像")
                    continue

                # 去畸变
                corrected = self.undistort_image(image, alpha)

                # 保存结果
                base_name = os.path.basename(image_path)
                output_path = os.path.join(output_dir, f"undistorted_{base_name}")
                cv2.imwrite(output_path, corrected)

                # 评估质量
                quality = self.evaluate_undistortion_quality(image, corrected)

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

        # 生成总结报告
        successful = len(results)
        total = len(image_paths)
        success_rate = (successful / total * 100) if total > 0 else 0

        print("\n📊 批量处理完成:")
        print(f"   总图像数: {total}")
        print(f"   成功处理: {successful}")
        print(f"   成功率: {success_rate:.1f}%")

        if results:
            avg_ssim = np.mean([r['quality']['ssim_score'] for r in results])
            print(f"   平均SSIM: {avg_ssim:.4f}")

        return results

def main():
    """主函数演示"""
    print("=" * 60)
    print("🎯 相机标定内参矫正工具")
    print("=" * 60)

    # 创建矫正器
    corrector = SimpleCorrector()

    # 1. 加载标定参数
    try:
        corrector.load_calibration("example_calibration.npz")
    except FileNotFoundError:
        print("❌ 请先运行标定过程生成npz文件")
        return

    # 2. 演示图像去畸变
    print("\n🔧 1. 图像去畸变演示")
    test_image = cv2.imread("simple_chessboard.jpg")
    if test_image is not None:
        print(f"原始图像: {test_image.shape[1]}x{test_image.shape[0]}")

        # 去畸变
        undistorted = corrector.undistort_image(test_image, alpha=0.5)
        cv2.imwrite("undistorted_result.jpg", undistorted)

        # 评估质量
        quality = corrector.evaluate_undistortion_quality(test_image, undistorted)
        print("\n去畸变质量评估:")
        print(f"  平均像素差异: {quality['mean_difference']:.2f}")
        print(f"  结构相似度(SSIM): {quality['ssim_score']:.4f}")
        print(f"  质量等级: {quality['quality_assessment']}")
        print(f"  输出图像: undistorted_result.jpg")

    # 3. 演示点坐标矫正
    print("\n📍 2. 点坐标矫正演示")
    test_points = [
        [100, 100],
        [200, 150],
        [300, 200]
    ]

    print(f"原始点坐标: {test_points}")
    corrected_points = corrector.undistort_points(test_points)

    print("矫正后点坐标:")
    for i, (orig, corr) in enumerate(zip(test_points, corrected_points)):
        print(f"  点 {i}: ({orig[0]:.1f}, {orig[1]:.1f}) -> ({corr[0]:.1f}, {corr[1]:.1f})")

    print("\n✅ 内参矫正演示完成!")
    print("\n📋 内参矫正功能总结:")
    print("• 🎯 图像去畸变 - 矫正镜头畸变")
    print("• 📍 点坐标矫正 - 矫正特征点位置")
    print("• 📊 质量评估 - 量化矫正效果")
    print("• 🔄 批量处理 - 高效处理多张图像")

    print("\n💡 使用建议:")
    print("• 选择alpha=0.5获得最佳平衡")
    print("• 关注SSIM分数，越接近1.0越好")
    print("• 对于高精度应用，建议多次测试不同参数")

if __name__ == "__main__":
    main()
