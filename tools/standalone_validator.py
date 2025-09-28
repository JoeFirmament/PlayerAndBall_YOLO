#!/usr/bin/env python3
"""
独立标定验证器 - 不依赖GUI组件

用于直接加载npz文件和图片进行标定验证的独立工具
"""

import numpy as np
import cv2
import os
import json
from datetime import datetime

class StandaloneValidator:
    """独立标定验证器"""

    def __init__(self):
        pass

    def validate_calibration_from_file(self, npz_path, image_path, board_size=(7, 6), square_size=25.0):
        """
        直接加载npz文件和图片进行标定验证

        参数:
        npz_path: str - 标定结果npz文件路径
        image_path: str - 测试图片路径
        board_size: tuple - 棋盘格尺寸 (内角点数)
        square_size: float - 棋盘格方格尺寸 (mm)

        返回:
        dict - 验证结果
        """
        try:
            # 1. 加载标定参数
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"NPZ file not found: {npz_path}")

            calibration_data = np.load(npz_path)
            camera_matrix = calibration_data['camera_matrix']
            dist_coeffs = calibration_data['dist_coeffs']

            print(f"✅ 加载标定文件: {npz_path}")
            print(f"   相机矩阵形状: {camera_matrix.shape}")
            print(f"   畸变系数形状: {dist_coeffs.shape}")

            # 2. 验证单张图片
            result = self.validate_single_image_with_params(
                image_path, camera_matrix, dist_coeffs,
                board_size, square_size
            )

            if result is None:
                return {
                    'success': False,
                    'error': 'Failed to validate image - no chessboard corners found',
                    'image_path': image_path,
                    'npz_path': npz_path
                }

            # 3. 返回详细结果
            validation_result = {
                'success': True,
                'image_path': image_path,
                'npz_path': npz_path,
                'board_size': board_size,
                'square_size': square_size,
                'mean_error': result['mean_error'],
                'max_error': result['max_error'],
                'min_error': result['min_error'],
                'corners_found': result['corners_found'],
                'errors': result['errors'],
                'quality_assessment': self.assess_calibration_quality(result['mean_error']),
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'timestamp': datetime.now().isoformat()
            }

            return validation_result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path,
                'npz_path': npz_path
            }

    def validate_single_image_with_params(self, image_path, camera_matrix, dist_coeffs,
                                        board_size=(7, 6), square_size=25.0):
        """
        使用指定参数验证单张图片

        验证算法步骤:
        1. 读取图片并转换为灰度图
        2. 使用cv2.findChessboardCorners检测棋盘格角点
        3. 使用cv2.cornerSubPix精确化角点位置
        4. 生成世界坐标点（3D坐标）
        5. 使用cv2.solvePnP计算相机位姿
        6. 使用cv2.projectPoints进行重投影
        7. 计算重投影误差（L2范数）
        8. 返回详细的验证结果

        参数:
        image_path: str - 图片路径
        camera_matrix: np.ndarray - 相机内参矩阵 3x3
        dist_coeffs: np.ndarray - 畸变系数
        board_size: tuple - 棋盘格尺寸 (width, height) 内角点数
        square_size: float - 棋盘格方格尺寸 (mm)

        返回:
        dict - 验证结果或None（失败时）
        """
        try:
            print(f"🔍 验证图片: {image_path}")

            # 1. 读取和预处理图像
            img = cv2.imread(image_path)
            if img is None:
                print(f"   ❌ 无法加载图片: {image_path}")
                return None

            print(f"   📏 图片尺寸: {img.shape[1]}x{img.shape[0]}")

            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. 检测棋盘格角点
            print(f"   🎯 检测棋盘格角点: {board_size[0]}×{board_size[1]}")
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)

            if not ret:
                print(f"   ❌ 未检测到棋盘格角点")
                return None

            print(f"   ✅ 检测到 {len(corners)} 个角点")

            # 3. 精确化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 4. 生成世界坐标点（3D坐标系）
            # 棋盘格在世界坐标系中的位置，Z=0（假设在XY平面上）
            objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size

            print(f"   📐 生成 {len(objp)} 个世界坐标点")
            print(f"   📏 坐标范围: X[0-{board_size[0]*square_size}]mm, Y[0-{board_size[1]*square_size}]mm")

            # 5. 计算相机位姿（PnP问题求解）
            # 使用已知的3D世界坐标点和对应的2D图像坐标点
            # 求解相机在世界坐标系中的旋转向量和平移向量
            retval, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)

            if not retval:
                print(f"   ❌ PnP求解失败")
                return None

            print(f"   📍 相机位姿已计算")

            # 6. 重投影验证
            # 使用计算得到的位姿将3D世界坐标点投影回图像平面
            projected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)

            # 7. 计算重投影误差
            # 比较投影点与实际检测点的距离（像素单位）
            errors = []
            for i, (projected, actual) in enumerate(zip(projected_points, corners2)):
                # 计算欧几里得距离（L2范数）
                error = np.linalg.norm(projected[0] - actual[0])
                errors.append(error)

            # 8. 计算统计信息
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)

            print(f"   📊 重投影误差统计:")
            print(f"      平均误差: {mean_error:.4f} 像素")
            print(f"      最大误差: {max_error:.4f} 像素")
            print(f"      最小误差: {min_error:.4f} 像素")

            # 9. 返回结果
            return {
                'mean_error': mean_error,
                'max_error': max_error,
                'min_error': min_error,
                'corners_found': len(corners2),
                'errors': errors,
                'projected_points': projected_points,
                'detected_corners': corners2,
                'world_points': objp,
                'rvec': rvec,
                'tvec': tvec
            }

        except Exception as e:
            print(f"   ❌ 验证过程中出错: {e}")
            return None

    def assess_calibration_quality(self, mean_error):
        """评估标定质量"""
        if mean_error < 0.5:
            return "EXCELLENT"
        elif mean_error < 1.0:
            return "GOOD"
        elif mean_error < 2.0:
            return "ACCEPTABLE"
        else:
            return "POOR - Recalibration Recommended"

def main():
    """主函数 - 演示验证功能"""
    print("=" * 70)
    print("🎯 标定验证算法演示")
    print("=" * 70)

    # 创建验证器
    validator = StandaloneValidator()

    # 测试文件路径
    npz_path = "example_calibration.npz"
    image_path = "test_calibration.jpg"

    # 检查文件
    if not os.path.exists(npz_path):
        print(f"❌ 标定文件不存在: {npz_path}")
        print("请先运行标定过程生成npz文件")
        return

    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        print("请准备一张包含棋盘格的测试图片")
        return

    print("📁 验证文件:")
    print(f"   NPZ文件: {npz_path}")
    print(f"   测试图片: {image_path}")
    print()

    # 执行验证
    print("🔬 执行验证算法...")
    print("验证步骤:")
    print("   1. 加载标定参数 (camera_matrix, dist_coeffs)")
    print("   2. 读取测试图片")
    print("   3. 检测棋盘格角点 (findChessboardCorners)")
    print("   4. 精确化角点位置 (cornerSubPix)")
    print("   5. 生成世界坐标点")
    print("   6. 计算相机位姿 (solvePnP)")
    print("   7. 重投影验证 (projectPoints)")
    print("   8. 计算重投影误差")
    print()

    result = validator.validate_calibration_from_file(
        npz_path=npz_path,
        image_path=image_path,
        board_size=(7, 6),    # 7x6内角点
        square_size=25.0      # 25mm方格
    )

    print("\n" + "=" * 70)
    print("📊 验证结果")
    print("=" * 70)

    if result['success']:
        print("✅ 验证成功!")
        print(f"🎯 平均重投影误差: {result['mean_error']:.4f} 像素")
        print(f"📈 最大误差: {result['max_error']:.4f} 像素")
        print(f"📉 最小误差: {result['min_error']:.4f} 像素")
        print(f"🎪 检测到角点: {result['corners_found']} 个")
        print(f"🏆 质量评估: {result['quality_assessment']}")

        # 算法解释
        print("\n🔍 算法详解:")
        print("   • 检测方法: Harris角点检测 + 棋盘格模式识别")
        print(f"   • 角点数量: {result['board_size'][0]}×{result['board_size'][1]} = {result['board_size'][0]*result['board_size'][1]}")
        print(f"   • 方格尺寸: {result['square_size']}mm")
        print("   • 误差计算: L2范数 (欧几里得距离)")
        print("   • PnP方法: 使用世界坐标和图像坐标求解相机位姿")
        print("   • 验证原理: 比较实际角点与重投影角点的距离")

        # 保存结果
        output_file = "validation_result.json"
        result_copy = result.copy()

        # 转换numpy数组为列表
        if 'camera_matrix' in result_copy:
            result_copy['camera_matrix'] = result_copy['camera_matrix'].tolist()
        if 'dist_coeffs' in result_copy:
            result_copy['dist_coeffs'] = result_copy['dist_coeffs'].tolist()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_copy, f, indent=2, ensure_ascii=False)

        print(f"\n📄 结果已保存到: {output_file}")

    else:
        print("❌ 验证失败!")
        print(f"错误信息: {result['error']}")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
