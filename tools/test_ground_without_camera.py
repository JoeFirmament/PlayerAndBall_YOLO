#!/usr/bin/env python3
"""
测试不加载Camera Calibration直接进行Ground Calibration的情况
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def simulate_ground_calibration_without_camera():
    """模拟没有相机标定数据时的Ground Calibration"""
    print("🔬 模拟: 不加载Camera Calibration直接进行Ground Calibration")
    print("=" * 80)

    # 1. 模拟棋盘格参数
    board_size = (9, 6)  # 9x6的内角点
    square_size = 25.0   # 25mm方格

    print("📏 棋盘格参数:")
    print(f"• 尺寸: {board_size[0]}x{board_size[1]} 内角点")
    print(f"• 方格大小: {square_size} mm")

    # 2. 模拟检测到的角点 (像素坐标)
    print("\n🎯 模拟角点检测:")

    # 创建模拟的图像角点 (假设检测成功)
    image_points = []
    for i in range(4):  # 4张图片
        # 模拟检测到的角点坐标
        img_points = np.random.rand(board_size[0] * board_size[1], 2) * 1000 + 100
        image_points.append(img_points.astype(np.float32))

    print(f"• 成功检测到 {len(image_points)} 张图片的角点")
    print("• 每张图片角点数量:", [len(pts) for pts in image_points])

    # 3. 创建世界坐标 (3D棋盘格坐标)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size  # 转换为实际尺寸

    print("\n🌍 世界坐标系统:")
    print("• Z=0 定义为棋盘格平面")
    print(f"• 棋盘格尺寸: {(board_size[0]-1)*square_size}mm x {(board_size[1]-1)*square_size}mm")

    # 4. 模拟Ground Calibration过程
    print("\n🔄 Ground Calibration过程:")

    # 步骤1: 计算单应性矩阵 (不需要相机内参)
    print("📐 步骤1: 计算单应性矩阵")
    print("   • 使用findHomography (2D点对应)")
    print("   • 不需要相机内参")

    # 模拟单应性矩阵计算
    src_points = objp[:, :2]  # 世界坐标的X,Y
    dst_points = image_points[0]  # 第一张图片的角点

    try:
        homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        print("   ✅ 单应性矩阵计算成功")
        print(f"   • 矩阵形状: {homography.shape}")
        print(f"   • 有效点比例: {np.sum(mask)/len(mask)*100:.1f}%")
    except Exception as e:
        print(f"   ❌ 单应性矩阵计算失败: {e}")
        return

    # 步骤2: 尝试计算相机高度 (需要相机内参)
    print("\n📏 步骤2: 计算相机高度")
    print("   • 需要相机内参 (camera_matrix, dist_coeffs)")
    print("   • 使用solvePnP方法")

    # 模拟没有相机标定数据的情况
    camera_matrix = None
    dist_coeffs = None

    if camera_matrix is None or dist_coeffs is None:
        print("   ⚠️  没有相机标定数据!")
        print("   • camera_matrix: None")
        print("   • dist_coeffs: None")
        print("   ❌ 无法计算相机高度")
        print("   ❌ 无法进行精确的3D坐标转换")

        # 尝试用默认参数
        print("\n🔧 尝试使用默认相机参数:")
        default_fx = 800  # 像素
        default_fy = 800
        default_cx = 320  # 假设640x480分辨率
        default_cy = 240

        default_camera_matrix = np.array([
            [default_fx, 0, default_cx],
            [0, default_fy, default_cy],
            [0, 0, 1]
        ], dtype=np.float32)

        default_dist_coeffs = np.zeros(5, dtype=np.float32)  # 假设无畸变

        print(f"   • 默认焦距: fx={default_fx}, fy={default_fy}")
        print(f"   • 默认主点: cx={default_cx}, cy={default_cy}")
        print("   • 默认畸变: 假设为0"

        try:
            retval, rvec, tvec = cv2.solvePnP(
                objp, image_points[0],
                default_camera_matrix, default_dist_coeffs
            )

            if retval:
                estimated_height = float(tvec[2][0])
                print(".1f"                print("   ⚠️  这个结果可能不准确!")
                print("   💡 原因: 使用了默认参数，没有考虑实际相机特性")
            else:
                print("   ❌ 即使使用默认参数也计算失败")

        except Exception as e:
            print(f"   ❌ 默认参数计算也失败: {e}")

    # 5. 总结影响
    print("\n" + "=" * 80)
    print("📊 结果总结:")
    print("✅ 可以完成:")
    print("   • 2D角点检测")
    print("   • 基础单应性矩阵计算")
    print("   • 简单的像素到世界的2D映射")

    print("\n❌ 无法完成或不准确:")
    print("   • 相机高度计算")
    print("   • 精确的3D坐标转换")
    print("   • 透视畸变校正")
    print("   • 准确的距离测量")

    print("\n💡 建议:")
    print("• 始终先进行Camera Calibration")
    print("• 使用高质量的标定图片")
    print("• 确保相机参数准确")
    print("• 定期重新标定相机")

def compare_with_and_without_camera_calibration():
    """比较有无相机标定的差异"""
    print("\n" + "=" * 80)
    print("⚖️  有无Camera Calibration的对比分析")
    print("=" * 80)

    comparison = {
        "项目": ["2D角点检测", "单应性矩阵", "像素映射", "3D重建", "距离测量", "高度计算", "畸变校正", "坐标精度"],
        "有相机标定": ["✅", "✅", "✅", "✅", "✅", "✅", "✅", "高(±5-10cm)"],
        "无相机标定": ["✅", "✅", "⚠️", "❌", "❌", "❌", "❌", "低(±50-100cm+)"]
    }

    print("
项目".ljust(15), "有相机标定".ljust(12), "无相机标定".ljust(12))
    print("-" * 45)

    for i, item in enumerate(comparison["项目"]):
        with_cal = comparison["有相机标定"][i]
        without_cal = comparison["无相机标定"][i]
        print("15")

    print("\n🎯 结论:")
    print("• Camera Calibration是Ground Calibration的基础")
    print("• 没有相机标定数据会严重影响测量精度")
    print("• 建议始终先进行Camera Calibration")

def main():
    """主函数"""
    print("🚀 Ground Calibration依赖性分析")
    print("=" * 80)

    # 模拟没有相机标定的情况
    simulate_ground_calibration_without_camera()

    # 对比分析
    compare_with_and_without_camera_calibration()

    print("\n" + "=" * 80)
    print("💡 关键发现:")
    print("• Ground Calibration可以部分进行，但精度严重下降")
    print("• 相机高度计算完全依赖Camera Calibration")
    print("• 3D坐标转换需要准确的相机内参")
    print("• 建议：始终先进行Camera Calibration")

if __name__ == "__main__":
    main()
