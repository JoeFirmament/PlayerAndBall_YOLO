#!/usr/bin/env python3
"""
广角摄像头棋盘格检测调试脚本
专门针对广角摄像头优化检测参数
"""

import cv2
import numpy as np
import os
from pathlib import Path

def debug_wide_angle_chessboard(image_path, board_size=(9, 6)):
    """
    广角摄像头棋盘格检测调试
    """
    print(f"🔍 调试广角摄像头棋盘格检测: {image_path}")
    print(f"📏 棋盘格尺寸: {board_size[0]}x{board_size[1]}")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图像")
        return False

    print(f"📐 图像尺寸: {img.shape[1]}x{img.shape[0]}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示图像统计信息
    print("📊 图像统计:")
    print(f"   • 平均亮度: {gray.mean():.1f}")
    print(f"   • 对比度: {gray.std():.1f}")
    print(f"   • 像素范围: {gray.min()} - {gray.max()}")

    # 尝试多种检测参数组合
    detection_configs = [
        {
            'name': '标准检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
        },
        {
            'name': '宽松检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH
        },
        {
            'name': '快速检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        },
        {
            'name': '无过滤检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        }
    ]

    # 尝试不同的棋盘格尺寸
    board_sizes_to_try = [
        board_size,
        (board_size[0]-1, board_size[1]),
        (board_size[0], board_size[1]-1),
        (board_size[0]-1, board_size[1]-1),
        (board_size[0]-2, board_size[1]-1),
        (board_size[0]-1, board_size[1]-2),
        (7, 5),
        (6, 4),
        (5, 4)
    ]

    best_result = None
    max_corners = 0

    print("
🔍 开始检测..."    for size in board_sizes_to_try:
        if size[0] < 4 or size[1] < 4:
            continue

        print(f"\n📏 测试尺寸 {size[0]}x{size[1]}:")

        for config in detection_configs:
            ret, corners = cv2.findChessboardCorners(gray, size, config['flags'])

            if ret:
                corners_count = len(corners)
                print(f"   ✅ {config['name']}: 找到 {corners_count} 个角点")

                if corners_count > max_corners:
                    max_corners = corners_count
                    best_result = {
                        'size': size,
                        'config': config['name'],
                        'corners': corners,
                        'corners_count': corners_count
                    }

                # 如果找到了足够多的角点，就停止测试
                expected_corners = size[0] * size[1]
                if corners_count >= expected_corners * 0.8:  # 80%的角点
                    print(f"   🎯 检测质量良好 ({corners_count}/{expected_corners})")
                    break
            else:
                print(f"   ❌ {config['name']}: 未找到棋盘格")

        # 如果找到了好的结果，就停止测试其他尺寸
        if best_result and best_result['corners_count'] >= size[0] * size[1] * 0.7:
            break

    # 显示最佳结果
    if best_result:
        print("
🏆 最佳检测结果:"        print(f"   • 棋盘格尺寸: {best_result['size'][0]}x{best_result['size'][1]}")
        print(f"   • 检测方法: {best_result['config']}")
        print(f"   • 角点数量: {best_result['corners_count']}")
        expected = best_result['size'][0] * best_result['size'][1]
        print(f"   • 检测率: {best_result['corners_count']/expected*100:.1f}%")

        # 细化角点并显示坐标范围
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, best_result['corners'], (11, 11), (-1, -1), criteria)

        corners_array = np.array(corners_refined)
        x_coords = corners_array[:, 0, 0]
        y_coords = corners_array[:, 0, 1]

        print(f"   • X坐标范围: {x_coords.min():.1f} - {x_coords.max():.1f}")
        print(f"   • Y坐标范围: {y_coords.min():.1f} - {y_coords.max():.1f}")

        # 在图像上绘制角点
        img_with_corners = cv2.drawChessboardCorners(img.copy(), best_result['size'], corners_refined, True)

        # 保存结果图像
        output_path = image_path.replace('.jpg', '_chessboard_detected.jpg')
        cv2.imwrite(output_path, img_with_corners)
        print(f"   • 结果已保存: {output_path}")

        return True
    else:
        print("
❌ 未找到任何棋盘格"        print("💡 建议:")
        print("   1. 检查棋盘格是否清晰可见")
        print("   2. 尝试更小的棋盘格尺寸")
        print("   3. 改善光照条件")
        print("   4. 调整摄像头角度")
        print("   5. 减少畸变（如果可能）")
        return False

def analyze_folder(folder_path):
    """
    分析文件夹中的所有图像
    """
    print(f"🔍 分析文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    print(f"📄 找到 {len(image_files)} 个JPG文件")

    success_count = 0
    total_count = min(3, len(image_files))  # 只测试前3个文件

    for i, filename in enumerate(image_files[:total_count], 1):
        image_path = os.path.join(folder_path, filename)
        print(f"\n🖼️ 测试图像 {i}/{total_count}: {filename}")

        if debug_wide_angle_chessboard(image_path):
            success_count += 1

    print("
📊 总体结果:"    print(f"• 测试图像: {total_count}")
    print(f"• 成功检测: {success_count}")
    print(".1f")

    if success_count > 0:
        print("✅ 检测成功！可以尝试在GUI中运行Ground Calibration")
    else:
        print("❌ 所有图像检测失败")
        print("   需要进一步调整检测参数或图像质量")

if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"

    try:
        analyze_folder(image_folder)
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        print("请确保安装了OpenCV: pip install opencv-python")
