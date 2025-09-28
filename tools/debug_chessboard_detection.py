#!/usr/bin/env python3
"""
调试棋盘格检测脚本
检查test_ground_images中的图片是否包含棋盘格
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path

def debug_chessboard_detection(image_folder, board_size=(9, 6)):
    """
    调试棋盘格检测
    """
    print("🔍 开始调试棋盘格检测...")
    print(f"📂 图片文件夹: {image_folder}")
    print(f"📏 棋盘格尺寸: {board_size[0]}x{board_size[1]}")

    # 获取所有jpg文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    print(f"📄 找到 {len(image_files)} 个JPG文件")

    detection_results = []

    for i, filename in enumerate(image_files[:5], 1):  # 只检查前5个文件作为示例
        image_path = os.path.join(image_folder, filename)
        print(f"\n🖼️ 检查图片 {i}: {filename}")

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ 无法读取图像: {filename}")
            continue

        print(f"   📐 图像尺寸: {img.shape[1]}x{img.shape[0]}")

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        print(f"   🔍 检测棋盘格角点 (尺寸: {board_size[0]}x{board_size[1]})...")
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            print(f"   ✅ 找到棋盘格! 角点数量: {len(corners)}")

            # 细化角点位置
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 在图像上绘制角点
            img_with_corners = cv2.drawChessboardCorners(img.copy(), board_size, corners_refined, ret)

            detection_results.append({
                'filename': filename,
                'detected': True,
                'corners_count': len(corners),
                'image_with_corners': img_with_corners
            })

        else:
            print("   ❌ 未找到棋盘格")
            print("   📊 图像统计信息:")
            print(f"      平均亮度: {gray.mean():.1f}")
            print(f"      对比度: {gray.std():.1f}")
            print(f"      最小值: {gray.min()}")
            print(f"      最大值: {gray.max()}")

            detection_results.append({
                'filename': filename,
                'detected': False,
                'corners_count': 0,
                'original_image': img
            })

    # 显示结果统计
    print("\n📊 检测结果统计:")
    detected_count = sum(1 for r in detection_results if r['detected'])
    total_count = len(detection_results)
    print(f"• 检测成功: {detected_count}/{total_count}")
    print(".1f")
    # 显示检测到的棋盘格图像
    if detected_count > 0:
        print("\n🖼️ 显示检测到的棋盘格...")
        plt.figure(figsize=(15, 10))

        for i, result in enumerate(detection_results):
            if result['detected']:
                plt.subplot(2, 3, i+1)
                plt.imshow(cv2.cvtColor(result['image_with_corners'], cv2.COLOR_BGR2RGB))
                plt.title(f"{result['filename']}\n{result['corners_count']} corners")
                plt.axis('off')

        plt.tight_layout()
        plt.show()
    else:
        print("\n❌ 没有检测到任何棋盘格")
        print("可能的问题:")
        print("1. 图片中没有棋盘格")
        print("2. 棋盘格尺寸设置不正确")
        print("3. 图像质量太差")
        print("4. 视角角度太大")
        print("5. 光照条件不佳")

        # 显示前几个失败的图像
        plt.figure(figsize=(15, 10))
        for i, result in enumerate(detection_results[:3]):
            plt.subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB))
            plt.title(f"{result['filename']}\nNo chessboard detected")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    return detection_results

def test_different_board_sizes(image_path, board_sizes=[(9,6), (8,6), (7,7), (6,6), (10,7)]):
    """
    测试不同的棋盘格尺寸
    """
    print(f"\n🔧 测试不同棋盘格尺寸: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for board_size in board_sizes:
        print(f"测试尺寸 {board_size[0]}x{board_size[1]}...")
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            print(f"  ✅ 找到棋盘格! 角点数量: {len(corners)}")
            return board_size
        else:
            print("  ❌ 未找到")

    print("所有尺寸都无法检测到棋盘格")
    return None

if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"

    if not os.path.exists(image_folder):
        print(f"❌ 文件夹不存在: {image_folder}")
        exit(1)

    # 调试棋盘格检测
    results = debug_chessboard_detection(image_folder)

    # 如果没有检测到，测试不同的棋盘格尺寸
    if not any(r['detected'] for r in results):
        print("\n🔄 尝试不同的棋盘格尺寸...")
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        if image_files:
            first_image = os.path.join(image_folder, image_files[0])
            correct_size = test_different_board_sizes(first_image)
            if correct_size:
                print(f"\n建议使用棋盘格尺寸: {correct_size[0]}x{correct_size[1]}")
            else:
                print("\n❌ 无法确定正确的棋盘格尺寸")
                print("请检查:")
                print("1. 图片中是否真的有棋盘格")
                print("2. 棋盘格是否清晰可见")
                print("3. 图像质量是否足够")
                print("4. 光照和对比度是否合适")
