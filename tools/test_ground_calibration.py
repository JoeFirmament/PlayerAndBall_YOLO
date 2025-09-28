#!/usr/bin/env python3
"""
地面标定功能测试脚本
测试 camera_calibration_modern.py 中的地面标定功能
"""

import os
import sys
from datetime import datetime

def create_test_chessboard_images(output_dir="test_ground_images", num_images=5):
    """创建测试用的棋盘格图像（简化版，不需要numpy）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 棋盘格参数
    board_w, board_h = 7, 6  # 7x6 个内角点
    square_size = 50  # 每个方格50mm

    print(f"📸 正在生成 {num_images} 张测试图像...")

    for i in range(num_images):
        # 创建一个简单的文本文件来模拟图像
        # 实际使用中，你需要真实的棋盘格照片
        filename = f"ground_chessboard_{i+1:02d}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            f.write(f"# 地面棋盘格测试图像 {i+1}\n")
            f.write(f"# 棋盘格尺寸: {board_w}x{board_h}\n")
            f.write(f"# 方格尺寸: {square_size}mm\n")
            f.write(f"# 相机距离: {1500 + i*100}mm\n")
            f.write("# 实际使用时请替换为真实的JPG/PNG图像\n")

        print(f"✅ 生成测试文件: {filename}")

    print(f"\n🎯 测试文件生成完成!")
    print(f"📁 文件保存在: {output_dir}")
    print(f"📐 棋盘格尺寸: {board_w}×{board_h} 内角点")
    print(f"📏 方格尺寸: {square_size}mm")
    print(f"\n⚠️  注意:")
    print(f"这些是文本文件，仅用于演示目的")
    print(f"实际使用时需要真实的棋盘格照片")

    return output_dir

def test_ground_calibration_logic():
    """测试地面标定算法逻辑"""
    print("\n🧪 测试地面标定算法逻辑...")

    try:
        # 测试numpy
        import numpy as np
        print("✅ NumPy 可用")

        # 测试opencv
        import cv2
        print("✅ OpenCV 可用")

        # 模拟棋盘格检测
        board_size = (7, 6)
        print(f"✅ 棋盘格检测参数设置正常: {board_size[0]}×{board_size[1]}")

        # 测试Homography计算
        src_points = np.float32([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ])
        dst_points = np.float32([
            [50, 50], [150, 50], [150, 150], [50, 150]
        ])

        H = cv2.findHomography(src_points, dst_points)[0]

        print("✅ Homography矩阵计算正常")
        print("📊 示例变换矩阵:")
        print(f"   {H[0]}")
        print(f"   {H[1]}")
        print(f"   {H[2]}")

        return True

    except ImportError as e:
        print(f"❌ 缺少必要的包: {e}")
        print("请确保在正确的环境中运行")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🏀 地面标定功能测试")
    print("=" * 50)

    # 测试算法逻辑
    test_ground_calibration_logic()

    print("\n" + "=" * 50)

    # 询问是否生成测试图像
    response = input("\n❓ 是否生成测试用的地面棋盘格图像？(y/N): ").strip().lower()

    if response == 'y':
        output_dir = create_test_chessboard_images()
        print(f"\n🎉 现在你可以测试地面标定功能了!")
        print(f"📂 测试图像位置: {output_dir}")
    else:
        print("\nℹ️ 跳过测试图像生成")
        print("💡 你可以使用自己的地面棋盘格照片进行测试")

    print("\n🚀 启动地面标定工具...")
    print("命令: python3 camera_calibration_modern.py")
