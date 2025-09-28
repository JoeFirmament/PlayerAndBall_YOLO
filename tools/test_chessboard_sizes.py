#!/usr/bin/env python3
"""
测试不同棋盘格尺寸的脚本
"""

import os
import sys
from PIL import Image

def test_chessboard_sizes(image_folder, test_sizes=[(9,6), (8,6), (7,7), (6,6), (10,7)]):
    """
    测试不同的棋盘格尺寸
    """
    print("🔍 测试不同棋盘格尺寸...")
    print(f"📂 图片文件夹: {image_folder}")

    if not os.path.exists(image_folder):
        print(f"❌ 文件夹不存在: {image_folder}")
        return

    # 获取第一张图片作为测试
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    if not image_files:
        print("❌ 没有找到JPG文件")
        return

    first_image = os.path.join(image_folder, image_files[0])
    print(f"🖼️ 使用测试图片: {image_files[0]}")

    # 检查图片是否可以打开
    try:
        with Image.open(first_image) as img:
            width, height = img.size
            print(f"📐 图片尺寸: {width}x{height}")
    except Exception as e:
        print(f"❌ 无法打开图片: {e}")
        return

    print("\n📏 测试结果:")
    print("由于没有OpenCV，建议手动测试以下尺寸:")

    for size in test_sizes:
        print(f"\n🔹 棋盘格尺寸 {size[0]}x{size[1]}:")
        corners = (size[0] - 1) * (size[1] - 1)  # 内角点数量
        print(f"   • 内角点数量: {corners}")
        print(f"   • 适用于: 标准相机标定")

    print("\n💡 使用建议:")
    print("1. 在GUI中修改棋盘格尺寸设置")
    print("2. 重新运行Ground Calibration")
    print("3. 检查图片中是否真的有棋盘格图案")
    print("4. 如果都没有找到，可能是图片质量或角度问题")

if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"

    try:
        test_chessboard_sizes(image_folder)
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
