#!/usr/bin/env python3
"""
纹理分析演示脚本
解释纹理值2.8-3.2的含义
"""

import os
from PIL import Image

def demonstrate_texture_levels():
    """
    演示不同纹理水平的含义
    """
    print("🔍 图像纹理分析演示")
    print("=" * 60)

    # 模拟不同纹理水平的图像数据
    print("📊 纹理水平对比:")
    print("\n🖼️ 不同纹理水平示例:")
    print("-" * 60)

    # 极低纹理 (类似你的图片)
    print("❌ 极低纹理 (2.8-3.2):")
    print("   • 相邻像素差异极小")
    print("   • 图像看起来非常平滑")
    print("   • 缺乏细节和边缘")
    print("   • 视觉效果: 像雾化或过度模糊的照片")
    print("   • 检测影响: OpenCV无法找到明显的特征点")

    print("\n⚠️ 低纹理 (3.0-10.0):")
    print("   • 相邻像素差异较小")
    print("   • 图像较为平滑")
    print("   • 有少量细节")
    print("   • 视觉效果: 轻微模糊但仍可辨认")
    print("   • 检测影响: 检测困难，需要更好算法")

    print("\n✅ 中等纹理 (10.0-20.0):")
    print("   • 相邻像素差异适中")
    print("   • 图像有明显细节")
    print("   • 边缘清晰")
    print("   • 视觉效果: 正常照片")
    print("   • 检测影响: 检测相对容易")

    print("\n✅ 高纹理 (20.0+):")
    print("   • 相邻像素差异很大")
    print("   • 图像非常丰富")
    print("   • 很多细节和纹理")
    print("   • 视觉效果: 高清、锐利")
    print("   • 检测影响: 检测非常容易")

    print("\n📈 你的图像纹理值:")
    print("• 纹理值: 2.8-3.2")
    print("• 水平: 极低")
    print("• 含义: 图像过于平滑，几乎没有纹理变化")

def analyze_texture_in_detail(image_path):
    """
    详细分析图像纹理
    """
    print(f"\n🔬 详细纹理分析: {os.path.basename(image_path)}")
    print("-" * 60)

    try:
        with Image.open(image_path) as img:
            gray_img = img.convert('L')
            pixels = list(gray_img.getdata())
            width, height = gray_img.size

            print(f"📐 图像尺寸: {width}x{height}")

            # 计算相邻像素差异
            differences = []
            img_array = list(gray_img.getdata())

            print("🔄 计算相邻像素差异...")

            # 水平差异 (左右相邻像素)
            horizontal_diffs = []
            for y in range(height):
                for x in range(width - 1):
                    current = img_array[y * width + x]
                    right = img_array[y * width + x + 1]
                    diff = abs(current - right)
                    horizontal_diffs.append(diff)

            # 垂直差异 (上下相邻像素)
            vertical_diffs = []
            for y in range(height - 1):
                for x in range(width):
                    current = img_array[y * width + x]
                    down = img_array[(y + 1) * width + x]
                    diff = abs(current - down)
                    vertical_diffs.append(diff)

            # 统计分析
            avg_horizontal = sum(horizontal_diffs) / len(horizontal_diffs) if horizontal_diffs else 0
            avg_vertical = sum(vertical_diffs) / len(vertical_diffs) if vertical_diffs else 0
            avg_texture = (avg_horizontal + avg_vertical) / 2

            print("📊 纹理统计:")
            print(f"   • 水平纹理: {avg_horizontal:.2f} (左右像素差异)")
            print(f"   • 垂直纹理: {avg_vertical:.2f} (上下像素差异)")
            print(f"   • 平均纹理: {avg_texture:.2f} (整体纹理水平)")

            # 差异分布
            small_diffs = sum(1 for d in horizontal_diffs + vertical_diffs if d <= 5)
            medium_diffs = sum(1 for d in horizontal_diffs + vertical_diffs if 5 < d <= 20)
            large_diffs = sum(1 for d in horizontal_diffs + vertical_diffs if d > 20)

            total_diffs = len(horizontal_diffs) + len(vertical_diffs)

            print("\n📈 差异分布:")
            print(f"   • 小差异 (≤5): {small_diffs:,} ({small_diffs/total_diffs*100:.1f}%)")
            print(f"   • 中等差异 (6-20): {medium_diffs:,} ({medium_diffs/total_diffs*100:.1f}%)")
            print(f"   • 大差异 (>20): {large_diffs:,} ({large_diffs/total_diffs*100:.1f}%)")

            # 解释结果
            print("\n💡 分析结果:")
            if avg_texture < 3:
                print("❌ 纹理极低!")
                print("   这意味着图像中几乎没有亮度变化")
                print("   相邻像素的亮度差异平均只有2.8个单位")
                print("   相当于512级灰度中只有0.5%的变化范围")
            elif avg_texture < 10:
                print("⚠️ 纹理较低")
                print("   图像较为平滑，但仍有一些细节")
            else:
                print("✅ 纹理正常")
                print("   图像有足够的细节变化")

            # 具体解释为什么检测失败
            print("\n🎯 为什么棋盘格检测失败:")
            print("• OpenCV的findChessboardCorners函数依赖于角点检测")
            print("• 角点检测需要明显的亮度变化和边缘")
            print("• 纹理值2.8意味着几乎没有亮度梯度")
            print("• 没有梯度 = 没有边缘 = 无法检测棋盘格")

    except Exception as e:
        print(f"❌ 分析失败: {e}")

def main():
    """主函数"""
    demonstrate_texture_levels()

    # 分析实际图片
    image_path = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images/829_1705_20250829_170542.jpg"

    if os.path.exists(image_path):
        analyze_texture_in_detail(image_path)
    else:
        print(f"\n⚠️ 示例图片不存在: {image_path}")
        print("请确保图片文件存在后再运行分析")

if __name__ == "__main__":
    main()
