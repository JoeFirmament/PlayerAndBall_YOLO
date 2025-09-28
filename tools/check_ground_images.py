#!/usr/bin/env python3
"""
检查Ground Calibration图片的简单脚本
不需要复杂的依赖，只检查基本信息
"""

import os
import sys
from PIL import Image

def check_image_basic_info(image_path):
    """检查图像的基本信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            # 转换为灰度并计算基本统计信息
            gray_img = img.convert('L')
            pixels = list(gray_img.getdata())

            if len(pixels) == 0:
                return {'valid': False, 'error': 'Empty image'}

            mean_brightness = sum(pixels) / len(pixels)
            min_value = min(pixels)
            max_value = max(pixels)

            # 计算对比度（标准差的简化版本）
            variance = sum((x - mean_brightness) ** 2 for x in pixels) / len(pixels)
            contrast = variance ** 0.5

            return {
                'width': width,
                'height': height,
                'mode': mode,
                'mean_brightness': mean_brightness,
                'contrast': contrast,
                'min_value': min_value,
                'max_value': max_value,
                'valid': True
            }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def analyze_ground_images(folder_path):
    """分析ground calibration图片"""
    print("🔍 分析Ground Calibration图片...")
    print(f"📂 文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 获取所有jpg文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.endswith('.jpg') and os.path.isfile(os.path.join(folder_path, f))]

    print(f"📄 找到 {len(image_files)} 个JPG文件")

    if len(image_files) == 0:
        print("❌ 没有找到任何JPG文件")
        return

    # 检查前几个文件
    valid_images = 0
    total_images = min(5, len(image_files))  # 检查前5个文件

    print(f"\n📊 检查前 {total_images} 个文件:")

    for i, filename in enumerate(image_files[:total_images], 1):
        image_path = os.path.join(folder_path, filename)
        print(f"\n🖼️ 文件 {i}: {filename}")

        info = check_image_basic_info(image_path)

        if info['valid']:
            print("   ✅ 文件有效")
            print(f"   📐 尺寸: {info['width']}x{info['height']}")
            print(f"   🎨 模式: {info['mode']}")
            print(f"   💡 平均亮度: {info['mean_brightness']:.1f}")
            print(f"   🔆 对比度: {info['contrast']:.1f}")
            print(f"   📈 像素范围: {info['min_value']} - {info['max_value']}")
            valid_images += 1
        else:
            print(f"   ❌ 文件无效: {info['error']}")

    print("\n📊 总结:")
    print(f"• 检查的文件: {total_images}")
    print(f"• 有效的文件: {valid_images}")
    print(".1f")
    if len(image_files) > total_images:
        print(f"• 其余文件: {len(image_files) - total_images} 个")

    # 给出建议
    print("\n💡 分析结果:")
    if valid_images == 0:
        print("❌ 所有检查的文件都无效")
        print("   建议:")
        print("   1. 检查文件路径是否正确")
        print("   2. 确认文件是否损坏")
        print("   3. 尝试重新生成图片")
    elif valid_images < total_images:
        print("⚠️ 部分文件可能有问题")
        print("   建议检查文件完整性")
    else:
        print("✅ 文件基本检查通过")
        print("   如果棋盘格检测仍然失败，可能的原因:")
        print("   1. 图片中没有棋盘格图案")
        print("   2. 棋盘格尺寸设置不正确")
        print("   3. 图像质量不足以检测棋盘格")
        print("   4. 视角角度太大")

if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"

    try:
        analyze_ground_images(image_folder)
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        print("请检查Python环境和依赖是否正确安装")
