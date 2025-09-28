#!/usr/bin/env python3
"""
简单的图像质量检查脚本
不需要额外的依赖包
"""

import os
from PIL import Image

def simple_image_analysis(image_path):
    """
    简单的图像分析
    """
    filename = os.path.basename(image_path)
    print(f"\n🖼️ 分析: {filename}")

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            file_size = os.path.getsize(image_path)

            print(f"   📐 尺寸: {width}x{height}")
            print(f"   🎨 模式: {mode}")
            print(f"   📁 大小: {file_size:,} bytes")

            # 转换为灰度并分析像素
            gray_img = img.convert('L')
            pixels = list(gray_img.getdata())

            if not pixels:
                print("   ❌ 图像为空")
                return False

            # 基本统计
            min_val = min(pixels)
            max_val = max(pixels)
            avg_val = sum(pixels) / len(pixels)

            print(f"   💡 亮度范围: {min_val} - {max_val}")
            print(f"   📊 平均亮度: {avg_val:.1f}")

            # 对比度评估
            contrast = max_val - min_val
            if contrast < 30:
                contrast_status = "❌ 极低 - 几乎没有对比度"
            elif contrast < 60:
                contrast_status = "⚠️ 低 - 对比度不足"
            elif contrast < 120:
                contrast_status = "✅ 中等 - 基本可用"
            else:
                contrast_status = "✅ 高 - 对比度良好"

            print(f"   🔆 对比度: {contrast} ({contrast_status})")

            # 曝光评估
            if avg_val < 50:
                exposure = "❌ 过暗"
            elif avg_val < 100:
                exposure = "⚠️ 偏暗"
            elif avg_val < 150:
                exposure = "✅ 正常"
            elif avg_val < 200:
                exposure = "⚠️ 偏亮"
            else:
                exposure = "❌ 过亮"

            print(f"   📷 曝光: {exposure}")

            # 简单的纹理检测
            # 计算相邻像素的差异
            differences = []
            img_array = list(gray_img.getdata())
            width, height = gray_img.size

            for y in range(height - 1):
                for x in range(width - 1):
                    current = img_array[y * width + x]
                    right = img_array[y * width + x + 1]
                    down = img_array[(y + 1) * width + x]
                    differences.extend([abs(current - right), abs(current - down)])

            if differences:
                avg_diff = sum(differences) / len(differences)
                if avg_diff < 3:
                    texture = "❌ 极低 - 图像过于平滑"
                elif avg_diff < 10:
                    texture = "⚠️ 低 - 缺乏细节"
                elif avg_diff < 20:
                    texture = "✅ 中等 - 有一定纹理"
                else:
                    texture = "✅ 高 - 纹理丰富"

                print(f"   🏗️ 纹理: {avg_diff:.1f} ({texture})")

            # 问题诊断
            issues = []

            if contrast < 40:
                issues.append("对比度不足")

            if avg_val < 60 or avg_val > 200:
                issues.append("曝光问题")

            if avg_diff < 5:
                issues.append("缺乏纹理/细节")

            if issues:
                print(f"   ⚠️ 发现问题: {', '.join(issues)}")
            else:
                print("   ✅ 图像质量看起来不错")

            return True

    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return False

def check_ground_images_quality(folder_path):
    """
    检查Ground Calibration图片质量
    """
    print("🔍 GROUND CALIBRATION 图像质量检查")
    print("=" * 70)
    print(f"📂 检查文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    print(f"📄 找到 {len(image_files)} 个JPG文件")

    if not image_files:
        print("❌ 没有找到任何JPG文件")
        return

    # 检查前几个文件
    check_count = min(5, len(image_files))
    success_count = 0

    print(f"\n📊 检查前 {check_count} 个文件:")
    print("-" * 70)

    for filename in image_files[:check_count]:
        image_path = os.path.join(folder_path, filename)
        if simple_image_analysis(image_path):
            success_count += 1

    print("\n📈 检查结果:")
    print(f"• 成功分析: {success_count}/{check_count}")
    print(".1f")
    # 基于结果给出建议
    print("\n💡 诊断建议:")
    if success_count == 0:
        print("❌ 所有文件都无法分析")
        print("   • 检查文件是否损坏")
        print("   • 确认文件路径是否正确")
        print("   • 尝试重新生成图片")
    else:
        print("✅ 文件基本正常")
        print("   如果棋盘格检测仍然失败:")
        print("   • 检查图片是否真的包含棋盘格")
        print("   • 尝试不同的棋盘格尺寸")
        print("   • 改善拍摄条件")

    print("\n🔧 广角摄像头特殊建议:")
    print("• 使用更小的棋盘格尺寸 (6x4, 7x5)")
    print("• 增加光照，提高对比度")
    print("• 确保棋盘格占画面比例适中")
    print("• 避免极端拍摄角度")

if __name__ == "__main__":
    folder_path = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"
    check_ground_images_quality(folder_path)