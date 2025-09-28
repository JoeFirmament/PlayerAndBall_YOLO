#!/usr/bin/env python3
"""
图像质量诊断工具
专门用于分析Ground Calibration图片质量问题
"""

import os
import sys
from PIL import Image
import numpy as np

def analyze_image_quality(image_path):
    """
    详细分析图像质量
    """
    print(f"\n🔍 分析图像: {os.path.basename(image_path)}")
    print("=" * 60)

    try:
        # 读取图像
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode

            print("📊 基本信息:"            print(f"   • 尺寸: {width}x{height}")
            print(f"   • 模式: {mode}")
            print(f"   • 文件大小: {os.path.getsize(image_path)} bytes")
            print(f"   • 像素总数: {width * height:,}")

            # 转换为numpy数组进行详细分析
            if mode == 'RGB':
                img_array = np.array(img)
                gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])  # RGB to grayscale
            else:
                gray = np.array(img.convert('L'))

            # 基本统计
            print("
📈 像素统计:"            print(f"   • 最小值: {gray.min()}")
            print(f"   • 最大值: {gray.max()}")
            print(f"   • 平均值: {gray.mean():.2f}")
            print(f"   • 标准差: {gray.std():.2f}")
            print(f"   • 中位数: {np.median(gray):.2f}")

            # 直方图分析
            hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 255))
            print("
📊 亮度分布:"            dark_pixels = np.sum(hist[:64])  # 0-63
            bright_pixels = np.sum(hist[192:])  # 192-255
            mid_pixels = np.sum(hist[64:192])  # 64-191

            total_pixels = width * height
            print(f"   • 暗像素 (0-63): {dark_pixels:,} ({dark_pixels/total_pixels*100:.1f}%)")
            print(f"   • 中等像素 (64-191): {mid_pixels:,} ({mid_pixels/total_pixels*100:.1f}%)")
            print(f"   • 亮像素 (192-255): {bright_pixels:,} ({bright_pixels/total_pixels*100:.1f}%)")

            # 对比度分析
            contrast = gray.std()
            if contrast < 20:
                contrast_level = "❌ 极低 (图像模糊或曝光不足)"
            elif contrast < 40:
                contrast_level = "⚠️ 低 (可能检测困难)"
            elif contrast < 70:
                contrast_level = "✅ 中等 (基本可检测)"
            else:
                contrast_level = "✅ 高 (检测容易)"

            print(f"   • 对比度水平: {contrast_level}")

            # 检测潜在问题
            print("
🔍 潜在问题分析:"            issues = []

            # 检查是否可能是全黑或全白图像
            if gray.mean() < 20:
                issues.append("• 图像过暗，可能曝光不足")
            elif gray.mean() > 230:
                issues.append("• 图像过亮，可能曝光过度")

            # 检查对比度
            if contrast < 30:
                issues.append("• 对比度太低，难以检测棋盘格")

            # 检查是否有明显的棋盘格模式
            # 计算水平和垂直方向的变化
            horizontal_diff = np.abs(np.diff(gray, axis=1)).mean()
            vertical_diff = np.abs(np.diff(gray, axis=0)).mean()

            print(f"   • 水平变化: {horizontal_diff:.2f}")
            print(f"   • 垂直变化: {vertical_diff:.2f}")

            if horizontal_diff < 5 or vertical_diff < 5:
                issues.append("• 图像缺乏明显的模式变化，可能不是棋盘格")

            # 分析图像的结构
            print("
🏗️ 结构分析:"            # 计算局部方差来检测纹理
            from scipy import ndimage
            try:
                local_var = ndimage.generic_filter(gray.astype(float), np.var, size=15)
                texture_level = local_var.mean()

                if texture_level < 10:
                    issues.append("• 纹理过于平滑，缺乏细节")
                elif texture_level > 500:
                    issues.append("• 纹理过于复杂，可能有噪点")

                print(f"   • 纹理复杂度: {texture_level:.1f}")

            except ImportError:
                print("   • 纹理分析跳过 (需要scipy)")

            # 显示问题总结
            if issues:
                print("
⚠️ 发现的问题:"                for issue in issues:
                    print(f"   {issue}")
            else:
                print("
✅ 未发现明显问题"                print("   如果仍然检测失败，可能需要:")
                print("   • 检查棋盘格尺寸是否正确")
                print("   • 确认棋盘格完全在画面内")
                print("   • 调整拍摄角度")

            return True

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return False

def diagnose_ground_calibration_folder(folder_path):
    """
    诊断整个Ground Calibration图片文件夹
    """
    print("🔬 GROUND CALIBRATION 图像质量诊断")
    print("=" * 80)
    print(f"📂 文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 获取所有jpg文件
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    print(f"📄 找到 {len(image_files)} 个JPG文件")

    if len(image_files) == 0:
        print("❌ 没有找到任何JPG文件")
        return

    # 分析前几个文件
    success_count = 0
    total_analyzed = min(5, len(image_files))

    print(f"\n📊 分析前 {total_analyzed} 个文件:")
    print("-" * 80)

    for i, filename in enumerate(image_files[:total_analyzed], 1):
        image_path = os.path.join(folder_path, filename)
        if analyze_image_quality(image_path):
            success_count += 1

    print("
📈 诊断总结:"    print(f"• 成功分析: {success_count}/{total_analyzed}")
    print(".1f"
    # 总体建议
    print("
💡 针对广角摄像头的建议:"    print("1. 📏 棋盘格尺寸:")
    print("   • 尝试更小的尺寸: 7x5, 6x4, 5x4")
    print("   • 确保棋盘格占画面比例合适 (20-60%)")

    print("
2. 📷 拍摄建议:"    print("   • 增加光照，改善对比度")
    print("   • 避免极端角度拍摄")
    print("   • 确保棋盘格完全在画面内")
    print("   • 保持相机稳定，避免抖动")

    print("
3. 🔧 技术调整:"    print("   • 尝试不同的检测参数")
    print("   • 考虑预处理图像 (去噪、增强对比度)")
    print("   • 使用更高质量的摄像头")

    print("
4. 🛠️ 故障排除:"    print("   • 检查图片是否真的包含棋盘格")
    print("   • 确认文件没有损坏")
    print("   • 尝试重新拍摄图片")

def main():
    """主函数"""
    folder_path = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"
    diagnose_ground_calibration_folder(folder_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        print("请检查Python环境和依赖是否正确安装")
