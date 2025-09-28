#!/usr/bin/env python3
"""
Ground Calibration图片处理诊断工具
帮助诊断为什么total images processed是0
"""

import os
import sys
import glob
from pathlib import Path
import json

class GroundCalibrationDiagnoser:
    """Ground Calibration诊断器"""

    def __init__(self):
        self.supported_extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.tiff']

    def diagnose_ground_calibration_issue(self):
        """诊断Ground Calibration图片处理问题"""
        print("🔍 Ground Calibration图片处理诊断")
        print("=" * 60)

        # 1. 检查是否有测试图片文件夹
        print("📂 步骤1: 检查图片文件夹")
        test_folders = self.find_potential_image_folders()

        if not test_folders:
            print("❌ 未找到包含图片的文件夹")
            print("💡 解决方案: 请创建包含标定图片的文件夹")
            self.show_folder_creation_guide()
            return

        print(f"✅ 找到 {len(test_folders)} 个可能的图片文件夹:")
        for i, folder in enumerate(test_folders, 1):
            print(f"   {i}. {folder}")

        # 2. 分析每个文件夹
        for folder in test_folders:
            print(f"\n🔍 分析文件夹: {folder}")
            self.analyze_folder_contents(folder)

        # 3. 显示解决方案
        self.show_solutions()

    def find_potential_image_folders(self):
        """查找可能的图片文件夹"""
        current_dir = Path.cwd()
        potential_folders = []

        # 检查当前目录的子文件夹
        for item in current_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 检查文件夹中是否有图片文件
                image_count = 0
                for ext in self.supported_extensions:
                    pattern = str(item / ext[1:])  # 移除*号
                    image_count += len(list(item.glob(ext[1:])))

                if image_count > 0:
                    potential_folders.append(str(item))

        # 检查当前目录是否有图片
        current_images = 0
        for ext in self.supported_extensions:
            current_images += len(list(current_dir.glob(ext[1:])))

        if current_images > 0:
            potential_folders.insert(0, str(current_dir))

        return potential_folders

    def analyze_folder_contents(self, folder_path):
        """分析文件夹内容"""
        folder = Path(folder_path)
        total_files = 0
        supported_files = []
        unsupported_files = []

        print(f"📊 文件夹内容分析: {folder.name}")
        print("-" * 40)

        # 统计各种文件类型
        for item in folder.iterdir():
            if item.is_file():
                total_files += 1
                if item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    supported_files.append(item.name)
                else:
                    unsupported_files.append(item.name)

        print(f"• 总文件数: {total_files}")
        print(f"• 支持的图片文件: {len(supported_files)}")
        print(f"• 不支持的文件: {len(unsupported_files)}")

        if supported_files:
            print(f"• 示例图片文件: {supported_files[:3]}")
            if len(supported_files) > 3:
                print(f"  ...还有{len(supported_files) - 3}个文件")
        else:
            print("❌ 没有找到支持的图片文件!")

        if unsupported_files:
            print(f"• 不支持的文件类型: {set(f.split('.')[-1] for f in unsupported_files[:5])}")

    def show_folder_creation_guide(self):
        """显示文件夹创建指南"""
        print("\n📁 创建标定图片文件夹的指南:")
        print("-" * 40)

        guide = """
Ground Calibration图片准备:

1. 创建文件夹:
   mkdir ground_calibration_images

2. 拍摄标定图片:
   • 将棋盘格平放在地面上
   • 从不同角度拍摄 (建议6-10张)
   • 确保棋盘格完全在画面中
   • 保持相机高度相对稳定

3. 支持的图片格式:
   • JPG/JPEG (*.jpg, *.jpeg)
   • PNG (*.png)
   • BMP (*.bmp)
   • TIFF (*.tiff)

4. 图片命名建议:
   • ground_cal_001.jpg
   • ground_cal_002.jpg
   • chessboard_01.jpg
   • chessboard_02.jpg

5. 图片质量要求:
   • 清晰度: 棋盘格角点清晰可见
   • 角度: 30-60度最佳
   • 距离: 1-3米
   • 光照: 均匀，避免强光
"""
        print(guide)

    def show_solutions(self):
        """显示解决方案"""
        print("\n💡 解决方案:")
        print("=" * 50)

        solutions = [
            "1. 选择正确的图片文件夹",
            "   • 使用'Select Folder'按钮选择包含图片的文件夹",
            "   • 确保文件夹中有支持格式的图片文件",

            "2. 检查图片文件格式",
            "   • 确保图片是 JPG/PNG/BMP/TIFF 格式",
            "   • 检查文件扩展名是否正确",

            "3. 验证文件夹路径",
            "   • 确保文件夹路径没有特殊字符",
            "   • 检查文件夹权限是否正确",

            "4. 检查图片质量",
            "   • 确保图片清晰，棋盘格角点可见",
            "   • 避免模糊、曝光过度或不足的图片",

            "5. 重新选择文件夹",
            "   • 如果当前文件夹有问题，选择其他文件夹",
            "   • 创建新的文件夹并放入标定图片"
        ]

        for solution in solutions:
            print(solution)
            print()

    def create_test_images_folder(self):
        """创建测试图片文件夹结构"""
        print("\n🔧 创建测试文件夹结构:")
        print("-" * 40)

        test_folder = "ground_calibration_test"
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
            print(f"✅ 创建测试文件夹: {test_folder}")

            # 创建说明文件
            readme_content = """
Ground Calibration Test Folder

请将您的标定图片放入此文件夹，然后在软件中选择此文件夹。

支持的图片格式:
- JPG/JPEG
- PNG
- BMP
- TIFF

图片要求:
- 包含清晰的棋盘格
- 棋盘格平放在地面上
- 从多个角度拍摄
"""
            with open(os.path.join(test_folder, "README.txt"), "w", encoding="utf-8") as f:
                f.write(readme_content)

            print("✅ 创建说明文件: README.txt")

        else:
            print(f"⚠️ 测试文件夹已存在: {test_folder}")

        print(f"\n💡 请将标定图片复制到: {os.path.abspath(test_folder)}")

    def show_step_by_step_guide(self):
        """显示逐步指南"""
        print("\n📋 Ground Calibration逐步指南:")
        print("=" * 50)

        steps = """
第一步: 准备图片文件夹
===========================
1. 创建文件夹: ground_calibration_images/
2. 将标定图片放入文件夹
3. 确保图片格式正确 (JPG/PNG)

第二步: 选择文件夹
===========================
1. 打开相机标定工具
2. 切换到"Ground Calibration"标签页
3. 点击"Select Folder"按钮
4. 选择包含图片的文件夹
5. 确认状态栏显示找到的图片数量

第三步: 加载相机标定数据
===========================
1. 点击"📂 Load Camera Calibration"按钮
2. 选择相机标定文件 (JSON格式)
3. 确认状态显示"✅ Camera calibration loaded"

第四步: 设置参数并运行
===========================
1. 设置棋盘格参数:
   - Chessboard size: 9x6
   - Square size: 25.0 mm
2. 点击"Start Ground Calibration"
3. 等待处理完成
4. 查看结果

第五步: 验证结果
===========================
1. 检查相机高度是否正确显示
2. 检查重投影误差 (< 1.0)
3. 保存标定结果
"""
        print(steps)

def main():
    """主函数"""
    print("🎯 Ground Calibration图片处理问题诊断")
    print("=" * 60)

    diagnoser = GroundCalibrationDiagnoser()

    # 执行诊断
    diagnoser.diagnose_ground_calibration_issue()

    # 显示详细指南
    diagnoser.show_step_by_step_guide()

    # 创建测试文件夹
    diagnoser.create_test_images_folder()

    print("\n" + "=" * 60)
    print("🎉 诊断完成!")
    print("请按照上述步骤检查和解决问题")
    print("如果仍有问题，请告诉我具体的情况")

if __name__ == "__main__":
    main()
