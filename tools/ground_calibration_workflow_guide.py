#!/usr/bin/env python3
"""
Ground Calibration工作流程指南
解决相机高度不可用的完整解决方案
"""

import os
import sys
from pathlib import Path

class GroundCalibrationGuide:
    """Ground Calibration工作流程指南"""

    def __init__(self):
        self.current_dir = Path.cwd()
        self.calibration_files = self.find_calibration_files()

    def find_calibration_files(self):
        """查找现有的标定文件"""
        json_files = list(self.current_dir.glob("*.json"))
        camera_files = [f for f in json_files if any(keyword in f.name.lower()
                      for keyword in ['camera', 'calibration']) and 'ground' not in f.name.lower()]
        ground_files = [f for f in json_files if 'ground' in f.name.lower()]

        return {
            'camera': camera_files,
            'ground': ground_files
        }

    def show_current_status(self):
        """显示当前状态"""
        print("🔍 当前标定文件状态")
        print("=" * 50)

        camera_files = self.calibration_files['camera']
        ground_files = self.calibration_files['ground']

        print(f"📹 相机标定文件: {len(camera_files)} 个")
        for i, f in enumerate(camera_files, 1):
            print(f"   {i}. {f.name}")

        print(f"\n🌍 地面标定文件: {len(ground_files)} 个")
        for i, f in enumerate(ground_files, 1):
            print(f"   {i}. {f.name}")

        if not camera_files:
            print("\n❌ 没有找到相机标定文件!")
            print("💡 解决方案: 请先进行Camera Calibration")
        elif not ground_files:
            print("\n⚠️  没有找到地面标定文件")
            print("💡 解决方案: 请进行Ground Calibration")
        else:
            print("\n✅ 标定文件齐全")

    def show_solution_steps(self):
        """显示解决方案步骤"""
        print("\n" + "=" * 50)
        print("🛠️  解决方案步骤")
        print("=" * 50)

        steps = [
            {
                'step': 1,
                'title': '检查相机标定文件',
                'action': '查找现有的相机标定文件',
                'status': '✅' if self.calibration_files['camera'] else '❌'
            },
            {
                'step': 2,
                'title': '准备相机标定数据',
                'action': '如果没有标定文件，先进行Camera Calibration',
                'status': '⏳'
            },
            {
                'step': 3,
                'title': '加载相机标定数据',
                'action': '在Ground Calibration界面点击"Load Camera Calibration"',
                'status': '⏳'
            },
            {
                'step': 4,
                'title': '执行Ground Calibration',
                'action': '选择地面图片并运行标定',
                'status': '⏳'
            },
            {
                'step': 5,
                'title': '验证结果',
                'action': '检查相机高度是否正确显示',
                'status': '⏳'
            }
        ]

        for step_info in steps:
            status = step_info['status']
            print(f"\n{status} 步骤 {step_info['step']}: {step_info['title']}")
            print(f"   {step_info['action']}")

    def show_file_loading_guide(self):
        """显示文件加载指南"""
        print("\n" + "=" * 50)
        print("📂 文件加载指南")
        print("=" * 50)

        print("方法1: 使用GUI加载")
        print("1. 打开相机标定工具")
        print("2. 切换到Ground Calibration标签页")
        print("3. 点击'📂 Load Camera Calibration'按钮")
        print("4. 选择相机标定文件 (JSON/XML格式)")

        print("\n方法2: 对话框加载")
        print("1. 点击'Start Ground Calibration'")
        print("2. 当弹出'Camera calibration required'对话框时")
        print("3. 点击'Yes'加载文件，或'No'使用方法1")

        print("\n支持的文件格式:")
        print("• JSON文件 (*.json) - 推荐")
        print("• XML文件 (*.xml) - OpenCV兼容")
        print("• NPZ文件 (*.npz) - Python专用")

    def show_why_camera_calibration_needed(self):
        """解释为什么需要相机标定"""
        print("\n" + "=" * 50)
        print("🤔 为什么需要Camera Calibration?")
        print("=" * 50)

        reasons = [
            "相机内参确定: 焦距、畸变参数等",
            "3D重建基础: Ground Calibration依赖相机参数",
            "坐标转换: 像素坐标 → 世界坐标",
            "高度计算: 使用solvePnP计算相机姿态",
            "精度保证: 没有相机参数，计算结果不准确"
        ]

        for i, reason in enumerate(reasons, 1):
            print(f"{i}. {reason}")

        print("\n💡 技术细节:")
        print("• solvePnP需要: camera_matrix, dist_coeffs")
        print("• 相机高度 = tvec[2] (从solvePnP获得)")
        print("• 没有相机参数 → 无法计算准确高度")

    def show_quick_fix(self):
        """显示快速修复方法"""
        print("\n" + "=" * 50)
        print("🚀 快速修复方法")
        print("=" * 50)

        if self.calibration_files['camera']:
            print("✅ 找到相机标定文件!")
            print("\n立即修复步骤:")
            print("1. 运行相机标定工具")
            print("2. 切换到Ground Calibration标签页")
            print(f"3. 点击'📂 Load Camera Calibration'按钮")
            print(f"4. 选择文件: {self.calibration_files['camera'][0].name}")
            print("5. 点击'Start Ground Calibration'")

        else:
            print("❌ 没有找到相机标定文件")
            print("\n需要先进行Camera Calibration:")
            print("1. 切换到Camera Calibration标签页")
            print("2. 拍摄15-20张棋盘格标定图片")
            print("3. 点击'Start Camera Calibration'")
            print("4. 保存标定结果")
            print("5. 返回Ground Calibration继续")

    def create_step_by_step_guide(self):
        """创建详细的步骤指南"""
        print("\n" + "=" * 50)
        print("📋 详细步骤指南")
        print("=" * 50)

        guide = """
Ground Calibration完整流程:

第一阶段: Camera Calibration
=====================================
1. 打开相机标定工具
2. 切换到"Camera Calibration"标签页
3. 连接相机设备
4. 设置分辨率 (建议: 640x480 或 1280x720)
5. 拍摄标定图片:
   - 使用棋盘格 (建议尺寸: 9x6, 25mm方格)
   - 拍摄15-20张图片
   - 角度范围: 30-60度
   - 距离范围: 0.5-2米
6. 点击"Start Camera Calibration"
7. 等待标定完成
8. 保存标定结果 (选择JSON格式)

第二阶段: Ground Calibration
=====================================
1. 切换到"Ground Calibration"标签页
2. 准备地面标定图片:
   - 将棋盘格平放在地面上
   - 从不同角度拍摄 (建议6-10张)
   - 确保棋盘格完全可见
3. 点击"📂 Load Camera Calibration"按钮
4. 选择之前保存的相机标定文件
5. 确认状态显示"✅ Camera calibration loaded"
6. 设置Ground Calibration参数:
   - Chessboard size: 9x6 (与Camera Calibration一致)
   - Square size: 25.0 mm (精确测量值)
7. 点击"Start Ground Calibration"
8. 等待处理完成
9. 检查结果:
   - 相机高度应该正确显示
   - 重投影误差应该 < 1.0
10. 保存Ground Calibration结果

第三阶段: 验证和使用
=====================================
1. 点击"Validate Ground Calibration"
2. 检查各项指标是否正常
3. 如果高度不准确，重新检查:
   - 棋盘格尺寸测量
   - 相机标定质量
   - 拍摄角度和距离
"""

        print(guide)

def main():
    """主函数"""
    print("🎯 Ground Calibration相机高度问题解决方案")
    print("=" * 60)

    guide = GroundCalibrationGuide()

    # 显示当前状态
    guide.show_current_status()

    # 显示解决方案
    guide.show_solution_steps()

    # 显示文件加载指南
    guide.show_file_loading_guide()

    # 解释为什么需要相机标定
    guide.show_why_camera_calibration_needed()

    # 显示快速修复方法
    guide.show_quick_fix()

    # 创建详细指南
    guide.create_step_by_step_guide()

    print("\n" + "=" * 60)
    print("🎉 总结:")
    print("• 相机高度不可用是因为缺少Camera Calibration数据")
    print("• 按照上述步骤操作即可解决问题")
    print("• 建议使用JSON格式保存标定结果")
    print("• 如果仍有问题，请检查标定图片质量")

if __name__ == "__main__":
    main()
