#!/usr/bin/env python3
"""
相机高度计算分析工具
分析为什么计算出的高度与实际高度相差较大
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

class CameraHeightAnalyzer:
    """相机高度分析器"""

    def __init__(self):
        self.actual_height_cm = 117.0  # 用户提供的实际高度

    def analyze_height_issue(self):
        """分析高度计算问题的根本原因"""
        print("🔍 相机高度计算问题分析")
        print("=" * 60)
        print(f"📏 实际相机高度: {self.actual_height_cm} cm")
        print()

        # 分析可能的根本原因
        causes = self.identify_root_causes()
        solutions = self.generate_solutions()

        print("🔬 可能的原因分析:")
        for i, cause in enumerate(causes, 1):
            print(f"{i}. {cause}")

        print("\n💡 解决方案:")
        for i, solution in enumerate(solutions, 1):
            print(f"{i}. {solution}")

        print("\n📋 立即检查清单:")
        checklist = self.create_checklist()
        for item in checklist:
            print(f"□ {item}")

        return causes, solutions

    def identify_root_causes(self):
        """识别根本原因"""
        causes = []

        causes.append("⚠️  坐标系参考点问题: Ground Calibration使用棋盘格作为Z=0参考点")
        causes.append("⚠️  棋盘格尺寸测量误差: 如果棋盘格实际尺寸与设置不符，会导致比例尺错误")
        causes.append("⚠️  棋盘格放置位置偏差: 如果棋盘格没有完全平放在地面上")
        causes.append("⚠️  相机标定质量问题: 相机内参不准确会影响3D重建精度")
        causes.append("⚠️  拍摄角度和距离问题: 极端的拍摄角度会影响计算精度")
        causes.append("⚠️  标定图像质量问题: 模糊、畸变或光照不均会影响角点检测")

        return causes

    def generate_solutions(self):
        """生成解决方案"""
        solutions = []

        solutions.append("🎯 校准坐标系: 确保棋盘格完全平放在地面上，作为真正的Z=0参考点")
        solutions.append("📏 精确测量: 重新精确测量棋盘格方格尺寸，使用游标卡尺测量")
        solutions.append("📷 优化拍摄: 在合适距离和角度拍摄标定图片，避免极端角度")
        solutions.append("🔧 提高标定质量: 重新进行高质量的Camera Calibration")
        solutions.append("🖼️ 增加样本: 拍摄更多不同角度的标定图片，提高统计可靠性")
        solutions.append("⚙️ 调整参数: 根据实际情况调整Ground Calibration参数")

        return solutions

    def create_checklist(self):
        """创建检查清单"""
        checklist = [
            "棋盘格是否完全平放在地面上？",
            "棋盘格方格尺寸测量是否准确？(使用游标卡尺测量)",
            "Camera Calibration是否使用高质量图片？",
            "Ground Calibration图片是否清晰，没有模糊？",
            "拍摄角度是否合适？(建议30-60度)",
            "拍摄距离是否合适？(建议1-3米)",
            "光照条件是否良好？",
            "是否拍摄了足够多的图片？(建议6-10张)",
            "标定结果的reprojection error是否在合理范围内？(<1.0)",
            "是否尝试了不同的参数设置？"
        ]
        return checklist

    def analyze_calibration_files(self):
        """分析现有的标定文件"""
        print("\n📂 标定文件分析:")
        print("-" * 40)

        current_dir = Path.cwd()
        json_files = list(current_dir.glob("*.json"))

        ground_files = [f for f in json_files if 'ground' in f.name.lower()]
        camera_files = [f for f in json_files if 'ground' not in f.name.lower() and 'calibration' in f.name.lower()]

        print(f"相机标定文件: {len(camera_files)} 个")
        for f in camera_files:
            print(f"  • {f.name}")

        print(f"地面标定文件: {len(ground_files)} 个")
        for f in ground_files:
            print(f"  • {f.name}")

        return camera_files, ground_files

    def create_corrected_workflow(self):
        """创建更正的工作流程"""
        print("\n🔄 改进的工作流程建议:")
        print("-" * 40)

        workflow = [
            "1. 准备高质量的棋盘格",
            "   • 使用大尺寸棋盘格(建议边长50-100cm)",
            "   • 精确测量方格尺寸(使用游标卡尺)",
            "   • 确保棋盘格平整无变形",

            "2. 精确的Camera Calibration",
            "   • 拍摄15-20张不同角度的标定图片",
            "   • 确保图片清晰、光照均匀",
            "   • 角度范围: 30-60度",
            "   • 距离范围: 0.5-2米",

            "3. 准确的Ground Calibration",
            "   • 将棋盘格完全平放在地面上",
            "   • 确保地面平整",
            "   • 从多个角度拍摄(建议6-10张)",
            "   • 保持相机高度相对稳定",

            "4. 验证和调整",
            "   • 检查计算出的高度是否合理",
            "   • 如果偏差大，重新检查上述步骤",
            "   • 考虑环境因素的影响"
        ]

        for step in workflow:
            print(f"   {step}")

    def estimate_height_accuracy(self, square_size_mm, camera_height_cm=None):
        """估算高度计算的理论精度"""
        print("\n📐 高度计算精度分析:")
        print("-" * 40)

        if camera_height_cm is None:
            camera_height_cm = self.actual_height_cm

        # 基于棋盘格尺寸估算精度
        pixel_accuracy = 0.5  # 像素级精度假设
        focal_length_mm = 4.0  # 典型手机相机焦距，毫米

        # 理论高度精度计算
        height_accuracy_mm = (camera_height_cm * 10 * pixel_accuracy) / focal_length_mm
        height_accuracy_cm = height_accuracy_mm / 10

        print(".2f"        print(".2f"        print(".2f"
        if square_size_mm < 20:
            print("⚠️  警告: 棋盘格尺寸过小可能影响精度")
        elif square_size_mm > 50:
            print("✅ 建议: 当前棋盘格尺寸适中")

        return height_accuracy_cm

def main():
    """主函数"""
    print("🎯 相机高度偏差问题诊断工具")
    print("=" * 70)

    analyzer = CameraHeightAnalyzer()

    # 分析问题原因
    causes, solutions = analyzer.analyze_height_issue()

    # 分析现有文件
    analyzer.analyze_calibration_files()

    # 创建改进工作流程
    analyzer.create_corrected_workflow()

    # 估算理论精度
    print("\n" + "=" * 70)
    print("📊 理论精度估算:")

    # 基于不同棋盘格尺寸的精度估算
    sizes = [20, 25, 30, 50]  # 毫米
    for size in sizes:
        accuracy = analyzer.estimate_height_accuracy(size)
        print("2d"
    print("\n💡 建议使用25-30mm的棋盘格方格尺寸以获得最佳精度")

    # 输出总结
    print("\n" + "=" * 70)
    print("🎯 问题总结:")
    print(f"• 实际高度: {analyzer.actual_height_cm} cm")
    print("• 主要问题: 坐标系参考点和测量精度")
    print("• 关键改进: 精确的棋盘格尺寸和放置位置")
    print("• 预期结果: 通过改进可将误差控制在10-20cm以内")

    print("\n✅ 请按照上述建议重新进行标定测试")

if __name__ == "__main__":
    main()
