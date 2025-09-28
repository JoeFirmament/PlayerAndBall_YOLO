#!/usr/bin/env python3
"""
相机高度标定诊断工具
帮助分析为什么计算出的相机高度与实际高度相差较大
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HeightCalibrationDiagnoser:
    """相机高度标定诊断器"""

    def __init__(self):
        self.issues = []
        self.suggestions = []

    def diagnose_height_accuracy(self, ground_calibration_file=None):
        """诊断高度计算准确性"""
        print("🔍 相机高度标定诊断工具")
        print("=" * 60)

        if ground_calibration_file and os.path.exists(ground_calibration_file):
            print(f"📂 加载Ground Calibration文件: {ground_calibration_file}")

            try:
                with open(ground_calibration_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if 'camera_height_info' in data:
                    height_info = data['camera_height_info']
                    calculated_height = height_info['camera_height_cm']

                    print("📊 当前计算结果:")
                    print(".1f")
                    print(f"• 测量方法: {height_info['measurement_method']}")
                    print(f"• 参考坐标系: {height_info['reference_frame']}")

                    self.analyze_height_accuracy(calculated_height, data)
                else:
                    print("❌ Ground Calibration文件中没有相机高度信息")
                    self.issues.append("Ground Calibration文件中缺少相机高度信息")

            except Exception as e:
                print(f"❌ 加载文件失败: {e}")
                self.issues.append(f"文件加载失败: {e}")
        else:
            print("❌ 未提供Ground Calibration文件或文件不存在")
            self.issues.append("Ground Calibration文件缺失")

        self.generate_diagnosis_report()

    def analyze_height_accuracy(self, calculated_height, data):
        """分析高度计算的准确性"""
        print("\n🔬 准确性分析:")

        # 1. 检查棋盘格参数
        if 'board_params' in data:
            board_params = data['board_params']
            print("📏 棋盘格参数检查:")
            print(f"• 尺寸: {board_params.get('size', 'Unknown')}")
            print(f"• 方格大小: {board_params.get('square_size', 'Unknown')} mm")

            # 检查方格大小的合理性
            square_size = board_params.get('square_size', 0)
            if square_size < 10 or square_size > 100:
                self.issues.append(f"棋盘格方格大小异常: {square_size}mm (建议范围: 10-100mm)")
                self.suggestions.append("请重新测量并设置准确的棋盘格方格尺寸")
        else:
            self.issues.append("Ground Calibration数据中缺少棋盘格参数")
            self.suggestions.append("请检查Ground Calibration是否正确执行")

        # 2. 检查标定图像数量
        successful_images = data.get('successful_images', 0)
        total_images = data.get('total_images', 0)

        print(f"\n🖼️  标定图像统计:")
        print(f"• 成功检测: {successful_images}/{total_images}")
        print(".1f")
        if successful_images < 4:
            self.issues.append("标定图像数量不足 (建议至少4张)")
            self.suggestions.append("请拍摄更多不同角度的棋盘格图片")

        # 3. 检查重投影误差
        reprojection_error = data.get('reprojection_error', 0)
        print(f"\n📐 重投影误差: {reprojection_error:.4f} pixels")

        if reprojection_error > 1.0:
            self.issues.append(f"重投影误差过高: {reprojection_error:.4f} pixels")
            self.suggestions.append("重投影误差过高可能导致高度计算不准确")

        # 4. 分析高度偏差
        print(f"\n📏 高度分析:")
        actual_height = 117.0  # 用户提供的实际高度
        deviation = abs(calculated_height - actual_height)
        deviation_percent = (deviation / actual_height) * 100

        print(".1f")
        print(".1f"
        if deviation > 50:  # 偏差超过50cm
            self.issues.append(f"高度偏差过大: {deviation:.1f}cm ({deviation_percent:.1f}%)")
            self.suggestions.append("高度偏差过大，建议检查以下项目:")

        # 5. 生成具体建议
        if deviation > 50:
            self.generate_specific_suggestions(calculated_height, actual_height, data)

    def generate_specific_suggestions(self, calculated, actual, data):
        """生成具体的改进建议"""
        deviation = calculated - actual

        print("\n💡 具体改进建议:")

        if abs(deviation) > 100:
            print("🔴 严重偏差 (偏差>100cm):")
            print("   1. 检查棋盘格是否真正平放在地面上")
            print("   2. 验证棋盘格尺寸测量是否准确")
            print("   3. 重新进行Camera Calibration")
            print("   4. 尝试不同的拍摄角度和距离")
        elif abs(deviation) > 50:
            print("🟡 中等偏差 (偏差50-100cm):")
            print("   1. 增加标定图片的数量和多样性")
            print("   2. 确保相机标定质量良好")
            print("   3. 检查Ground Calibration的原点设置")
        else:
            print("🟢 轻微偏差 (偏差<50cm):")
            print("   1. 检查是否是测量误差")
            print("   2. 考虑环境因素影响")

        self.suggestions.append("请按照上述建议逐步排查和改进")

    def generate_diagnosis_report(self):
        """生成诊断报告"""
        print("\n📋 诊断报告")
        print("=" * 40)

        if self.issues:
            print("❌ 发现的问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ 未发现明显问题")

        if self.suggestions:
            print("\n💡 建议的改进措施:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"   {i}. {suggestion}")

        print("\n" + "=" * 40)
        print("🎯 快速检查清单:")
        print("□ 棋盘格尺寸测量准确吗？")
        print("□ 棋盘格完全平放在地面上吗？")
        print("□ Camera Calibration质量良好吗？")
        print("□ 拍摄了足够多的标定图片吗？")
        print("□ 拍摄角度和距离合适吗？")

def interactive_diagnosis():
    """交互式诊断"""
    print("🔍 相机高度标定交互式诊断")
    print("=" * 50)

    diagnoser = HeightCalibrationDiagnoser()

    # 查找Ground Calibration文件
    current_dir = Path.cwd()
    json_files = list(current_dir.glob("*ground*.json")) + list(current_dir.glob("*calibration*.json"))

    if json_files:
        print("📂 找到的标定文件:")
        for i, file in enumerate(json_files, 1):
            print(f"   {i}. {file.name}")

        try:
            choice = input("\n请选择文件编号 (1-{}): ".format(len(json_files)))
            if choice.isdigit() and 1 <= int(choice) <= len(json_files):
                selected_file = json_files[int(choice) - 1]
                diagnoser.diagnose_height_accuracy(str(selected_file))
            else:
                print("❌ 无效选择")
        except KeyboardInterrupt:
            print("\n👋 诊断已取消")
    else:
        print("❌ 未找到Ground Calibration文件")
        print("\n💡 请先运行Ground Calibration并保存结果")
        diagnoser.diagnose_height_accuracy()

def main():
    """主函数"""
    try:
        interactive_diagnosis()
    except Exception as e:
        print(f"❌ 诊断过程中出现错误: {e}")

if __name__ == "__main__":
    main()
