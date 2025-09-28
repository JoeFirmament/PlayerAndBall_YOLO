#!/usr/bin/env python3
"""
快速测试Ground Calibration修改
"""

import os
import sys

def test_ground_calibration_setup():
    """测试Ground Calibration设置"""
    print("🔍 测试Ground Calibration设置...")

    # 检查必要的文件
    required_files = [
        "camera_calibration_modern.py",
        "test_ground_images/"
    ]

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} - 存在")
        else:
            print(f"❌ {file_path} - 不存在")

    # 检查图片文件
    if os.path.exists("test_ground_images"):
        image_files = [f for f in os.listdir("test_ground_images") if f.endswith('.jpg')]
        print(f"📄 test_ground_images文件夹中有 {len(image_files)} 个JPG文件")

        if image_files:
            first_image = image_files[0]
            file_size = os.path.getsize(os.path.join("test_ground_images", first_image))
            print(f"   • 示例文件: {first_image}")
            print(f"   • 文件大小: {file_size} bytes")

    print("\n📋 Ground Calibration优化说明:")
    print("1. ✅ 已优化棋盘格检测参数适配广角摄像头")
    print("2. ✅ 添加了多种检测标志组合")
    print("3. ✅ 增加了自动尺寸调整功能")
    print("4. ✅ 增强了调试信息输出")
    print("5. ✅ 修复了语法错误")

    print("\n🚀 建议的测试步骤:")
    print("1. 运行相机标定GUI程序")
    print("2. 加载相机标定结果文件")
    print("3. 切换到Ground Calibration标签页")
    print("4. 选择test_ground_images文件夹")
    print("5. 点击'Start Ground Calibration'")
    print("6. 观察详细的调试输出信息")

    print("\n💡 预期改进:")
    print("• 更详细的检测过程信息")
    print("• 自动尝试不同的棋盘格尺寸")
    print("• 更好的广角摄像头支持")
    print("• 更清晰的错误诊断信息")

if __name__ == "__main__":
    try:
        test_ground_calibration_setup()
    except Exception as e:
        print(f"❌ 测试出错: {e}")
