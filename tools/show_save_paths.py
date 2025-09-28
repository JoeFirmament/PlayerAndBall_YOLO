#!/usr/bin/env python3
"""
显示相机标定工具的保存路径信息
"""

import os
import sys

def show_save_paths():
    """显示保存路径信息"""
    print("📁 相机标定工具保存路径信息")
    print("=" * 50)

    # 当前脚本位置
    current_script = os.path.abspath(__file__)
    print(f"当前脚本位置: {current_script}")

    # tools目录
    tools_dir = os.path.dirname(current_script)
    print(f"tools目录: {tools_dir}")

    # 项目根目录
    project_root = os.path.dirname(tools_dir)
    print(f"项目根目录: {project_root}")

    # 默认保存目录（当前工具目录）
    default_save_dir = tools_dir
    print(f"默认保存目录: {default_save_dir}")

    # 检查默认目录是否存在
    if os.path.exists(default_save_dir):
        print(f"✅ 默认保存目录存在")
        print(f"   目录内容:")
        try:
            files = os.listdir(default_save_dir)
            if files:
                for file in sorted(files):
                    file_path = os.path.join(default_save_dir, file)
                    size = os.path.getsize(file_path) / 1024
                    print(f"   • {file} ({size:.1f} KB)")
            else:
                print("   (空目录)")
        except Exception as e:
            print(f"   读取目录失败: {e}")
    else:
        print(f"❌ 默认保存目录不存在")
        print("   运行标定工具时会自动创建")

    print("\n💡 保存路径说明:")
    print("• 默认保存到当前工具目录 (tools/)")
    print("• 文件名格式: YYYYMMDD_HHMMSS_calibration.{ext}")
    print("• 支持格式: JSON (推荐), XML (OpenCV兼容), NPZ (Python专用)")

    print("\n📂 完整路径示例:")
    example_filename = "20250829_151327_calibration.json"
    print(f"• JSON: {os.path.join(default_save_dir, example_filename)}")
    example_filename = "20250829_151327_calibration.xml"
    print(f"• XML:  {os.path.join(default_save_dir, example_filename)}")
    example_filename = "20250829_151327_calibration.npz"
    print(f"• NPZ:  {os.path.join(default_save_dir, example_filename)}")

if __name__ == "__main__":
    show_save_paths()
