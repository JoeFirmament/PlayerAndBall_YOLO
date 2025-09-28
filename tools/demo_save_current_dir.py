#!/usr/bin/env python3
"""
演示默认保存到当前目录的功能
"""

import os
from datetime import datetime

def demo_save_paths():
    """演示保存路径计算"""
    print("🎯 演示默认保存到当前目录")
    print("=" * 40)

    # 模拟camera_calibration_modern.py中的路径计算
    script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_path)

    print(f"当前脚本位置: {script_path}")
    print(f"计算出的当前目录: {current_dir}")
    print(f"实际工作目录: {os.getcwd()}")
    print(f"路径匹配: {current_dir == os.getcwd()}")

    # 模拟生成保存文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"{timestamp}_calibration.json"
    xml_filename = f"{timestamp}_calibration.xml"
    npz_filename = f"{timestamp}_calibration.npz"

    print(f"\n📄 生成的保存文件名:")
    print(f"• JSON: {json_filename}")
    print(f"• XML:  {xml_filename}")
    print(f"• NPZ:  {npz_filename}")

    print(f"\n📁 完整的保存路径:")
    print(f"• JSON: {os.path.join(current_dir, json_filename)}")
    print(f"• XML:  {os.path.join(current_dir, xml_filename)}")
    print(f"• NPZ:  {os.path.join(current_dir, npz_filename)}")

    print(f"\n✅ 现在标定结果将保存到当前工具目录中!")
    print(f"   方便查看和管理所有标定文件。")

if __name__ == "__main__":
    demo_save_paths()
