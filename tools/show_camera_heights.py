#!/usr/bin/env python3
"""
简单显示相机标定文件中的高度信息
"""

import json
import os

def show_heights_simple():
    """简单显示高度信息"""
    print("🎯 相机标定文件高度信息")
    print("=" * 50)

    # 检查生成的标定文件
    json_file = "20250829_153422_calibration.json"

    if not os.path.exists(json_file):
        print(f"❌ 找不到标定文件: {json_file}")
        return

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tvecs = data.get('tvecs', [])

        if not tvecs:
            print("❌ 未找到相机位置信息")
            return

        print(f"📊 标定位置数量: {len(tvecs)}")
        print("\n🎯 相机高度信息 (Z坐标):")
        print("-" * 40)

        heights = []
        for i, tvec in enumerate(tvecs[:10]):  # 只显示前10个
            if len(tvec) >= 3:
                height = tvec[2][0] if isinstance(tvec[2], list) else tvec[2]
                heights.append(height)
                print(f"位置 {i+1:2d}: {height:8.2f} mm")

        if len(tvecs) > 10:
            print(f"... 还有 {len(tvecs) - 10} 个位置")

        print("
📈 统计:"        if heights:
            print(f"  • 平均高度: {sum(heights)/len(heights):.2f} mm")
            print(f"  • 最大高度: {max(heights):.2f} mm")
            print(f"  • 最小高度: {min(heights):.2f} mm")

    except Exception as e:
        print(f"❌ 读取文件失败: {e}")

if __name__ == "__main__":
    show_heights_simple()
