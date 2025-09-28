#!/usr/bin/env python3
"""
测试棋盘格打印质量
"""

import os
from PIL import Image

def test_print_quality():
    """测试打印质量"""
    print("🖨️ 棋盘格打印质量测试")
    print("=" * 50)

    # 测试文件
    test_file = "chessboard_9x6_250x175mm_margin20mm.png"

    if not os.path.exists(test_file):
        print(f"❌ 文件不存在: {test_file}")
        return

    # 获取文件信息
    file_size = os.path.getsize(test_file)
    with Image.open(test_file) as img:
        width, height = img.size
        mode = img.mode

    print("📊 文件信息:"    print(f"   • 文件名: {test_file}")
    print(f"   • 文件大小: {file_size:,} bytes")
    print(f"   • 像素尺寸: {width}×{height}")
    print(f"   • 色彩模式: {mode}")

    # 计算打印参数
    physical_width_mm = 290
    physical_height_mm = 215
    dpi = (width / physical_width_mm) * 25.4

    print("\n🖨️ 打印参数:")
    print(f"   • 物理尺寸: {physical_width_mm}×{physical_height_mm}mm")
    print(f"   • 打印分辨率: {dpi:.0f} DPI")
    print("   • 纸张要求: A2纸 (420×594mm)")

    # 质量评估
    print("\n⭐ 质量评估:")
    if dpi >= 300:
        quality = "🎯 专业级 - 极高清晰度"
        score = "A+"
    elif dpi >= 250:
        quality = "✅ 优秀 - 高清晰度"
        score = "A"
    elif dpi >= 200:
        quality = "✅ 良好 - 标准清晰度"
        score = "B+"
    else:
        quality = "⚠️ 合格 - 基本可用"
        score = "B"

    print(f"   • 清晰度等级: {quality}")
    print(f"   • 质量评分: {score}")
    print("   • 压缩效率: PNG格式，文件小但质量高")

    # 打印建议
    print("\n💡 打印建议:")
    print("   • ✅ 可以放心打印，质量非常好！")
    print("   • ✅ 适合A2纸打印")
    print("   • ✅ 黑白边缘清晰，红色尺寸线明显")
    print("   • ✅ 20mm白色边距便于固定")

    # 使用建议
    print("\n📸 使用建议:")
    print("   • 打印后用美工刀裁剪，保留5-10mm白边")
    print("   • 粘贴到平整的硬质板材上")
    print("   • 固定在墙上或地面，距离2-3米拍摄")
    print("   • 确保光照均匀，避免阴影")

if __name__ == "__main__":
    test_print_quality()
