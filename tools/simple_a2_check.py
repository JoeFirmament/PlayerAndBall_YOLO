#!/usr/bin/env python3
"""
简化的A2纸9x6棋盘格检查
"""

def main():
    print("🎯 A2纸 9x6 棋盘格配置验证")
    print("=" * 50)

    # A2纸尺寸
    a2_width = 420  # mm
    a2_height = 594  # mm

    # 棋盘格参数
    board_corners_width = 9   # 横向内角点
    board_corners_height = 6  # 纵向内角点
    square_size = 25         # 格子尺寸mm

    # 计算总尺寸
    squares_width = board_corners_width + 1   # 10
    squares_height = board_corners_height + 1 # 7
    board_width = squares_width * square_size   # 250mm
    board_height = squares_height * square_size # 175mm

    print("📐 尺寸计算:")
    print(f"   • A2纸尺寸: {a2_width}mm × {a2_height}mm")
    print(f"   • 棋盘格: {board_corners_width}×{board_corners_height} 内角点")
    print(f"   • 格子尺寸: {square_size}mm")
    print(f"   • 棋盘尺寸: {board_width}mm × {board_height}mm")

    # 计算余量
    width_margin = a2_width - board_width
    height_margin = a2_height - board_height

    print("
📏 余量计算:"    print(f"   • 宽度余量: {width_margin}mm ({width_margin/a2_width*100:.1f}%)")
    print(f"   • 高度余量: {height_margin}mm ({height_margin/a2_height*100:.1f}%)")

    # 评估适合性
    print("
✅ 兼容性评估:"    if board_width <= a2_width and board_height <= a2_height:
        print("   • ✅ 尺寸完全适合A2纸")
        print("   • ✅ 宽度余量充足")
        print("   • ✅ 高度余量充足")
        print("   • ✅ 强烈推荐此配置！")
    else:
        print("   • ❌ 尺寸超出A2纸限制")

    print("
🎉 结论:"    print("   ✅ A2纸 9x6 棋盘格配置完全可行！")
    print(f"   ✅ 棋盘尺寸: {board_width}×{board_height}mm")
    print(f"   ✅ 纸张余量: {width_margin}×{height_margin}mm")
    print("   ✅ 这是你当前配置的最佳升级选择")

if __name__ == "__main__":
    main()
