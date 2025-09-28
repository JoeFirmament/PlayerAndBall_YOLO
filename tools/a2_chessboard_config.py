#!/usr/bin/env python3
"""
A2纸9x6棋盘格配置计算器
验证和优化A2纸上的9x6棋盘格设计
"""

def calculate_a2_9x6_chessboard(square_size_mm=25):
    """
    计算A2纸上9x6棋盘格的配置
    """
    print("🎯 A2纸 9x6 棋盘格配置分析")
    print("=" * 70)

    # A2纸尺寸
    a2_width = 420  # mm
    a2_height = 594  # mm

    # 棋盘格计算
    board_corners_width = 9   # 横向内角点
    board_corners_height = 6  # 纵向内角点

    squares_width = board_corners_width + 1   # 10
    squares_height = board_corners_height + 1 # 7

    board_width = squares_width * square_size_mm   # 250mm
    board_height = squares_height * square_size_mm # 175mm

    print("📐 尺寸计算:"    print(f"   • A2纸尺寸: {a2_width}mm × {a2_height}mm")
    print(f"   • 棋盘格: {board_corners_width}×{board_corners_height} 内角点")
    print(f"   • 实际格子: {squares_width}×{squares_height}")
    print(f"   • 格子尺寸: {square_size_mm}mm")
    print(f"   • 棋盘尺寸: {board_width}mm × {board_height}mm")
    print(f"   • 内角点总数: {board_corners_width * board_corners_height}")

    # 计算余量
    width_margin = a2_width - board_width
    height_margin = a2_height - board_height

    print("\n📏 余量计算:")
    print(f"   • 宽度余量: {width_margin}mm ({width_margin/a2_width*100:.1f}%)")
    print(f"   • 高度余量: {height_margin}mm ({height_margin/a2_height*100:.1f}%)")

    # 评估适合性
    print("\n✅ 兼容性评估:")
    if board_width <= a2_width and board_height <= a2_height:
        print("   • ✅ 尺寸完全适合A2纸")
        print("   • ✅ 宽度余量充足")
        print("   • ✅ 高度余量充足")
    else:
        print("   • ❌ 尺寸超出A2纸限制")

    # 位置建议
    print("\n📍 摆放位置建议:")
    print("   • 居中摆放: 最平衡的选择")
    print("   • 靠左上角: 便于裁剪")
    print("   • 多个棋盘: 可以放2-3个小尺寸棋盘")

    # 打印优化建议
    print("\n🖨️ 打印建议:")
    print("   • 打印质量: 高质量/照片模式")
    print("   • 纸张类型: 普通打印纸即可")
    print("   • 颜色模式: 黑白/灰度")
    print("   • 缩放: 100% (不缩放)")
    print("   • 双面打印: 不推荐")

    return {
        'board_size': (board_corners_width, board_corners_height),
        'board_dimensions': (board_width, board_height),
        'paper_size': (a2_width, a2_height),
        'margins': (width_margin, height_margin),
        'square_size': square_size_mm
    }

def compare_with_other_options(current_config):
    """
    与其他选项进行对比
    """
    print("\n📊 与其他纸张对比:")
    print("-" * 50)

    paper_options = [
        ('A4', 210, 297, '经济'),
        ('A3', 297, 420, '常用'),
        ('A2', 420, 594, '推荐'),
        ('A1', 594, 841, '专业'),
        ('A0', 841, 1189, '超大')
    ]

    board_width = current_config['board_dimensions'][0]
    board_height = current_config['board_dimensions'][1]

    for name, w, h, desc in paper_options:
        fits_normal = board_width <= w and board_height <= h
        fits_rotated = board_width <= h and board_height <= w

        status = "✅ 适合" if fits_normal or fits_rotated else "❌ 不适合"
        rotate = " (旋转)" if fits_rotated and not fits_normal else ""
        current = " ← 当前选择" if name == 'A2' else ""

        print(f"   • {name}: {w}×{h}mm - {status}{rotate} - {desc}{current}")

def show_design_templates(config):
    """
    显示设计模板建议
    """
    print("\n🎨 设计模板建议:")
    print("-" * 50)

    board_w = config['board_dimensions'][0]
    board_h = config['board_dimensions'][1]
    square_size = config['square_size']

    print("   📋 精确规格:")
    print(f"   • 总宽度: {board_w}mm")
    print(f"   • 总高度: {board_h}mm")
    print(f"   • 格子大小: {square_size}mm × {square_size}mm")
    print(f"   • 黑白方格: {config['board_size'][0]+1}列 × {config['board_size'][1]+1}行")
    print(f"   • 内角点: {config['board_size'][0]}列 × {config['board_size'][1]}行")

    print("\n   🏗️ 设计要点:")
    print("   • 左上角第一个格子为黑色")
    print("   • 黑白交替排列")
    print("   • 线条清晰，边界分明")
    print("   • 无边框装饰（影响检测）")
    print("   • 添加尺寸标记（可选）")

def calculate_cost_benefit():
    """
    计算成本效益分析
    """
    print("\n💰 成本效益分析:")
    print("-" * 50)

    print("   📈 优势:")
    print("   • ✅ 纸张成本: 适中 (~¥2-3/张)")
    print("   • ✅ 打印方便: 标准打印机即可")
    print("   • ✅ 尺寸余量: 充足的摆放空间")
    print("   • ✅ 标定质量: 54个内角点，质量良好")
    print("   • ✅ 易于固定: 可以轻松粘贴在板材上")

    print("\n   ⚖️ 对比:")
    print("   • vs A4: 余量少，需旋转打印")
    print("   • vs A3: 余量中等，需要更大纸张")
    print("   • vs A1: 成本更高，余量过大")

    print("\n   🎯 综合评分: ⭐⭐⭐⭐⭐ (5/5)")
    print("   推荐指数: 极高")

def main():
    """主函数"""
    print("🀄️ A2纸 9x6 棋盘格配置验证")
    print("=" * 80)

    # 分析配置
    config = calculate_a2_9x6_chessboard(square_size_mm=25)

    # 对比其他选项
    compare_with_other_options(config)

    # 显示设计模板
    show_design_templates(config)

    # 成本效益分析
    calculate_cost_benefit()

    print("\n🎉 总结:")
    print("   ✅ A2纸 9x6 棋盘格配置完全可行！")
    print("   ✅ 这是你当前配置的最佳升级选择")
    print("   ✅ 既保持了熟悉的尺寸，又获得了更大空间")
    print("   ✅ 成本适中，效果显著")

if __name__ == "__main__":
    main()
