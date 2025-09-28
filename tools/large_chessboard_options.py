#!/usr/bin/env python3
"""
大型棋盘格配置建议器
为用户提供更大的棋盘格选项
"""

def generate_large_chessboard_configs(square_size_mm=25):
    """
    生成大型棋盘格配置选项
    """
    print("🎯 大型棋盘格配置建议")
    print("=" * 70)
    print(f"📏 格子尺寸: {square_size_mm}mm (保持你喜欢的尺寸)")
    print()

    # 纸张尺寸
    paper_sizes = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A2': (420, 594),
        'A1': (594, 841),
        'Letter': (216, 279),
        'Legal': (216, 356),
        'Tabloid': (279, 432)
    }

    # 常见的棋盘格配置 (内角点)
    board_configs = [
        (9, 6),   # 你的当前配置
        (10, 7),  # 增加内角点
        (11, 8),  # 更大
        (12, 8),  # 最大化内角点
        (8, 8),   # 正方形
        (9, 8),   # 纵向更大
        (10, 6),  # 横向更大
    ]

    print("📋 配置选项 (按内角点数量排序):")
    print("-" * 70)

    options = []

    for board in board_configs:
        # 计算总尺寸
        width_squares = board[0] + 1  # 内角点数+1 = 格子数
        height_squares = board[1] + 1
        total_width = width_squares * square_size_mm
        total_height = height_squares * square_size_mm
        corners = board[0] * board[1]

        # 评估适合的纸张
        suitable_papers = []
        for paper_name, (p_w, p_h) in paper_sizes.items():
            # 考虑边距
            usable_width = p_w - 20
            usable_height = p_h - 20

            fits_normal = total_width <= usable_width and total_height <= usable_height
            fits_rotated = total_width <= usable_height and total_height <= usable_width

            if fits_normal or fits_rotated:
                rotation_note = " (旋转)" if fits_rotated and not fits_normal else ""
                suitable_papers.append(f"{paper_name}{rotation_note}")

        # 计算配置评分 (内角点数量 + 尺寸合理性)
        size_ratio = min(total_width / 300, total_height / 400)  # 相对于A3的比率
        score = corners * 0.7 + (1 - size_ratio) * 30  # 偏好更多角点但不超大

        options.append({
            'board': board,
            'size': (total_width, total_height),
            'corners': corners,
            'papers': suitable_papers,
            'score': score
        })

    # 按评分排序
    options.sort(key=lambda x: x['score'], reverse=True)

    for i, opt in enumerate(options, 1):
        board = opt['board']
        size = opt['size']
        corners = opt['corners']
        papers = opt['papers']

        print(f"💡 选项 {i}: {board[0]}x{board[1]} 棋盘格")
        print(f"   • 总尺寸: {size[0]}x{size[1]}mm")
        print(f"   • 内角点: {corners} 个")
        print(f"   • 适合纸张: {', '.join(papers) if papers else '需要更大纸张'}")

        # 特殊推荐
        if corners >= 80:
            print("   ⭐ 推荐: 角点丰富，标定精度高")
        elif corners >= 60:
            print("   ✅ 良好: 平衡的角点数量")
        elif board == (9, 6):
            print("   📝 当前: 你的现有配置")

        print()

def analyze_large_paper_options(square_size_mm=25, target_corners=60):
    """
    分析大纸张选项
    """
    print("\n📄 大纸张选项分析")
    print("=" * 50)
    print(f"🎯 目标: {target_corners}+ 内角点，{square_size_mm}mm格子")
    print()

    # 计算需要的尺寸
    min_squares_per_side = int((target_corners ** 0.5) + 1)
    estimated_size = min_squares_per_side * square_size_mm

    print("📊 尺寸估算:")
    print(f"   • 每边最小格子数: {min_squares_per_side}")
    print(f"   • 估算边长: {estimated_size}mm")
    print()

    # 推荐纸张
    recommendations = [
        ('A3', 297, 420, '最常用，性价比高'),
        ('Tabloid/B2', 279, 432, '接近A3，易获取'),
        ('A2', 420, 594, '专业级，精度最高'),
        ('A1', 594, 841, '最大尺寸，专业应用'),
        ('Custom', estimated_size*1.5, estimated_size*1.5, '自定义尺寸')
    ]

    print("📋 推荐纸张:")
    for name, w, h, desc in recommendations:
        fits = estimated_size <= min(w, h) - 20  # 考虑边距
        status = "✅ 适合" if fits else "❌ 太小"

        print(f"   • {name}: {w}x{h}mm")
        print(f"     {status} - {desc}")

        if fits and w >= estimated_size * 1.2:
            print("     💡 建议选择: 留有余量，便于操作")

def create_custom_chessboard_design(board_size=(10, 7), square_size=25, paper='A3'):
    """
    创建自定义棋盘格设计说明
    """
    print(f"\n🎨 自定义棋盘格设计: {board_size[0]}x{board_size[1]}, {square_size}mm格子")
    print("=" * 70)

    # 计算尺寸
    width_squares = board_size[0] + 1
    height_squares = board_size[1] + 1
    total_width = width_squares * square_size
    total_height = height_squares * square_size

    print("📐 设计参数:")
    print(f"   • 棋盘格: {board_size[0]}x{board_size[1]} 内角点")
    print(f"   • 实际格子: {width_squares}x{height_squares}")
    print(f"   • 格子尺寸: {square_size}mm")
    print(f"   • 总尺寸: {total_width}x{total_height}mm")
    print(f"   • 内角点: {board_size[0] * board_size[1]} 个")
    print()

    # 打印设计说明
    print("🖨️ 打印建议:")
    print(f"   • 纸张: {paper} 或更大")
    print("   • 打印质量: 高质量模式")
    print("   • 颜色: 黑白棋盘格")
    print("   • 边距: 至少15mm")
    print()

    print("📏 精确尺寸标记:")
    print(f"   • 总宽度: {total_width}mm")
    print(f"   • 总高度: {total_height}mm")
    print(f"   • 每个格子: {square_size}mm x {square_size}mm")
    print(f"   • 黑白交替: 从左上角开始黑色")
    print()

    print("🎯 使用建议:")
    print("   • 固定在硬板上使用")
    print("   • 确保表面平整")
    print("   • 保持足够距离拍摄")
    print("   • 避免弯曲变形")

def main():
    """主函数"""
    print("🀄️ 大型棋盘格配置建议器")
    print("=" * 80)
    print("💡 你想把棋盘尺寸做大，我们来探索最佳配置！")
    print()

    # 展示大型配置选项
    generate_large_chessboard_configs(square_size_mm=25)

    # 分析大纸张选项
    analyze_large_paper_options(square_size_mm=25, target_corners=60)

    # 提供具体设计示例
    print("\n" + "=" * 80)
    print("🎨 具体设计示例:")
    print("-" * 80)

    # 高质量配置
    create_custom_chessboard_design((10, 7), 25, 'A3')

    print("\n" + "-" * 80)

    # 最大化配置
    create_custom_chessboard_design((12, 8), 25, 'A2')

if __name__ == "__main__":
    main()
