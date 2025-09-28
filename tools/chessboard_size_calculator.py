#!/usr/bin/env python3
"""
棋盘格尺寸计算器
帮助确定合适的棋盘格尺寸和纸张大小
"""

def calculate_chessboard_size(board_size, square_size_mm):
    """
    计算棋盘格的总尺寸

    Args:
        board_size: (width, height) - 内角点数量，如 (9, 6)
        square_size_mm: 单个格子的边长，单位mm
    """
    inner_corners_width = board_size[0]   # 横向内角点数
    inner_corners_height = board_size[1]  # 纵向内角点数

    # 计算需要的格子数量
    squares_width = inner_corners_width + 1   # 横向格子数
    squares_height = inner_corners_height + 1 # 纵向格子数

    # 计算总尺寸
    total_width = squares_width * square_size_mm
    total_height = squares_height * square_size_mm

    return {
        'board_size': board_size,
        'square_size_mm': square_size_mm,
        'squares_width': squares_width,
        'squares_height': squares_height,
        'total_width_mm': total_width,
        'total_height_mm': total_height,
        'inner_corners': inner_corners_width * inner_corners_height,
        'total_squares': squares_width * squares_height
    }

def check_paper_compatibility(chessboard_info, paper_sizes):
    """
    检查棋盘格与各种纸张的兼容性
    """
    results = []

    for paper_name, paper_size in paper_sizes.items():
        paper_width, paper_height = paper_size
        board_width = chessboard_info['total_width_mm']
        board_height = chessboard_info['total_height_mm']

        # 检查是否适合（考虑边距）
        margin = 10  # 10mm边距
        fits_width = board_width <= (paper_width - 2 * margin)
        fits_height = board_height <= (paper_height - 2 * margin)

        # 可以旋转90度再试
        fits_rotated = False
        if not (fits_width and fits_height):
            fits_rotated_width = board_height <= (paper_width - 2 * margin)
            fits_rotated_height = board_width <= (paper_height - 2 * margin)
            fits_rotated = fits_rotated_width and fits_rotated_height

        results.append({
            'paper_name': paper_name,
            'paper_size': paper_size,
            'fits_normal': fits_width and fits_height,
            'fits_rotated': fits_rotated,
            'fits_any': (fits_width and fits_height) or fits_rotated,
            'overflow_width': max(0, board_width - paper_width + 2 * margin),
            'overflow_height': max(0, board_height - paper_height + 2 * margin)
        })

    return results

def suggest_optimal_configurations(target_paper='A4'):
    """
    建议最佳的棋盘格配置
    """
    print("🎯 棋盘格尺寸优化建议")
    print("=" * 60)

    # 常用纸张尺寸 (宽x高, mm)
    paper_sizes = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A2': (420, 594),
        'Letter': (216, 279),
        'Legal': (216, 356),
        'Tabloid': (279, 432)
    }

    # 建议的棋盘格配置
    suggestions = [
        {'board': (9, 6), 'square': 20},   # 你的当前配置
        {'board': (9, 6), 'square': 18},   # 缩小格子
        {'board': (7, 5), 'square': 25},   # 减少角点
        {'board': (8, 5), 'square': 22},   # 平衡配置
        {'board': (6, 4), 'square': 30},   # 更小的棋盘格
        {'board': (10, 7), 'square': 18},  # 更大的棋盘格，更小的格子
    ]

    target_paper_size = paper_sizes.get(target_paper, paper_sizes['A4'])

    print(f"📋 目标纸张: {target_paper} ({target_paper_size[0]}x{target_paper_size[1]}mm)")
    print(f"📏 纸张可用区域: {target_paper_size[0]-20}x{target_paper_size[1]-20}mm (扣除20mm边距)")
    print()

    best_suggestion = None
    best_score = 0

    for i, config in enumerate(suggestions, 1):
        board_info = calculate_chessboard_size(config['board'], config['square'])

        # 检查与目标纸张的兼容性
        compatibility = check_paper_compatibility(board_info, {target_paper: target_paper_size})[0]

        # 计算评分（基于角点数量和尺寸合理性）
        corners_score = board_info['inner_corners'] / 54  # 9x6=54作为基准
        size_score = min(board_info['total_width_mm'] / target_paper_size[0],
                        board_info['total_height_mm'] / target_paper_size[1])
        score = corners_score * 0.6 + size_score * 0.4

        print(f"💡 建议 {i}:")
        print(f"   • 棋盘格: {config['board'][0]}x{config['board'][1]} (内角点)")
        print(f"   • 格子尺寸: {config['square']}mm")
        print(f"   • 总尺寸: {board_info['total_width_mm']}x{board_info['total_height_mm']}mm")
        print(f"   • 内角点: {board_info['inner_corners']} 个")

        if compatibility['fits_any']:
            print("   ✅ 适合目标纸张")
            if compatibility['fits_rotated']:
                print("   ↻ 需要旋转90度")
        else:
            overflow_w = compatibility['overflow_width']
            overflow_h = compatibility['overflow_height']
            print(f"   ❌ 不适合 - 超出: {overflow_w}x{overflow_h}mm")

        print(".1f")
        print()

        # 记录最佳建议
        if compatibility['fits_any'] and score > best_score:
            best_score = score
            best_suggestion = config

    if best_suggestion:
        print("🏆 最佳推荐配置:")
        best_info = calculate_chessboard_size(best_suggestion['board'], best_suggestion['square'])
        print(f"   • 棋盘格: {best_suggestion['board'][0]}x{best_suggestion['board'][1]}")
        print(f"   • 格子尺寸: {best_suggestion['square']}mm")
        print(f"   • 总尺寸: {best_info['total_width_mm']}x{best_info['total_height_mm']}mm")
        print(f"   • 内角点: {best_info['inner_corners']} 个")

def analyze_current_setup():
    """
    分析当前的用户设置
    """
    print("🔍 分析你的当前设置")
    print("-" * 40)

    # 用户的当前配置
    current_board = (9, 6)  # 9x6内角点
    current_square = 25     # 25mm格子

    board_info = calculate_chessboard_size(current_board, current_square)

    print("📊 当前配置:")
    print(f"   • 棋盘格: {current_board[0]}x{current_board[1]} 内角点")
    print(f"   • 格子尺寸: {current_square}mm")
    print(f"   • 实际格子数: {board_info['squares_width']}x{board_info['squares_height']}")
    print(f"   • 总尺寸: {board_info['total_width_mm']}x{board_info['total_height_mm']}mm")
    print(f"   • 内角点总数: {board_info['inner_corners']} 个")

    # 检查各种纸张
    paper_sizes = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A5': (148, 210),
        'Letter': (216, 279)
    }

    print("\n📋 与纸张兼容性:")
    compatibility_results = check_paper_compatibility(board_info, paper_sizes)

    for result in compatibility_results:
        status = "✅ 适合" if result['fits_any'] else "❌ 不适合"
        rotate = " (需旋转)" if result['fits_rotated'] and not result['fits_normal'] else ""

        if not result['fits_any']:
            overflow = f" (超出 {result['overflow_width']}x{result['overflow_height']}mm)"
        else:
            overflow = ""

        print(f"   • {result['paper_name']}: {status}{rotate}{overflow}")

    # 给出具体建议
    print("\n💡 建议:")
    if board_info['total_width_mm'] > 210:  # A4宽度
        print("❌ 棋盘格宽度超出A4纸张，需要:")
        print("   1. 使用更大的纸张 (A3或Tabloid)")
        print("   2. 减小格子尺寸 (建议18-20mm)")
        print("   3. 减少棋盘格尺寸 (7x5或8x5)")

        # 计算适合A4纸的建议尺寸
        max_width_a4 = 200  # A4宽度减去边距
        max_height_a4 = 277  # A4高度减去边距

        suggested_square = min(max_width_a4 / board_info['squares_width'],
                              max_height_a4 / board_info['squares_height'])

        print(f"   4. 适合A4纸的格子尺寸: {suggested_square:.0f}mm")
    else:
        print("✅ 尺寸合适，可以在A4纸上打印")

def main():
    """主函数"""
    print("🎯 棋盘格尺寸计算器")
    print("=" * 60)

    # 分析当前设置
    analyze_current_setup()

    print("\n" + "=" * 60)

    # 给出优化建议
    suggest_optimal_configurations('A4')

if __name__ == "__main__":
    main()
