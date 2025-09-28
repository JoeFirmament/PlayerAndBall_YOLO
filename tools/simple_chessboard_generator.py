#!/usr/bin/env python3
"""
简化的A2纸9x6棋盘格生成器
"""

import os
from PIL import Image, ImageDraw

def generate_chessboard():
    """生成9x6棋盘格"""
    print("生成A2纸适配的9x6棋盘格...")

    # 参数设置
    board_corners = (9, 6)  # 9x6内角点
    square_size_mm = 25    # 25mm格子
    dpi = 300              # 300dpi

    # 计算尺寸
    squares_width = board_corners[0] + 1   # 10
    squares_height = board_corners[1] + 1  # 7
    board_width_mm = squares_width * square_size_mm   # 250mm
    board_height_mm = squares_height * square_size_mm # 175mm

    # 转换为像素
    pixels_per_mm = dpi / 25.4
    board_width_px = int(board_width_mm * pixels_per_mm)
    board_height_px = int(board_height_mm * pixels_per_mm)
    square_size_px = int(square_size_mm * pixels_per_mm)

    print(f"棋盘格尺寸: {board_width_mm}mm x {board_height_mm}mm")
    print(f"像素尺寸: {board_width_px} x {board_height_px}")
    print(f"格子数量: {squares_width} x {squares_height}")

    # 创建图像
    img = Image.new('RGB', (board_width_px, board_height_px), 'white')
    draw = ImageDraw.Draw(img)

    # 绘制棋盘格
    for row in range(squares_height):
        for col in range(squares_width):
            x1 = col * square_size_px
            y1 = row * square_size_px
            x2 = x1 + square_size_px
            y2 = y1 + square_size_px

            # 黑白交替
            if (row + col) % 2 == 0:
                color = 'black'
            else:
                color = 'white'

            draw.rectangle([x1, y1, x2, y2], fill=color)

    # 保存文件
    filename = f"chessboard_9x6_{board_width_mm}x{board_height_mm}mm.png"
    filepath = os.path.join(os.getcwd(), filename)
    img.save(filepath, 'PNG', dpi=(dpi, dpi))

    print(f"✅ 棋盘格已生成: {filename}")
    print(f"📁 保存路径: {filepath}")

    # 文件信息
    file_size = os.path.getsize(filepath)
    print(".2f")

    return filepath

def show_print_instructions():
    """显示打印说明"""
    print("\n🖨️ 打印说明:")
    print("1. 纸张: A2纸 (420x594mm)")
    print("2. 质量: 最高质量/照片模式")
    print("3. 缩放: 100% (实际尺寸)")
    print("4. 颜色: 黑白模式")
    print("5. 摆放: 居中打印")

    print("\n✂️ 后期处理:")
    print("1. 裁剪多余纸张")
    print("2. 粘贴到硬质板材")
    print("3. 确保表面平整")

    print("\n📸 使用建议:")
    print("1. 固定在墙上或地上")
    print("2. 距离2-3米拍摄")
    print("3. 光照均匀")
    print("4. 确保棋盘格完全可见")

if __name__ == "__main__":
    print("🀄️ A2纸 9x6 棋盘格生成器")
    print("=" * 50)

    # 生成棋盘格
    filepath = generate_chessboard()

    # 显示说明
    show_print_instructions()

    print("\n🎉 生成完成！")
    print("现在可以将生成的图片打印到A2纸上")
