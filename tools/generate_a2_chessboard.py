#!/usr/bin/env python3
"""
生成A2纸适配的9x6棋盘格图片
尺寸：250mm × 175mm，适配A2纸打印
"""

import os
from PIL import Image, ImageDraw

def generate_a2_chessboard(board_size=(9, 6), square_size_mm=25, dpi=300):
    """
    生成A2纸适配的棋盘格

    Args:
        board_size: (width, height) 内角点数量，如 (9, 6)
        square_size_mm: 单个格子边长，单位mm
        dpi: 打印分辨率，标准为300dpi
    """

    print("🎨 生成A2纸适配的9x6棋盘格")
    print("=" * 70)

    # 计算物理尺寸
    squares_width = board_size[0] + 1   # 10格子
    squares_height = board_size[1] + 1  # 7格子

    board_width_mm = squares_width * square_size_mm   # 250mm
    board_height_mm = squares_height * square_size_mm # 175mm

    # 转换为像素尺寸 (300dpi = 300像素/英寸, 1英寸=25.4mm)
    pixels_per_mm = dpi / 25.4
    board_width_px = int(board_width_mm * pixels_per_mm)
    board_height_px = int(board_height_mm * pixels_per_mm)
    square_size_px = int(square_size_mm * pixels_per_mm)

    print("📐 尺寸计算:"    print(f"   • 棋盘格: {board_size[0]}×{board_size[1]} 内角点")
    print(f"   • 实际格子: {squares_width}×{squares_height}")
    print(f"   • 格子尺寸: {square_size_mm}mm")
    print(f"   • 总尺寸: {board_width_mm}mm × {board_height_mm}mm")
    print(f"   • 分辨率: {dpi}dpi")
    print(f"   • 像素尺寸: {board_width_px}×{board_height_px}px")
    print(f"   • 格子像素: {square_size_px}px")

    # 创建白色背景图像
    img = Image.new('RGB', (board_width_px, board_height_px), 'white')
    draw = ImageDraw.Draw(img)

    # 绘制棋盘格
    print("
🔲 绘制棋盘格..."    for row in range(squares_height):
        for col in range(squares_width):
            # 计算格子的位置
            x1 = col * square_size_px
            y1 = row * square_size_px
            x2 = x1 + square_size_px
            y2 = y1 + square_size_px

            # 黑白交替 (左上角为黑色)
            if (row + col) % 2 == 0:
                color = 'black'
            else:
                color = 'white'

            # 绘制方格
            draw.rectangle([x1, y1, x2, y2], fill=color)

    print(f"   ✅ 棋盘格绘制完成")
    print(f"   📊 总共绘制了 {squares_width * squares_height} 个格子")

    # 添加尺寸标记（可选）
    print("
📏 添加尺寸标记..."    # 在图像边缘添加尺寸线和标记
    margin = int(5 * pixels_per_mm)  # 5mm边距

    # 绘制尺寸线
    line_color = 'red'
    line_width = 2

    # 水平尺寸线
    draw.line([margin, board_height_px - margin,
              board_width_px - margin, board_height_px - margin],
             fill=line_color, width=line_width)

    # 垂直尺寸线
    draw.line([board_width_px - margin, margin,
              board_width_px - margin, board_height_px - margin],
             fill=line_color, width=line_width)

    # 添加尺寸文本
    font_size = int(8 * pixels_per_mm)  # 8mm高的文字
    try:
        # 尝试使用默认字体
        draw.text((board_width_px//2, board_height_px - margin//2),
                 f"{board_width_mm}mm", fill='red', anchor='mm')
        draw.text((board_width_px - margin//2, board_height_px//2),
                 f"{board_height_mm}mm", fill='red', anchor='mm')
    except:
        print("   ⚠️ 无法添加尺寸文本，使用默认字体")

    # 保存图像
    output_filename = f"chessboard_9x6_{board_width_mm}x{board_height_mm}mm_{dpi}dpi.png"
    output_path = os.path.join(os.getcwd(), output_filename)

    print("
💾 保存图像..."    img.save(output_path, 'PNG', dpi=(dpi, dpi))
    print(f"   ✅ 图像已保存: {output_filename}")
    print(f"   📁 文件路径: {output_path}")

    # 显示文件信息
    file_size = os.path.getsize(output_path)
    print("
📊 文件信息:"    print(f"   • 文件大小: {file_size:,} bytes")
    print(".2f"    print(f"   • 图像格式: PNG")
    print(f"   • 色彩模式: RGB")

    return output_path

def show_printing_instructions(board_width_mm, board_height_mm, output_path):
    """显示打印说明"""

    print("
🖨️ 打印说明:"    print("=" * 50)
    print("1. 📄 纸张选择:")
    print("   • 使用A2纸 (420×594mm)")
    print("   • 纸张类型: 普通打印纸或相纸")
    print("   • 纸张质量: 建议80g以上")

    print("
2. 🖨️ 打印设置:"    print("   • 打印质量: 最高质量")
    print("   • 色彩模式: 黑白/灰度")
    print("   • 缩放: 100% (实际尺寸)")
    print("   • 边距: 无边距打印")

    print("
3. 📐 摆放位置:"    print("   • 居中打印: 最大化余量")
    print("   • 靠左上角: 便于裁剪")
    print("   • 多个棋盘: A2纸可以放多个小尺寸")

    print("
4. ✂️ 后期处理:"    print("   • 裁剪多余纸张")
    print("   • 粘贴到硬质板材上")
    print("   • 确保表面平整")
    print("   • 避免弯曲变形")

    print("
5. 📸 使用建议:"    print("   • 固定在墙上或地上")
    print("   • 距离摄像头2-3米")
    print("   • 确保完整画面")
    print("   • 光照均匀")

def main():
    """主函数"""
    print("🀄️ A2纸 9x6 棋盘格生成器")
    print("=" * 80)

    # 生成棋盘格
    output_path = generate_a2_chessboard(
        board_size=(9, 6),    # 9x6内角点
        square_size_mm=25,    # 25mm格子
        dpi=300               # 300dpi高分辨率
    )

    # 显示打印说明
    show_printing_instructions(250, 175, output_path)

    print("
🎉 生成完成！"    print("   ✅ 棋盘格图像已生成")
    print("   ✅ 尺寸完美适配A2纸")
    print("   ✅ 高分辨率打印质量")
    print("   ✅ 包含尺寸标记")
    print("
🚀 接下来:"    print("   1. 打印生成的PNG文件")
    print("   2. 粘贴到硬质板材上")
    print("   3. 重新拍摄Ground Calibration图片")
    print("   4. 测试OpenCV检测效果")

if __name__ == "__main__":
    main()
