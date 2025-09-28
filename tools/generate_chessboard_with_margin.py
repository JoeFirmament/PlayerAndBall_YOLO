#!/usr/bin/env python3
"""
生成带白色边距的A2纸9x6棋盘格
"""

import os
from PIL import Image, ImageDraw

def generate_chessboard_with_margin(margin_mm=20):
    """生成带边距的棋盘格"""
    print("🎨 生成带白色边距的A2纸9x6棋盘格")
    print("=" * 60)

    # 原始参数
    board_corners = (9, 6)   # 9x6内角点
    square_size_mm = 25      # 25mm格子
    dpi = 300                # 300dpi

    # 计算原始棋盘尺寸
    squares_width = board_corners[0] + 1   # 10
    squares_height = board_corners[1] + 1  # 7
    board_width_mm = squares_width * square_size_mm   # 250mm
    board_height_mm = squares_height * square_size_mm # 175mm

    # 计算带边距的总尺寸
    total_width_mm = board_width_mm + 2 * margin_mm   # 250 + 40 = 290mm
    total_height_mm = board_height_mm + 2 * margin_mm # 175 + 40 = 215mm

    # 转换为像素
    pixels_per_mm = dpi / 25.4
    board_width_px = int(board_width_mm * pixels_per_mm)
    board_height_px = int(board_height_mm * pixels_per_mm)
    margin_px = int(margin_mm * pixels_per_mm)
    square_size_px = int(square_size_mm * pixels_per_mm)

    total_width_px = board_width_px + 2 * margin_px
    total_height_px = board_height_px + 2 * margin_px

    print("📐 尺寸计算:"    print(f"   • 棋盘格: {board_corners[0]}×{board_corners[1]} 内角点")
    print(f"   • 原始尺寸: {board_width_mm}mm × {board_height_mm}mm")
    print(f"   • 边距: {margin_mm}mm")
    print(f"   • 总尺寸: {total_width_mm}mm × {total_height_mm}mm")
    print(f"   • 分辨率: {dpi}dpi")
    print(f"   • 像素尺寸: {total_width_px}×{total_height_px}px")

    # 创建白色背景图像（更大的画布）
    img = Image.new('RGB', (total_width_px, total_height_px), 'white')
    draw = ImageDraw.Draw(img)

    # 计算棋盘格在画布上的位置（居中）
    board_x_start = margin_px
    board_y_start = margin_px

    # 绘制棋盘格
    print("
🔲 绘制棋盘格..."    for row in range(squares_height):
        for col in range(squares_width):
            # 计算格子的位置（加上边距偏移）
            x1 = board_x_start + col * square_size_px
            y1 = board_y_start + row * square_size_px
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

    # 添加尺寸标记
    print("
📏 添加尺寸标记..."    # 绘制尺寸线
    line_color = 'red'
    line_width = 2

    # 水平尺寸线（在上方边距中）
    text_y = margin_px // 2
    draw.line([margin_px, text_y, margin_px + board_width_px, text_y],
             fill=line_color, width=line_width)

    # 垂直尺寸线（在右侧边距中）
    text_x = margin_px + board_width_px + margin_px // 2
    draw.line([text_x, margin_px, text_x, margin_px + board_height_px],
             fill=line_color, width=line_width)

    # 添加尺寸文本
    try:
        # 添加水平尺寸文本
        draw.text((margin_px + board_width_px // 2, text_y - 10),
                 f"{board_width_mm}mm", fill='red', anchor='mm')
        # 添加垂直尺寸文本
        draw.text((text_x + 10, margin_px + board_height_px // 2),
                 f"{board_height_mm}mm", fill='red', anchor='lm')
    except:
        print("   ⚠️ 无法添加尺寸文本")

    # 保存图像
    filename = f"chessboard_9x6_{board_width_mm}x{board_height_mm}mm_margin{margin_mm}mm.png"
    filepath = os.path.join(os.getcwd(), filename)

    print("
💾 保存图像..."    img.save(filepath, 'PNG', dpi=(dpi, dpi))
    print(f"   ✅ 图像已保存: {filename}")
    print(f"   📁 文件路径: {filepath}")

    # 显示文件信息
    file_size = os.path.getsize(filepath)
    print("
📊 文件信息:"    print(f"   • 文件大小: {file_size:,} bytes")
    print(".2f"    print(f"   • 图像格式: PNG")
    print(f"   • 色彩模式: RGB")

    return filepath, total_width_mm, total_height_mm

def show_usage_guide(total_width_mm, total_height_mm, margin_mm):
    """显示使用指南"""

    print("
🖨️ 打印指南:"    print("=" * 50)
    print("1. 📄 纸张选择:")
    print("   • A2纸 (420×594mm)"    print(f"   • 打印区域: {total_width_mm}×{total_height_mm}mm")
    print(f"   • 白色边距: {margin_mm}mm (四周)")

    print("
2. 🖨️ 打印设置:"    print("   • 打印质量: 最高质量/照片模式")
    print("   • 色彩模式: 彩色 (保留红色尺寸线)")
    print("   • 缩放: 100% (实际尺寸)")
    print("   • 纸张方向: 根据需要旋转")

    print("
3. 📐 摆放位置选项:"    print("   • 居中打印: 最大化余量")
    print("   • 靠左上角: 便于裁剪")
    print("   • 自定义位置: 根据使用场景")

    print("
4. ✂️ 后期处理:"    print("   • 保留白色边距: 便于固定和裁剪")
    print("   • 裁剪时留出: 至少5mm白边")
    print("   • 粘贴到硬质板材上")
    print("   • 确保表面平整")

    print("
5. 🎯 优势说明:"    print("   • ✅ 白色边距便于固定")
    print("   • ✅ 红色尺寸线便于对齐")
    print("   • ✅ 专业外观")
    print("   • ✅ 更好的检测效果")

def main():
    """主函数"""
    print("🀄️ 带白色边距的A2纸9x6棋盘格生成器")
    print("=" * 80)

    # 生成带边距的棋盘格
    filepath, total_width, total_height = generate_chessboard_with_margin(margin_mm=20)

    # 显示使用指南
    show_usage_guide(total_width, total_height, 20)

    print("
🎉 生成完成！"    print("   ✅ 带白色边距的棋盘格")
    print("   ✅ 包含尺寸标记")
    print("   ✅ 高分辨率打印质量")
    print("   ✅ 专业外观")

    print("
📋 文件对比:"    print(f"   • 旧版: chessboard_9x6_250x175mm.png (无边距)")
    print(f"   • 新版: {os.path.basename(filepath)} (20mm边距)")

if __name__ == "__main__":
    main()
