#!/usr/bin/env python3
"""
相机标定相对高度转换为绝对高度的工具
"""

import json
import os

def calculate_absolute_heights(calibration_file, board_height_from_ground=0):
    """
    将标定文件中的相对高度转换为绝对高度

    参数:
    calibration_file: 标定文件路径 (JSON格式)
    board_height_from_ground: 标定板距离地面的高度 (mm)
    """
    if not os.path.exists(calibration_file):
        print(f"❌ 找不到标定文件: {calibration_file}")
        return None

    try:
        with open(calibration_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tvecs = data.get('tvecs', [])

        if not tvecs:
            print("❌ 标定文件中没有相机位置信息")
            return None

        print("🎯 相机高度转换工具"        print("=" * 60)
        print(f"标定文件: {os.path.basename(calibration_file)}")
        print(f"标定板距地面高度: {board_height_from_ground} mm")
        print(f"标定位置数量: {len(tvecs)}")
        print()

        absolute_heights = []
        print("📊 高度转换结果:")
        print("-" * 60)
        print(f"{'位置':<6} {'相对高度(mm)':<15} {'绝对高度(mm)':<15} {'绝对高度(cm)':<15}")
        print("-" * 60)

        for i, tvec in enumerate(tvecs):
            if len(tvec) >= 3:
                # 获取相对高度 (Z坐标)
                relative_height = tvec[2][0] if isinstance(tvec[2], list) else tvec[2]

                # 计算绝对高度
                absolute_height = board_height_from_ground + relative_height
                absolute_heights.append(absolute_height)

                # 显示结果
                print(f"{i+1:<6} {relative_height:<15.2f} {absolute_height:<15.2f} {absolute_height/10:<15.1f}")

        print("-" * 60)

        # 统计信息
        if absolute_heights:
            print("
📈 绝对高度统计:"            print(f"  • 平均绝对高度: {sum(absolute_heights)/len(absolute_heights):.2f} mm ({sum(absolute_heights)/len(absolute_heights)/10:.1f} cm)")
            print(f"  • 最大绝对高度: {max(absolute_heights):.2f} mm ({max(absolute_heights)/10:.1f} cm)")
            print(f"  • 最小绝对高度: {min(absolute_heights):.2f} mm ({min(absolute_heights)/10:.1f} cm)")
            print(f"  • 高度变化范围: {max(absolute_heights) - min(absolute_heights):.2f} mm ({(max(absolute_heights) - min(absolute_heights))/10:.1f} cm)")

        return absolute_heights

    except Exception as e:
        print(f"❌ 处理文件时出错: {e}")
        return None

def interactive_mode():
    """交互模式，让用户输入标定板高度"""
    print("🔧 相机绝对高度计算工具 (交互模式)")
    print("=" * 50)

    # 查找标定文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_files = []

    for file in os.listdir(current_dir):
        if file.endswith('.json') and 'calibration' in file.lower():
            calibration_files.append(file)

    if not calibration_files:
        print("❌ 在当前目录中未找到标定文件")
        return

    print("📁 找到的标定文件:")
    for i, file in enumerate(calibration_files, 1):
        file_path = os.path.join(current_dir, file)
        size = os.path.getsize(file_path) / 1024
        print(f"  {i}. {file} ({size:.1f} KB)")

    # 选择文件
    if len(calibration_files) == 1:
        selected_file = calibration_files[0]
        print(f"\n✅ 自动选择文件: {selected_file}")
    else:
        while True:
            try:
                choice = input(f"\n请选择标定文件 (1-{len(calibration_files)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(calibration_files):
                    selected_file = calibration_files[int(choice) - 1]
                    break
                else:
                    print("❌ 无效选择，请重新输入")
            except KeyboardInterrupt:
                print("\n👋 退出程序")
                return

    # 输入标定板高度
    while True:
        try:
            board_height = input("请输入标定板距离地面的高度 (mm，默认为0): ").strip()
            if board_height == "":
                board_height = 0
            else:
                board_height = float(board_height)
            break
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n👋 退出程序")
            return

    # 计算绝对高度
    file_path = os.path.join(current_dir, selected_file)
    calculate_absolute_heights(file_path, board_height)

if __name__ == "__main__":
    # 检查命令行参数
    if len(os.sys.argv) > 1:
        # 命令行模式
        if len(os.sys.argv) >= 3:
            calibration_file = os.sys.argv[1]
            board_height = float(os.sys.argv[2])
            calculate_absolute_heights(calibration_file, board_height)
        else:
            print("用法: python3 calculate_absolute_height.py <标定文件> <标定板高度>")
            print("示例: python3 calculate_absolute_height.py calibration.json 750")
    else:
        # 交互模式
        interactive_mode()
