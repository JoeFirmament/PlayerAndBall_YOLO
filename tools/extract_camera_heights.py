#!/usr/bin/env python3
"""
从标定文件中提取相机高度信息
"""

import json
import os

def extract_camera_heights_from_json(json_file_path):
    """从JSON文件中提取相机高度信息"""
    print(f"📄 从JSON文件中提取相机高度: {os.path.basename(json_file_path)}")
    print("=" * 60)

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取tvecs（平移向量）
        tvecs = data.get('tvecs', [])

        if not tvecs:
            print("❌ 未找到相机位置信息 (tvecs)")
            return

        print(f"📊 共找到 {len(tvecs)} 个标定位置")
        print("\n🎯 相机高度信息 (Z坐标，单位: mm):")
        print("-" * 45)

        heights = []
        for i, tvec in enumerate(tvecs):
            if len(tvec) >= 3:
                # tvec格式: [[x], [y], [z]]
                height = tvec[2][0] if isinstance(tvec[2], list) else tvec[2]
                heights.append(height)
                print("2d")

        if heights:
            print("
📈 统计信息:"            print(f"  • 平均高度: {sum(heights) / len(heights):.2f} mm")
            print(f"  • 最大高度: {max(heights):.2f} mm")
            print(f"  • 最小高度: {min(heights):.2f} mm")

        return heights

    except Exception as e:
        print(f"❌ 读取JSON文件失败: {e}")
        return None

def extract_camera_heights_from_xml(xml_file_path):
    """从XML文件中提取相机高度信息"""
    print(f"📄 从XML文件中提取相机高度: {os.path.basename(xml_file_path)}")
    print("=" * 60)

    try:
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找translation_vectors部分
        if 'translation_vectors' not in content:
            print("❌ 未找到相机位置信息 (translation_vectors)")
            return

        print("🎯 相机高度信息 (Z坐标，单位: mm):")
        print("-" * 45)

        heights = []
        lines = content.split('\n')
        i = 0

        for line in lines:
            if '<tvec_' in line and 'type_id="opencv-matrix"' in line:
                # 找到下一个包含数据的行
                for j in range(i + 1, len(lines)):
                    data_line = lines[j].strip()
                    if 'data>' in data_line and '</data>' in data_line:
                        # 提取三个数值，第三个是高度
                        data_match = data_line.replace('<data>', '').replace('</data>', '').strip()
                        values = data_match.split()
                        if len(values) >= 3:
                            height = float(values[2])  # Z坐标（高度）
                            heights.append(height)
                            print("6.2f")
                        break
            i += 1

        if heights:
            print("
📈 统计信息:"            print(f"  • 平均高度: {sum(heights) / len(heights):.2f} mm")
            print(f"  • 最大高度: {max(heights):.2f} mm")
            print(f"  • 最小高度: {min(heights):.2f} mm")

        return heights

    except Exception as e:
        print(f"❌ 读取XML文件失败: {e}")
        return None

def find_calibration_files(directory):
    """查找目录中的标定文件"""
    files = []
    for file in os.listdir(directory):
        if file.endswith(('.json', '.xml')) and 'calibration' in file.lower():
            files.append(os.path.join(directory, file))

    return sorted(files)

def main():
    """主函数"""
    print("🔍 相机标定文件高度信息提取工具")
    print("=" * 50)

    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 查找标定文件
    calibration_files = find_calibration_files(current_dir)

    if not calibration_files:
        print("❌ 在当前目录中未找到标定文件")
        print("请确保标定文件包含 'calibration' 关键字")
        return

    print(f"📁 找到 {len(calibration_files)} 个标定文件:")
    for i, file in enumerate(calibration_files, 1):
        print(f"  {i}. {os.path.basename(file)}")

    print("\n" + "=" * 50)

    # 处理每个文件
    for file_path in calibration_files:
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.json':
            extract_camera_heights_from_json(file_path)
        elif file_ext == '.xml':
            extract_camera_heights_from_xml(file_path)

        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
