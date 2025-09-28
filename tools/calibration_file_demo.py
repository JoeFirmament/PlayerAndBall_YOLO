#!/usr/bin/env python3
"""
多格式标定文件管理器使用演示

演示如何使用多格式保存和加载标定文件
"""

import numpy as np
import os
from datetime import datetime
from calibration_file_manager import CalibrationFileManager

def create_sample_calibration_data():
    """创建示例标定数据"""
    print("🎯 创建示例标定数据...")

    # 示例相机标定参数
    camera_matrix = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.array([-0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

    # 示例外参数据
    rvecs = [
        np.array([0.1, 0.05, -0.02], dtype=np.float32),
        np.array([0.08, 0.03, 0.01], dtype=np.float32),
        np.array([0.12, -0.01, 0.03], dtype=np.float32)
    ]

    tvecs = [
        np.array([100.0, 50.0, 500.0], dtype=np.float32),
        np.array([120.0, 30.0, 480.0], dtype=np.float32),
        np.array([80.0, 70.0, 520.0], dtype=np.float32)
    ]

    # 完整的标定数据
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'board_params': {'size': (7, 6), 'square_size': 25.0},
        'calibration_date': datetime.now().isoformat(),
        'image_size': (1280, 720),
        'per_view_errors': [0.23, 0.19, 0.25],
        'successful_image_indices': [0, 1, 2],
        'total_images_processed': 5,
        'successful_images_count': 3,
        'calibration_method': 'OpenCV calibrateCamera',
        'opencv_version': '4.8.0',
        'comments': 'Sample calibration data for demonstration'
    }

    print("✅ 示例标定数据创建完成")
    return calibration_data

def demonstrate_multi_format_save():
    """演示多格式保存功能"""
    print("\n" + "="*60)
    print("💾 多格式标定文件保存演示")
    print("="*60)

    # 创建文件管理器
    manager = CalibrationFileManager()

    # 创建示例数据
    calibration_data = create_sample_calibration_data()

    # 设置输出目录
    output_dir = "./demo_calibration_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n📁 输出目录: {output_dir}")

    # 保存为所有支持格式
    print("\n🔄 保存为多种格式...")
    saved_files = manager.save_calibration_multi_format(
        calibration_data,
        output_dir,
        formats=['npz', 'json', 'xml']
    )

    print("\n📋 保存结果:")
    for fmt, filepath in saved_files.items():
        filename = os.path.basename(filepath)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  • {fmt.upper()}: {filename} ({size_kb:.1f} KB)")

    # 显示文件内容对比
    print("\n📊 文件内容对比:")
    for fmt, filepath in saved_files.items():
        print(f"\n{fmt.upper()} 格式文件内容预览:")
        try:
            if fmt == 'npz':
                data = np.load(filepath)
                print("    包含的数组:")
                for key in data.keys():
                    if isinstance(data[key], np.ndarray):
                        shape = data[key].shape
                        dtype = data[key].dtype
                        print(f"      • {key}: {shape} ({dtype})")
                    else:
                        print(f"      • {key}: {type(data[key]).__name__}")

            elif fmt == 'json':
                with open(filepath, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                print(f"    包含 {len(data)} 个字段")
                print(f"    主要字段: {list(data.keys())[:5]}...")

            elif fmt == 'xml':
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                lines = content.count('\n')
                print(f"    XML 文件行数: {lines}")
                print("    包含 OpenCV 标准格式数据")
        except Exception as e:
            print(f"    预览失败: {e}")

    return saved_files

def demonstrate_format_loading():
    """演示格式加载功能"""
    print("\n" + "="*60)
    print("📂 标定文件加载演示")
    print("="*60)

    manager = CalibrationFileManager()
    output_dir = "./demo_calibration_output"

    # 加载不同格式的文件
    formats_to_test = ['npz', 'json', 'xml']

    for fmt in formats_to_test:
        filename = f"{manager.file_prefix}.{fmt}"
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            print(f"\n❌ {fmt.upper()} 文件不存在: {filename}")
            continue

        print(f"\n📖 加载 {fmt.upper()} 格式文件: {filename}")

        try:
            # 加载文件
            data, detected_format = manager.load_calibration_file(filepath)

            print(f"  ✅ 加载成功，检测格式: {detected_format.upper()}")
            print("  📋 加载的数据:"            print(f"    • 相机矩阵: {data.get('camera_matrix', 'N/A')}")
            print(f"    • 畸变系数: {data.get('dist_coeffs', 'N/A')}")
            print(f"    • 外参数量: {len(data.get('rvecs', []))} 视图")
            print(f"    • 标定日期: {data.get('calibration_date', 'N/A')}")
            print(f"    • 图像尺寸: {data.get('image_size', 'N/A')}")

            # 验证数据完整性
            required_fields = ['camera_matrix', 'dist_coeffs']
            optional_fields = ['rvecs', 'tvecs', 'board_params']

            print("  🔍 数据完整性检查:")
            for field in required_fields:
                status = "✅" if field in data else "❌"
                print(f"    • {field}: {status}")

            for field in optional_fields:
                status = "✅" if field in data else "⚠️"
                print(f"    • {field}: {status}")

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

def demonstrate_file_conversion():
    """演示文件格式转换"""
    print("\n" + "="*60)
    print("🔄 文件格式转换演示")
    print("="*60)

    manager = CalibrationFileManager()
    output_dir = "./demo_calibration_output"
    convert_dir = "./demo_conversion_output"
    os.makedirs(convert_dir, exist_ok=True)

    # 选择一个源文件进行转换
    source_file = os.path.join(output_dir, f"{manager.file_prefix}.npz")

    if not os.path.exists(source_file):
        print("❌ 找不到源文件进行转换演示")
        return

    print(f"📁 源文件: {os.path.basename(source_file)}")
    print(f"📁 输出目录: {convert_dir}")

    # 转换到其他格式
    conversions = [
        ('npz', 'json'),
        ('npz', 'xml'),
        ('json', 'xml')
    ]

    for from_fmt, to_fmt in conversions:
        try:
            if from_fmt == 'npz':
                input_file = source_file
            else:
                input_file = os.path.join(output_dir, f"{manager.file_prefix}.{from_fmt}")

            if not os.path.exists(input_file):
                print(f"⚠️ 跳过 {from_fmt.upper()} -> {to_fmt.upper()}: 源文件不存在")
                continue

            print(f"\n🔄 转换: {from_fmt.upper()} -> {to_fmt.upper()}")

            # 执行转换
            output_file = manager.convert_format(input_file, to_fmt, convert_dir)

            # 显示转换结果
            original_size = os.path.getsize(input_file) / 1024
            converted_size = os.path.getsize(output_file) / 1024

            print("  ✅ 转换成功"            print(".1f"            print(".1f"            print(".1f"
            # 验证转换后的文件可以加载
            converted_data, detected_fmt = manager.load_calibration_file(output_file)
            print(f"  🔍 验证加载: {detected_fmt.upper()} 格式加载成功")

        except Exception as e:
            print(f"  ❌ 转换失败: {e}")

def demonstrate_file_management():
    """演示文件管理功能"""
    print("\n" + "="*60)
    print("📋 标定文件管理演示")
    print("="*60)

    manager = CalibrationFileManager()
    output_dir = "./demo_calibration_output"

    # 列出标定文件
    print(f"📂 扫描目录: {output_dir}")
    files = manager.list_calibration_files(output_dir)

    if not files:
        print("❌ 没有找到标定文件")
        return

    print("\
📋 发现的标定文件:"    for i, file_info in enumerate(files, 1):
        timestamp = datetime.fromisoformat(file_info['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {i}. {file_info['filename']}")
        print(f"     • 格式: {file_info['format'].upper()}")
        print(f"     • 大小: {file_info['size_kb']:.1f} KB")
        print(f"     • 时间: {timestamp}")
        print(f"     • 路径: {file_info['path']}")

    # 显示文件详细信息
    if files:
        print("\
📊 文件详细信息:"        file_info = files[0]  # 显示第一个文件
        detailed_info = manager.get_file_info(file_info['path'])

        print(f"文件名: {detailed_info['filename']}")
        print(f"格式: {detailed_info['format'].upper()}")
        print(f"大小: {detailed_info['size_kb']:.1f} KB")
        print(f"修改时间: {detailed_info['modified_time']}")

        if 'timestamp' in detailed_info:
            print(f"标定时间戳: {detailed_info['timestamp']}")

def main():
    """主函数"""
    print("🚀 多格式标定文件管理器演示")
    print("=" * 60)

    try:
        # 演示1: 多格式保存
        saved_files = demonstrate_multi_format_save()

        # 演示2: 格式加载
        demonstrate_format_loading()

        # 演示3: 文件转换
        demonstrate_file_conversion()

        # 演示4: 文件管理
        demonstrate_file_management()

        print("\n" + "="*60)
        print("🎉 演示完成!")
        print("="*60)

        print("\
📁 生成的文件:"        output_dirs = ["./demo_calibration_output", "./demo_conversion_output"]
        for output_dir in output_dirs:
            if os.path.exists(output_dir):
                files = os.listdir(output_dir)
                print(f"  • {output_dir}: {len(files)} 个文件")
                for file in sorted(files)[:3]:  # 显示前3个
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files) - 3} 个文件")

        print("\
💡 使用建议:"        print("• 在实际项目中使用时，确保选择合适的格式")
        print("• NPZ格式适合Python开发和快速处理")
        print("• JSON格式适合跨平台和版本控制")
        print("• XML格式适合C++ OpenCV应用程序")
        print("• 可以同时保存多种格式以满足不同需求")

        print("\
🛠️ 命令行工具使用:"        print("python calibration_file_manager.py save <data.npz> [options]")
        print("python calibration_file_manager.py load <file>")
        print("python calibration_file_manager.py convert <input> <format>")
        print("python calibration_file_manager.py list <directory>")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
