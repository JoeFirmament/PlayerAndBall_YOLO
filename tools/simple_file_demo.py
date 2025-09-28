#!/usr/bin/env python3
"""
多格式标定文件管理器简单演示

演示如何保存和加载多种格式的标定文件
"""

import numpy as np
import os
from datetime import datetime
from calibration_file_manager import CalibrationFileManager

def main():
    """主函数演示"""
    print("🚀 多格式标定文件管理器演示")
    print("=" * 50)

    # 创建文件管理器
    manager = CalibrationFileManager()

    # 1. 创建示例标定数据
    print("\n📝 创建示例标定数据...")
    camera_matrix = np.array([
        [800.0, 0.0, 640.0],
        [0.0, 800.0, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    dist_coeffs = np.array([-0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'calibration_date': datetime.now().isoformat(),
        'board_params': {'size': (7, 6), 'square_size': 25.0}
    }

    print("✅ 示例数据创建完成")

    # 2. 保存为多种格式
    print("\n💾 保存标定数据为多种格式...")
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)

    saved_files = manager.save_calibration_multi_format(
        calibration_data,
        output_dir,
        formats=['npz', 'json', 'xml']
    )

    print("📋 保存结果:")
    for fmt, filepath in saved_files.items():
        filename = os.path.basename(filepath)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  • {fmt.upper()}: {filename} ({size_kb:.1f} KB)")

    # 3. 加载不同格式的文件
    print("\n📂 加载测试...")
    for fmt, filepath in saved_files.items():
        print(f"\n🔍 加载 {fmt.upper()} 格式:")
        try:
            data, detected_format = manager.load_calibration_file(filepath)
            print(f"  ✅ 成功加载，格式: {detected_format.upper()}")
            print(f"  📏 相机矩阵: {data.get('camera_matrix', 'N/A')}")
            print(f"  📐 畸变系数: {data.get('dist_coeffs', 'N/A')}")
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

    # 4. 格式转换演示
    print("\n🔄 格式转换演示...")
    if saved_files.get('npz'):
        try:
            # NPZ -> JSON
            json_file = manager.convert_format(saved_files['npz'], 'json', output_dir)
            print(f"✅ NPZ -> JSON: {os.path.basename(json_file)}")

            # JSON -> XML
            xml_file = manager.convert_format(json_file, 'xml', output_dir)
            print(f"✅ JSON -> XML: {os.path.basename(xml_file)}")

        except Exception as e:
            print(f"❌ 转换失败: {e}")

    # 5. 文件列表演示
    print("\n📋 文件列表:")
    files = manager.list_calibration_files(output_dir)
    for file_info in files:
        print(f"  • {file_info['filename']} ({file_info['format'].upper()}) - {file_info['size_kb']:.1f} KB")

    print("\n" + "=" * 50)
    print("🎉 演示完成!")
    print("=" * 50)

    print("\n💡 使用建议:")
    print("• NPZ: Python原生格式，性能最佳")
    print("• JSON: 人类可读，跨平台兼容")
    print("• XML: OpenCV标准格式，C++兼容")

    print("
📁 生成的文件保存在: ./demo_output"if __name__ == "__main__":
    main()
