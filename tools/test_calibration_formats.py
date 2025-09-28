#!/usr/bin/env python3
"""
测试标定文件多格式保存功能

用于验证JSON和XML格式的保存和加载是否正常工作
"""

import numpy as np
import os
import tempfile
from calibration_file_manager import CalibrationFileManager

def create_test_calibration_data():
    """创建测试标定数据"""
    # 模拟相机标定结果
    camera_matrix = np.array([
        [800.0, 0.0, 320.0],
        [0.0, 800.0, 240.0],
        [0.0, 0.0, 1.0]
    ])

    dist_coeffs = np.array([0.1, -0.2, 0.01, 0.001, 0.0])

    # 模拟外参
    rvecs = [np.array([0.1, 0.2, 0.3]), np.array([0.05, 0.15, 0.25])]
    tvecs = [np.array([10.0, 5.0, 100.0]), np.array([15.0, 8.0, 105.0])]

    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'board_params': {'size': (7, 6), 'square_size': 25.0},
        'calibration_date': '2024-01-01T12:00:00',
        'image_size': (640, 480),
        'rvecs': rvecs,
        'tvecs': tvecs,
        'per_view_errors': [0.5, 0.3],
        'successful_image_indices': [0, 1],
        'total_images_processed': 2,
        'successful_images_count': 2
    }

    return calibration_data

def test_multi_format_save_load():
    """测试多格式保存和加载"""
    print("🧪 开始测试多格式标定文件保存和加载...")

    manager = CalibrationFileManager()
    test_data = create_test_calibration_data()

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 测试目录: {temp_dir}")

        # 测试保存所有格式
        print("\n💾 保存测试数据为多种格式...")
        saved_files = manager.save_calibration_multi_format(
            test_data, temp_dir, formats=['npz', 'json', 'xml']
        )

        print(f"✅ 保存完成，共生成 {len(saved_files)} 个文件:")
        for fmt, filepath in saved_files.items():
            size = os.path.getsize(filepath) / 1024
            print(f"  • {fmt.upper()}: {os.path.basename(filepath)} ({size:.1f} KB)")

        # 测试加载每种格式
        print("\n📖 测试加载各种格式...")
        for fmt, filepath in saved_files.items():
            try:
                loaded_data, format_type = manager.load_calibration_file(filepath)
                print(f"  ✅ {fmt.upper()} 格式加载成功")

                # 验证关键数据
                if 'camera_matrix' in loaded_data:
                    cm_loaded = loaded_data['camera_matrix']
                    cm_original = test_data['camera_matrix']
                    if np.allclose(cm_loaded, cm_original, rtol=1e-10):
                        print(f"    • 相机矩阵数据一致")
                    else:
                        print(f"    ❌ 相机矩阵数据不一致")

                if 'dist_coeffs' in loaded_data:
                    dc_loaded = loaded_data['dist_coeffs']
                    dc_original = test_data['dist_coeffs']
                    if np.allclose(dc_loaded, dc_original, rtol=1e-10):
                        print(f"    • 畸变系数数据一致")
                    else:
                        print(f"    ❌ 畸变系数数据不一致")

            except Exception as e:
                print(f"  ❌ {fmt.upper()} 格式加载失败: {e}")

        # 显示JSON文件内容示例
        json_file = saved_files.get('json')
        if json_file:
            print("\n📄 JSON文件内容预览 (前500字符):")
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(content[:500] + "..." if len(content) > 500 else content)
            except Exception as e:
                print(f"❌ 读取JSON文件失败: {e}")

    print("\n🎉 测试完成!")

def test_format_detection():
    """测试格式自动检测"""
    print("\n🔍 测试格式自动检测...")

    manager = CalibrationFileManager()
    test_data = create_test_calibration_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        # 保存不同格式的文件
        saved_files = manager.save_calibration_multi_format(
            test_data, temp_dir, formats=['npz', 'json', 'xml']
        )

        # 测试自动检测
        for fmt, filepath in saved_files.items():
            try:
                # 移除文件扩展名，测试自动检测
                base_path = os.path.splitext(filepath)[0]
                temp_path = base_path + "_test"

                # 复制文件内容但不带扩展名
                with open(filepath, 'rb') as src, open(temp_path, 'wb') as dst:
                    dst.write(src.read())

                # 尝试加载并自动检测格式
                loaded_data, detected_format = manager.load_calibration_file(temp_path)
                print(f"  ✅ 自动检测 {fmt.upper()} 格式 -> {detected_format.upper()}")

                # 清理临时文件
                os.remove(temp_path)

            except Exception as e:
                print(f"  ❌ 自动检测 {fmt.upper()} 格式失败: {e}")

if __name__ == "__main__":
    test_multi_format_save_load()
    test_format_detection()
