#!/usr/bin/env python3
"""
测试Ground Calibration保存功能和相机高度显示
"""

import numpy as np
import json
import os
from datetime import datetime

def test_ground_calibration_save():
    """测试ground calibration保存功能"""
    print("🧪 测试Ground Calibration保存功能")
    print("=" * 60)

    # 模拟ground calibration结果
    ground_homography = np.array([
        [0.5, 0.0, 1000.0],
        [0.0, 0.5, 1500.0],
        [0.0, 0.0, 1.0]
    ])

    # 模拟相机高度信息
    camera_height_info = {
        'camera_height_mm': 892.22,
        'camera_height_cm': 89.22,
        'measurement_method': 'solvePnP_from_ground_plane',
        'reference_frame': 'ground_level_Z=0'
    }

    # 完整的ground calibration结果
    ground_calibration_results = {
        'homography_matrix': ground_homography,
        'reprojection_error': 0.8,
        'board_params': {'size': [9, 6], 'square_size': 25.0},
        'successful_images': 5,
        'total_images': 5,
        'calibration_date': datetime.now().isoformat(),
        'camera_height_info': camera_height_info
    }

    # 测试保存到JSON格式
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"test_ground_calibration_{timestamp}.json"

    try:
        print("📄 测试JSON格式保存...")

        # 准备保存数据
        results_dict = {
            'ground_homography': ground_homography.tolist(),
            'reprojection_error': float(0.8),
            'board_params': {'size': [9, 6], 'square_size': 25.0},
            'calibration_results': ground_calibration_results,
            'save_timestamp': datetime.now().isoformat(),
            'file_format': 'json'
        }

        # 保存到JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON保存成功: {json_file}")

        # 读取并验证
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # 验证相机高度信息
        loaded_results = loaded_data.get('calibration_results', {})
        loaded_height_info = loaded_results.get('camera_height_info')

        if loaded_height_info:
            print("✅ 相机高度信息保存成功:")
            print(f"   • 高度 (mm): {loaded_height_info['camera_height_mm']}")
            print(f"   • 高度 (cm): {loaded_height_info['camera_height_cm']}")
            print(f"   • 测量方法: {loaded_height_info['measurement_method']}")
        else:
            print("❌ 相机高度信息保存失败")

        # 测试保存到NPZ格式
        print("\n📦 测试NPZ格式保存...")

        npz_file = f"test_ground_calibration_{timestamp}.npz"

        save_data = {
            'ground_homography': ground_homography,
            'reprojection_error': 0.8,
            'board_params': {'size': [9, 6], 'square_size': 25.0},
            'calibration_results': ground_calibration_results
        }

        # 如果有相机高度信息，也保存
        if camera_height_info:
            save_data['camera_height_mm'] = camera_height_info['camera_height_mm']
            save_data['camera_height_cm'] = camera_height_info['camera_height_cm']

        np.savez(npz_file, **save_data)
        print(f"✅ NPZ保存成功: {npz_file}")

        # 验证NPZ文件
        loaded_npz = np.load(npz_file)
        if 'camera_height_mm' in loaded_npz:
            print("✅ NPZ中的相机高度信息保存成功:")
            print(f"   • 高度 (mm): {loaded_npz['camera_height_mm']}")
            print(f"   • 高度 (cm): {loaded_npz['camera_height_cm']}")
        else:
            print("❌ NPZ中的相机高度信息保存失败")

        # 清理测试文件
        os.remove(json_file)
        os.remove(npz_file)
        print("\n🧹 测试文件已清理")
        print("🎉 所有测试通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        # 清理可能的残留文件
        for file in [json_file, npz_file]:
            if os.path.exists(file):
                os.remove(file)

def demonstrate_height_display():
    """演示相机高度信息的显示"""
    print("\n📊 相机高度信息显示演示")
    print("=" * 50)

    # 模拟完整的ground calibration结果显示
    camera_height_info = {
        'camera_height_mm': 892.22,
        'camera_height_cm': 89.22,
        'measurement_method': 'solvePnP_from_ground_plane',
        'reference_frame': 'ground_level_Z=0'
    }

    print("Ground Calibration Results:")
    print()
    print("Calibration Summary:")
    print("• Total images processed: 5")
    print("• Successful detections: 5")
    print("• Success rate: 100.0%")
    print()
    print("Calibration Parameters:")
    print("• Chessboard size: 9×6")
    print("• Square size: 25mm")
    print()
    print("Accuracy Metrics:")
    print("• Reprojection error: 0.800 pixels")
    print("• Expected coordinate accuracy: ±4.0mm")
    print("• Expected height accuracy: ±8.0mm")
    print()

    if camera_height_info:
        print("🎯 Camera Height Information:")
        print(f"• Camera height: {camera_height_info['camera_height_mm']:.2f} mm ({camera_height_info['camera_height_cm']:.1f} cm)")
        print(f"• Measurement method: {camera_height_info['measurement_method']}")
        print(f"• Reference frame: {camera_height_info['reference_frame']}")
        print(f"• Height accuracy: ±8.0mm")
    else:
        print("⚠️  Camera Height Information:")
        print("• Camera height: Not available (需要先进行相机标定)")

    print()
    print("Next Steps:")
    print("1. Click 'Validate Ground Calibration' to verify accuracy")
    print("2. Click 'Save Ground Calibration Results' to export")
    print("3. Use this homography matrix and camera height in your measurement system")

if __name__ == "__main__":
    test_ground_calibration_save()
    demonstrate_height_display()
