#!/usr/bin/env python3
"""
独立标定验证测试脚本

演示如何使用 validate_calibration_from_file 方法
直接加载npz文件和图片进行标定验证
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera_calibration_modern import ModernCalibrationGUI
import json

def test_single_validation():
    """测试单张图片验证功能"""
    print("=" * 60)
    print("单张图片标定验证测试")
    print("=" * 60)

    # 创建验证器实例
    validator = ModernCalibrationGUI()

    # 测试参数
    npz_path = "example_calibration.npz"  # 标定结果文件
    image_path = "test_calibration.jpg"   # 测试图片

    # 检查文件是否存在
    if not os.path.exists(npz_path):
        print(f"❌ 标定文件不存在: {npz_path}")
        print("请先运行标定过程生成npz文件")
        return

    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        print("请准备一张包含棋盘格的测试图片")
        return

    print(f"📁 NPZ文件: {npz_path}")
    print(f"🖼️  测试图片: {image_path}")
    print()

    # 执行验证
    print("🔍 开始验证...")
    result = validator.validate_calibration_from_file(
        npz_path=npz_path,
        image_path=image_path,
        board_size=(7, 6),    # 7x6内角点
        square_size=25.0      # 25mm方格
    )

    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    if result['success']:
        print("✅ 验证成功!")
        print(f"📊 平均重投影误差: {result['mean_error']:.4f} 像素")
        print(f"📈 最大误差: {result['max_error']:.4f} 像素")
        print(f"📉 最小误差: {result['min_error']:.4f} 像素")
        print(f"🎯 检测到角点: {result['corners_found']} 个")
        print(f"🏆 质量评估: {result['quality_assessment']}")

        # 详细分析
        print("\n📋 详细分析:")
        print(f"   • 棋盘格尺寸: {result['board_size'][0]}×{result['board_size'][1]}")
        print(f"   • 方格尺寸: {result['square_size']} mm")
        print(f"   • 误差范围: {result['min_error']:.2f} - {result['max_error']:.2f} 像素")

        # 质量判断
        mean_error = result['mean_error']
        if mean_error < 0.5:
            print("   ✅ 优秀: 适合高精度应用")
        elif mean_error < 1.0:
            print("   ✅ 良好: 适合大多数计算机视觉应用")
        elif mean_error < 2.0:
            print("   ⚠️  可接受: 谨慎使用")
        else:
            print("   ❌ 较差: 建议重新标定")

    else:
        print("❌ 验证失败!")
        print(f"错误信息: {result['error']}")

    print("\n" + "=" * 60)

    return result

def save_result_to_json(result, output_path="validation_result.json"):
    """将验证结果保存为JSON文件"""
    try:
        # 转换numpy数组为列表以便JSON序列化
        json_result = result.copy()

        if 'camera_matrix' in json_result:
            json_result['camera_matrix'] = json_result['camera_matrix'].tolist()
        if 'dist_coeffs' in json_result:
            json_result['dist_coeffs'] = json_result['dist_coeffs'].tolist()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        print(f"📄 结果已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 保存结果失败: {e}")

def main():
    """主函数"""
    try:
        # 执行验证测试
        result = test_single_validation()

        if result and result.get('success', False):
            # 保存结果
            save_result_to_json(result)

            print("\n🎉 测试完成!")
            print("您可以使用这个验证功能来:")
            print("• 验证现有标定结果的质量")
            print("• 测试不同图片的标定效果")
            print("• 比较不同标定参数的效果")
        else:
            print("\n⚠️  测试未完成，请检查文件和参数")

    except KeyboardInterrupt:
        print("\n\n👋 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
