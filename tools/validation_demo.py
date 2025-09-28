#!/usr/bin/env python3
"""
相机标定验证功能演示脚本

演示如何使用各种验证功能来评估标定质量
"""

import numpy as np
import cv2
import os
import sys
from datetime import datetime

class ValidationDemo:
    """验证功能演示类"""

    def __init__(self):
        self.demo_data_dir = "demo_validation_data"
        self.create_demo_data()

    def create_demo_data(self):
        """创建演示用的标定数据"""
        print("🎯 创建演示标定数据...")

        # 创建演示目录
        os.makedirs(self.demo_data_dir, exist_ok=True)

        # 生成模拟的相机标定参数
        camera_matrix = np.array([
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist_coeffs = np.array([-0.1, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

        # 模拟一些旋转和平移向量
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

        # 保存标定数据
        demo_calibration_path = os.path.join(self.demo_data_dir, "demo_calibration.npz")
        np.savez(demo_calibration_path,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs,
                calibration_date=datetime.now().isoformat(),
                image_size=(1280, 720))

        print(f"✅ 演示标定数据已保存: {demo_calibration_path}")

        # 生成一些测试图像
        self.create_test_images()

        return demo_calibration_path

    def create_test_images(self):
        """创建测试图像"""
        print("📷 生成测试图像...")

        # 创建不同类型的测试图像
        test_images = [
            self.create_checkerboard_image(),
            self.create_line_pattern_image(),
            self.create_realistic_scene_image()
        ]

        for i, img in enumerate(test_images):
            filename = f"test_image_{i+1}.jpg"
            filepath = os.path.join(self.demo_data_dir, filename)
            cv2.imwrite(filepath, img)
            print(f"  ✅ {filename} 已生成")

    def create_checkerboard_image(self):
        """创建棋盘格测试图像"""
        # 创建一个简单的棋盘格图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(255)  # 白色背景

        # 绘制黑白相间的棋盘格
        square_size = 40
        for i in range(0, img.shape[0], square_size):
            for j in range(0, img.shape[1], square_size):
                if (i // square_size + j // square_size) % 2 == 1:
                    img[i:i+square_size, j:j+square_size] = [0, 0, 0]  # 黑色

        return img

    def create_line_pattern_image(self):
        """创建直线图案测试图像"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色背景

        # 绘制水平线和垂直线
        for i in range(0, img.shape[0], 60):
            cv2.line(img, (0, i), (img.shape[1], i), (0, 0, 0), 2)

        for i in range(0, img.shape[1], 60):
            cv2.line(img, (i, 0), (i, img.shape[0]), (0, 0, 0), 2)

        # 添加一些对角线
        cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 0, 0), 3)
        cv2.line(img, (img.shape[1], 0), (0, img.shape[0]), (255, 0, 0), 3)

        return img

    def create_realistic_scene_image(self):
        """创建类似真实场景的测试图像"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 灰色背景

        # 绘制一些几何形状来模拟真实场景
        # 矩形（类似建筑）
        cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), 3)
        cv2.rectangle(img, (100, 100), (150, 130), (0, 0, 0), -1)  # 填充

        # 圆形（类似圆形标志或球体）
        cv2.circle(img, (400, 120), 50, (0, 0, 0), 3)
        cv2.circle(img, (400, 120), 30, (100, 100, 100), -1)

        # 多边形（复杂形状）
        pts = np.array([[500, 200], [550, 180], [580, 220], [520, 250]], np.int32)
        cv2.polylines(img, [pts], True, (0, 0, 0), 3)

        # 添加一些文本
        cv2.putText(img, "CALIBRATION TEST", (250, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return img

    def demonstrate_validation_features(self):
        """演示验证功能"""
        print("\n" + "="*60)
        print("🎯 相机标定验证功能演示")
        print("="*60)

        # 导入验证器（模拟）
        print("\n📋 可用验证功能演示:")
        print("1. 🚀 快速验证 - 基础参数检查")
        print("2. 🎯 高级验证 - 全面质量分析")
        print("3. 📐 畸变矫正 - 视觉质量评估")
        print("4. 📏 内参验证 - 相机矩阵检查")
        print("5. 📍 外参验证 - 位姿参数分析")
        print("6. 🌍 地面标定 - 坐标系验证")
        print("7. ⚡ 性能测试 - 速度和内存分析")
        print("8. 👁️ 可视化验证 - 图像对比分析")

        print("\n📁 演示数据位置:")
        print(f"   标定文件: {os.path.join(self.demo_data_dir, 'demo_calibration.npz')}")
        print(f"   测试图像: {self.demo_data_dir}/test_image_*.jpg")

        print("\n🔍 验证结果示例:")

        # 模拟验证结果显示
        self.show_sample_validation_results()

        print("\n💡 使用建议:")
        print("• 在GUI中运行完整验证功能")
        print("• 使用多种类型的测试图像")
        print("• 定期验证标定质量")
        print("• 保存验证历史记录")

        print("\n✅ 演示完成!")
        print("运行 'python camera_calibration_modern.py' 启动完整GUI验证功能")

    def show_sample_validation_results(self):
        """显示示例验证结果"""
        print("\n" + "-"*50)
        print("📊 示例验证结果:")
        print("-"*50)

        # 模拟综合验证结果
        print("""
🔍 COMPREHENSIVE CALIBRATION VALIDATION REPORT
Validation Summary:
• Validation Type: COMPREHENSIVE
• Overall Quality: EXCELLENT
• Timestamp: 2024-08-29T10:30:00

📏 INTRINSIC PARAMETERS VALIDATION
• Focal Length Check: ✅ PASS
• Principal Point Check: ✅ PASS
• Distortion Check: ✅ PASS
• Matrix Validity: ✅ PASS
• Quality Score: 0.95

📐 DISTORTION CORRECTION ANALYSIS
• Correction Effectiveness: 0.88
• Visual Improvement: 0.85
• Analysis Score: 0.90

🎯 REPROJECTION ERROR ANALYSIS
• Mean Error: 0.23 pixels
• Standard Deviation: 0.15 pixels
• Max Error: 0.67 pixels
• Error Assessment: EXCELLENT
• Analysis Score: 0.95

💡 RECOMMENDATIONS
✅ EXCELLENT CALIBRATION QUALITY!
• Your calibration parameters are optimal
• Ready for high-precision computer vision applications
• Consider using these parameters in production
• Regular validation recommended (monthly)
        """)

        print("\n📐 畸变矫正验证示例:")
        print("""
📐 DISTORTION CORRECTION VALIDATION REPORT
Validation Summary:
• Test Images Analyzed: 3
• Overall Quality Score: 0.87
• Assessment: GOOD

📊 CORRECTION ANALYSIS
Image 1: test_image_1.jpg
• Original Curvature: 0.15
• Corrected Curvature: 0.08
• Improvement Score: 0.73
• Quality Score: 0.80

🎯 RECOMMENDATIONS
Distortion Correction Quality: GOOD
• Use images with straight lines and grids
• Ensure good lighting for corner detection
• Consider different alpha values (0.0-1.0)
        """)

        print("\n👁️ 可视化指标示例:")
        print("""
Visual Correction Metrics:
Original Image Size: 640x480
Corrected Image Size: 636x476
Quality Metrics:
• Mean Pixel Difference: 18.45
• Structural Similarity (SSIM): 0.934
• Correction Effectiveness: EXCELLENT

Assessment:
• ✅ Excellent correction quality!
• Higher SSIM values indicate better correction
• Look for reduced barrel/pincushion distortion
        """)

def main():
    """主函数"""
    print("🚀 启动相机标定验证功能演示...")

    try:
        demo = ValidationDemo()
        demo.demonstrate_validation_features()

    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
