#!/usr/bin/env python3
"""
演示Ground Calibration的工作原理和优势
"""

import numpy as np
import cv2

def demonstrate_ground_calibration_concept():
    """演示ground calibration的基本概念"""
    print("🎯 Ground Calibration 工作原理演示")
    print("=" * 60)

    print("\n📹 Camera Calibration vs Ground Calibration:")
    print("-" * 50)
    print("Camera Calibration:")
    print("  • 相机内参: [fx, fy, cx, cy]")
    print("  • 畸变系数: [k1, k2, p1, p2, k3]")
    print("  • 相对位置: tvecs (相对于标定板)")
    print("  • 高度信息: 相对标定板的高度")

    print("\n🌍 Ground Calibration:")
    print("  • Homography矩阵: 3x3变换矩阵")
    print("  • Z=0基准: 实际地面")
    print("  • 坐标转换: 像素 → 地面实际坐标")
    print("  • 高度信息: 绝对地面高度")

    print("\n🏀 实际应用示例:")
    print("-" * 30)

    # 模拟一个ground homography矩阵
    # 这个矩阵将图像坐标转换为地面坐标
    H = np.array([
        [0.5, 0.0, 1000],    # X缩放和平移
        [0.0, 0.5, 1500],    # Y缩放和平移
        [0.0, 0.0, 1.0]      # 齐次坐标
    ])

    # 测试几个像素点
    test_points = [
        (320, 240, "图像中心点"),
        (0, 0, "图像左上角"),
        (640, 480, "图像右下角"),
        (320, 0, "图像顶部中心")
    ]

    print("像素坐标 → 地面坐标转换:")
    print("-" * 40)

    for pixel_x, pixel_y, description in test_points:
        # 将像素坐标转换为齐次坐标
        pixel_homogeneous = np.array([pixel_x, pixel_y, 1.0])

        # 应用homography变换
        ground_homogeneous = H @ pixel_homogeneous

        # 转换为实际坐标 (毫米)
        ground_x = ground_homogeneous[0] / ground_homogeneous[2]
        ground_y = ground_homogeneous[1] / ground_homogeneous[2]

        print("10s"
              "6.0f"
              "6.0f")

    print("\n💡 Ground Calibration的优势:")
    print("-" * 30)
    print("✅ 建立绝对坐标系基准")
    print("✅ Z=0对应实际地面")
    print("✅ 像素到实际距离的精确转换")
    print("✅ 适合篮球场等精确测量应用")

def compare_calibration_methods():
    """比较两种标定方法的区别"""
    print("\n🔍 标定方法对比")
    print("=" * 50)

    print("\n📊 特征对比表:")
    print("-" * 70)
    print("<12")
    print("-" * 70)
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("<12")
    print("-" * 70)

    print("\n🎯 推荐使用场景:")
    print("-" * 30)
    print("Camera Calibration:")
    print("  • 需要校正镜头畸变")
    print("  • 需要精确的相机内参")
    print("  • 基础的姿态估计")

    print("\nGround Calibration:")
    print("  • 篮球场位置测量")
    print("  • 运动员轨迹分析")
    print("  • 投篮角度计算")
    print("  • 精确距离测量")

def show_ground_calibration_workflow():
    """展示ground calibration的工作流程"""
    print("\n🚀 Ground Calibration 工作流程")
    print("=" * 50)

    workflow = [
        "1. 📸 准备图像: 棋盘格平放在地面上拍摄",
        "2. 🔍 检测角点: 自动检测棋盘格角点位置",
        "3. 📐 计算Homography: 像素坐标→地面坐标变换",
        "4. ✅ 验证精度: 检查重投影误差",
        "5. 💾 保存结果: 生成ground_calibration.json"
    ]

    for step in workflow:
        print(f"  {step}")

    print("\n📁 输出文件内容:")
    print("-" * 20)
    print("""
{
  "ground_homography": [
    [0.5, 0.0, 1000.0],
    [0.0, 0.5, 1500.0],
    [0.0, 0.0, 1.0]
  ],
  "reprojection_error": 0.8,
  "board_params": {
    "size": [9, 6],
    "square_size": 25.0
  }
}
    """)

if __name__ == "__main__":
    demonstrate_ground_calibration_concept()
    compare_calibration_methods()
    show_ground_calibration_workflow()

    print("\n🎉 总结:")
    print("-" * 10)
    print("✅ Ground Calibration 确实能提供相机Z轴的绝对高度信息!")
    print("✅ 它建立Z=0的精确基准，对应实际地面")
    print("✅ 与Camera Calibration配合使用效果最佳")
    print("✅ 适合需要精确地面坐标的应用场景")
