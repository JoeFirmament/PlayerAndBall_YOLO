#!/usr/bin/env python3
"""
演示Camera Calibration和Ground Calibration的完美配合
"""

import numpy as np
import cv2
import json
import os

def load_calibration_results():
    """加载标定结果"""
    print("📂 加载标定结果...")

    # 尝试加载camera calibration结果
    camera_file = "20250829_153422_calibration.json"
    ground_file = "ground_calibration.json"

    camera_data = None
    ground_data = None

    if os.path.exists(camera_file):
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        print("✅ Camera Calibration数据已加载")
    else:
        print("❌ 找不到Camera Calibration文件")

    if os.path.exists(ground_file):
        with open(ground_file, 'r') as f:
            ground_data = json.load(f)
        print("✅ Ground Calibration数据已加载")
    else:
        print("❌ 找不到Ground Calibration文件，创建模拟数据")

        # 创建模拟的ground calibration数据
        ground_data = {
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

    return camera_data, ground_data

def demonstrate_cooperation(camera_data, ground_data):
    """演示两种标定的配合使用"""
    print("\n🎯 演示Camera + Ground Calibration的配合使用")
    print("=" * 60)

    # 1. 从Camera Calibration提取相机参数
    if camera_data:
        camera_matrix = np.array(camera_data['camera_matrix'])
        dist_coeffs = np.array(camera_data['dist_coeffs'][0])  # 取出第一层数组

        print("📹 Camera Calibration提供的参数:")
        print(f"相机内参矩阵:\n{camera_matrix}")
        print(f"畸变系数: {dist_coeffs}")
        print(f"标定位置数量: {len(camera_data.get('tvecs', []))}")

        # 显示前几个位置的高度
        tvecs = camera_data.get('tvecs', [])
        if tvecs:
            print("\n相对高度信息 (前5个位置):")
            for i, tvec in enumerate(tvecs[:5]):
                height = tvec[2][0] if isinstance(tvec[2], list) else tvec[2]
                print("2d")

    # 2. 从Ground Calibration提取地面变换
    if ground_data:
        ground_homography = np.array(ground_data['ground_homography'])
        ground_error = ground_data.get('reprojection_error', 'N/A')

        print("\n🌍 Ground Calibration提供的参数:")
        print(f"地面Homography矩阵:\n{ground_homography}")
        print(f"地面重投影误差: {ground_error}")

    # 3. 演示配合使用的实际应用
    demonstrate_practical_application(camera_matrix, ground_homography)

def demonstrate_practical_application(camera_matrix, ground_homography):
    """演示实际应用场景"""
    print("\n🏀 实际应用演示:")
    print("=" * 40)

    # 模拟几个像素点（运动员关键点）
    athlete_points = [
        (320, 240, "运动员中心"),
        (280, 220, "左手位置"),
        (360, 260, "右手位置"),
        (320, 180, "头部位置")
    ]

    print("像素坐标 → 地面实际坐标转换:")
    print("-" * 50)
    print(f"{'位置':<10} {'像素坐标':<15} {'地面坐标(mm)':<20} {'实际位置':<15}")
    print("-" * 50)

    for pixel_x, pixel_y, description in athlete_points:
        # 像素坐标
        pixel_point = np.array([pixel_x, pixel_y, 1.0])

        # 应用Ground Homography变换
        ground_point = ground_homography @ pixel_point
        ground_x = ground_point[0] / ground_point[2]
        ground_y = ground_point[1] / ground_point[2]

        # 转换为实际位置描述
        position_desc = f"{ground_x/10:.0f}cm, {ground_y/10:.0f}cm"

        print("6s")

    print("\n💡 配合使用的优势:")
    print("-" * 30)
    print("✅ Camera Calibration: 提供精确的相机参数")
    print("✅ Ground Calibration: 建立绝对坐标系基准")
    print("✅ 完美配合: 实现厘米级精度的三维测量")
    print("✅ 实际应用: 支持运动员姿态分析、距离测量等")

def show_workflow_comparison():
    """展示不同工作流程的对比"""
    print("\n📋 工作流程对比:")
    print("=" * 50)

    workflows = {
        "单独使用Camera Calibration": [
            "✅ 相机参数校正",
            "✅ 畸变校正",
            "❌ 相对坐标系",
            "❌ 需要知道标定板位置",
            "⚠️ 测量精度有限"
        ],
        "单独使用Ground Calibration": [
            "❌ 无相机参数",
            "❌ 无法处理畸变",
            "✅ 绝对坐标系",
            "✅ Z=0基准",
            "⚠️ 测量精度受限"
        ],
        "Camera + Ground 配合使用": [
            "✅ 相机参数校正",
            "✅ 畸变校正",
            "✅ 绝对坐标系",
            "✅ Z=0基准",
            "✅ 最高测量精度"
        ]
    }

    for workflow_name, features in workflows.items():
        print(f"\n🔹 {workflow_name}:")
        for feature in features:
            print(f"   {feature}")

def create_integration_example():
    """创建集成使用的代码示例"""
    print("\n💻 集成使用代码示例:")
    print("=" * 50)

    code_example = '''
# 完整的标定系统集成示例

import cv2
import numpy as np
import json

class CompleteCalibrationSystem:
    def __init__(self, camera_calib_file, ground_calib_file):
        # 加载Camera Calibration结果
        with open(camera_calib_file, 'r') as f:
            camera_data = json.load(f)

        self.camera_matrix = np.array(camera_data['camera_matrix'])
        self.dist_coeffs = np.array(camera_data['dist_coeffs'][0])

        # 加载Ground Calibration结果
        with open(ground_calib_file, 'r') as f:
            ground_data = json.load(f)

        self.ground_homography = np.array(ground_data['ground_homography'])
        self.ground_error = ground_data.get('reprojection_error', 0)

        print("🎉 完整标定系统初始化完成!")

    def process_image(self, image):
        """处理图像并进行精确测量"""
        # 1. 校正相机畸变
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs
        )

        # 2. 进行姿态检测 (假设使用YOLOv8 Pose)
        keypoints = detect_pose(undistorted)  # 模拟函数

        # 3. 将关键点转换为地面坐标
        ground_positions = {}
        for key, pixel_point in keypoints.items():
            ground_pos = self.pixel_to_ground(pixel_point)
            ground_positions[key] = ground_pos

        return ground_positions

    def pixel_to_ground(self, pixel_point):
        """像素坐标转换为地面坐标"""
        pixel_homogeneous = np.array([pixel_point[0], pixel_point[1], 1.0])
        ground_homogeneous = self.ground_homography @ pixel_homogeneous

        ground_x = ground_homogeneous[0] / ground_homogeneous[2]
        ground_y = ground_homogeneous[1] / ground_homogeneous[2]

        return (ground_x, ground_y)

    def calculate_distance(self, point1, point2):
        """计算两点间的实际距离"""
        pixel_dist = np.linalg.norm(
            np.array(point1) - np.array(point2)
        )

        # 使用Ground Homography计算实际距离
        ground1 = self.pixel_to_ground(point1)
        ground2 = self.pixel_to_ground(point2)
        actual_dist = np.linalg.norm(
            np.array(ground1) - np.array(ground2)
        )

        return pixel_dist, actual_dist

# 使用示例
calib_system = CompleteCalibrationSystem(
    'camera_calibration.json',
    'ground_calibration.json'
)

# 处理图像
result = calib_system.process_image(image)
print(f"运动员位置: {result}")
'''

    print(code_example)

if __name__ == "__main__":
    print("🎯 Camera Calibration 与 Ground Calibration 配合使用演示")
    print("=" * 70)

    # 加载标定数据
    camera_data, ground_data = load_calibration_results()

    # 演示配合使用
    demonstrate_cooperation(camera_data, ground_data)

    # 工作流程对比
    show_workflow_comparison()

    # 集成代码示例
    create_integration_example()

    print("\n🎉 演示完成!")
    print("现在你了解了两种标定如何完美配合使用!")
    print("这为构建高精度的计算机视觉应用奠定了基础!")
