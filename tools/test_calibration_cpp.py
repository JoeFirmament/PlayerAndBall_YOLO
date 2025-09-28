#!/usr/bin/env python3
"""
C++相机标定工具测试脚本

此脚本用于测试C++工具是否能正确读取Python生成的标定文件。
"""

import os
import sys
import subprocess
import numpy as np
from pathlib import Path

def log_info(message):
    print(f"[INFO] {message}")

def log_success(message):
    print(f"[SUCCESS] {message}")

def log_error(message):
    print(f"[ERROR] {message}")

def log_warning(message):
    print(f"[WARNING] {message}")

def check_files():
    """检查必要的文件是否存在"""
    required_files = [
        "camera_calibration.npz",
        "ground_calibration.npz",
        "calibration_example"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)

    if missing_files:
        log_error(f"Missing required files: {', '.join(missing_files)}")
        log_info("Please ensure:")
        log_info("1. Python calibration tool has generated .npz files")
        log_info("2. C++ tools have been built successfully")
        return False

    log_success("All required files found.")
    return True

def create_test_image():
    """创建一个简单的测试图像"""
    import cv2

    # 创建一个简单的测试图像（棋盘格图案）
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img.fill(255)  # 白色背景

    # 绘制简单的棋盘格
    square_size = 50
    for i in range(0, 640, square_size):
        for j in range(0, 480, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                cv2.rectangle(img, (i, j), (i + square_size, j + square_size), (0, 0, 0), -1)

    cv2.imwrite("test_calibration.jpg", img)
    log_success("Test image created: test_calibration.jpg")
    return "test_calibration.jpg"

def run_cpp_test():
    """运行C++测试"""
    log_info("Running C++ calibration example...")

    cmd = [
        "./calibration_example",
        "camera_calibration.npz",
        "ground_calibration.npz",
        "test_calibration.jpg"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            log_success("C++ test completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            log_error("C++ test failed!")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            return False

    except subprocess.TimeoutExpired:
        log_error("C++ test timed out!")
        return False
    except FileNotFoundError:
        log_error("C++ executable not found!")
        return False
    except Exception as e:
        log_error(f"C++ test failed with exception: {e}")
        return False

def validate_npz_content():
    """验证npz文件内容"""
    log_info("Validating npz file contents...")

    try:
        # 检查相机标定文件
        camera_data = np.load("camera_calibration.npz")
        required_keys = ["camera_matrix", "dist_coeffs"]

        for key in required_keys:
            if key not in camera_data:
                log_error(f"Missing key in camera_calibration.npz: {key}")
                return False

        camera_matrix = camera_data["camera_matrix"]
        dist_coeffs = camera_data["dist_coeffs"]

        log_success(f"Camera matrix shape: {camera_matrix.shape}")
        log_success(f"Distortion coefficients shape: {dist_coeffs.shape}")

        # 检查地面标定文件
        ground_data = np.load("ground_calibration.npz")
        if "ground_homography" not in ground_data:
            log_error("Missing key in ground_calibration.npz: ground_homography")
            return False

        homography = ground_data["ground_homography"]
        log_success(f"Ground homography shape: {homography.shape}")

        return True

    except Exception as e:
        log_error(f"Failed to validate npz files: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("C++ Camera Calibration Tools Test")
    print("=" * 60)

    # 检查文件
    if not check_files():
        return 1

    # 验证npz文件内容
    if not validate_npz_content():
        return 1

    # 创建测试图像
    test_image = create_test_image()

    # 运行C++测试
    if run_cpp_test():
        log_success("All tests passed!")
        return 0
    else:
        log_error("Some tests failed!")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log_warning("Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)
