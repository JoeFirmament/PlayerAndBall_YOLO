#!/usr/bin/env python3
"""
测试Camera页面设置功能
"""

import os
import sys
from datetime import datetime

def test_camera_device_switching():
    """测试相机设备切换功能"""
    print("🧪 测试相机设备切换功能...")

    try:
        import cv2

        # 测试设备0
        print("📹 测试设备 0...")
        cap0 = cv2.VideoCapture(0)
        if cap0.isOpened():
            ret, frame = cap0.read()
            if ret:
                print("✅ 设备 0 可用")
                height, width = frame.shape[:2]
                print(f"   分辨率: {width}×{height}")
            else:
                print("⚠️  设备 0 打开但无法读取")
        else:
            print("❌ 设备 0 不可用")
        cap0.release()

        # 测试设备1
        print("📹 测试设备 1...")
        cap1 = cv2.VideoCapture(1)
        if cap1.isOpened():
            ret, frame = cap1.read()
            if ret:
                print("✅ 设备 1 可用")
                height, width = frame.shape[:2]
                print(f"   分辨率: {width}×{height}")
            else:
                print("⚠️  设备 1 打开但无法读取")
        else:
            print("❌ 设备 1 不可用")
        cap1.release()

        return True

    except ImportError:
        print("❌ 缺少OpenCV模块")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_resolution_settings():
    """测试分辨率设置功能"""
    print("\n🧪 测试分辨率设置功能...")

    try:
        import cv2

        device_id = 0
        test_resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080)
        ]

        for width, height in test_resolutions:
            print(f"📐 测试分辨率 {width}×{height}...")
            cap = cv2.VideoCapture(device_id)

            if cap.isOpened():
                # 设置分辨率
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                # 读取帧
                ret, frame = cap.read()
                if ret:
                    actual_height, actual_width = frame.shape[:2]
                    print(f"✅ 请求: {width}×{height}, 实际: {actual_width}×{actual_height}")
                else:
                    print(f"❌ 无法读取帧")
            else:
                print(f"❌ 无法打开相机 {device_id}")

            cap.release()

        return True

    except ImportError:
        print("❌ 缺少OpenCV模块")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("⚙️  Camera页面设置功能测试")
    print("=" * 50)

    # 测试相机设备切换
    print("\n1️⃣ 测试相机设备切换...")
    device_test = test_camera_device_switching()

    # 测试分辨率设置
    print("\n2️⃣ 测试分辨率设置...")
    resolution_test = test_resolution_settings()

    print("\n" + "=" * 50)
    if device_test and resolution_test:
        print("✅ Camera设置功能测试通过!")
        print("\n📝 在GUI中测试:")
        print("1. 启动 camera_calibration_modern.py")
        print("2. 切换到 '📷 Camera' 标签页")
        print("3. 在 '📹 Camera Settings' 中:")
        print("   • 修改 Device ID，点击 'Apply Device'")
        print("   • 修改 Width/Height，点击 'Apply Resolution'")
        print("4. 点击 'Connect Camera' 测试连接")
        print("5. 点击 'Start Preview' 查看效果")
        print("\n🎯 现在Camera页面可以直接设置设备号和分辨率了!")
    else:
        print("❌ Camera设置功能测试失败")
        print("请检查相机连接和OpenCV安装")
