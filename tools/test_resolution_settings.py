#!/usr/bin/env python3
"""
测试相机分辨率设置功能
"""

import cv2
import sys

def test_camera_resolution(device_id=0, width=1280, height=720):
    """测试相机分辨率设置"""
    print(f"🧪 测试相机分辨率设置...")
    print(f"📹 设备: {device_id}")
    print(f"📐 请求分辨率: {width}×{height}")

    try:
        # 打开相机
        cap = cv2.VideoCapture(device_id)

        if not cap.isOpened():
            print(f"❌ 无法打开相机 {device_id}")
            return False

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 读取一帧
        ret, frame = cap.read()

        if ret:
            actual_height, actual_width = frame.shape[:2]
            print(f"✅ 相机工作正常!")
            print(f"📊 实际分辨率: {actual_width}×{actual_height}")

            if actual_width == width and actual_height == height:
                print("🎉 分辨率设置成功!")
                return True
            else:
                print(f"⚠️ 相机返回不同分辨率")
                print(f"   可能不支持 {width}×{height}")
                return True
        else:
            print("❌ 无法读取相机帧")
            return False

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        if 'cap' in locals():
            cap.release()

if __name__ == "__main__":
    print("📷 相机分辨率设置测试")
    print("=" * 50)

    # 测试默认设置
    print("\n1️⃣ 测试默认分辨率 (640×480)...")
    test_camera_resolution(0, 640, 480)

    # 测试HD分辨率
    print("\n2️⃣ 测试HD分辨率 (1280×720)...")
    test_camera_resolution(0, 1280, 720)

    # 测试更高分辨率
    print("\n3️⃣ 测试更高分辨率 (1920×1080)...")
    test_camera_resolution(0, 1920, 1080)

    print("\n" + "=" * 50)
    print("💡 测试完成!")
    print("\n📝 在GUI中:")
    print("1. 进入 Settings 标签页")
    print("2. 设置相机分辨率")
    print("3. 点击 'Apply Settings'")
    print("4. 点击 'Test Camera' 验证")
    print("\n🎯 只有点击 'Apply Settings' 后，设置才会生效!")
