#!/usr/bin/env python3
"""
测试相机拍摄功能
"""

import os
import sys
from datetime import datetime

def test_camera_capture_basic(device_id=0, width=1280, height=720):
    """测试基本的相机拍摄功能"""
    print("🧪 测试相机拍摄功能...")
    print(f"📹 设备: {device_id}")
    print(f"📐 分辨率: {width}×{height}")

    try:
        import cv2

        # 连接相机
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"❌ 无法打开相机 {device_id}")
            return False

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 测试拍摄
        ret, frame = cap.read()
        if ret:
            actual_height, actual_width = frame.shape[:2]
            print(f"✅ 相机工作正常!")
            print(f"📊 实际分辨率: {actual_width}×{actual_height}")

            # 创建测试保存目录
            test_dir = "./test_captures"
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            # 保存测试图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_capture_{timestamp}.jpg"
            filepath = os.path.join(test_dir, filename)

            cv2.imwrite(filepath, frame)
            print(f"✅ 图像已保存: {filepath}")

            cap.release()
            return True
        else:
            print("❌ 无法读取相机帧")
            cap.release()
            return False

    except ImportError:
        print("❌ 缺少OpenCV模块")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("📷 相机拍摄功能测试")
    print("=" * 50)

    # 测试基本拍摄功能
    print("\n1️⃣ 测试基本拍摄功能...")
    success = test_camera_capture_basic()

    if success:
        print("\n" + "=" * 50)
        print("✅ 相机拍摄功能测试通过!")
        print("\n📝 在GUI中测试:")
        print("1. 启动 camera_calibration_modern.py")
        print("2. 切换到 '📷 Camera' 标签页")
        print("3. 点击 'Connect Camera' 连接相机")
        print("4. 点击 'Start Preview' 开始预览")
        print("5. 点击 '📷 Capture Single' 拍摄单张")
        print("6. 或设置批量参数后点击 '🔄 Capture Multiple'")
        print("\n🎯 现在你可以享受完整的相机拍摄功能了!")
    else:
        print("\n❌ 相机拍摄功能测试失败")
        print("请检查相机连接和OpenCV安装")
