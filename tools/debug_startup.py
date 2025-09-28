#!/usr/bin/env python3
"""
调试应用启动问题
"""

import sys
import traceback
import os

def debug_startup():
    """逐步调试启动过程"""
    print("🔍 调试应用启动问题")
    print("=" * 50)

    try:
        print("📦 步骤1: 导入基础模块...")

        # 测试tkinter
        print("  - 导入tkinter...")
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        print("  ✅ tkinter导入成功")

        # 测试OpenCV
        print("  - 导入OpenCV...")
        import cv2
        print(f"  ✅ OpenCV导入成功 (版本: {cv2.__version__})")

        # 测试NumPy
        print("  - 导入NumPy...")
        import numpy as np
        print(f"  ✅ NumPy导入成功 (版本: {np.__version__})")

        # 测试PIL
        print("  - 导入PIL...")
        from PIL import Image, ImageTk
        print("  ✅ PIL导入成功")

        print("\n📂 步骤2: 检查文件路径...")
        current_dir = os.getcwd()
        print(f"  - 当前目录: {current_dir}")

        # 检查主要文件是否存在
        main_file = "camera_calibration_modern.py"
        if os.path.exists(main_file):
            print(f"  ✅ 主文件存在: {main_file}")
            file_size = os.path.getsize(main_file)
            print(f"     文件大小: {file_size:,} bytes")
        else:
            print(f"  ❌ 主文件不存在: {main_file}")

        # 检查CalibrationFileManager
        cfm_file = "calibration_file_manager.py"
        if os.path.exists(cfm_file):
            print(f"  ✅ 文件管理器存在: {cfm_file}")
        else:
            print(f"  ⚠️  文件管理器不存在: {cfm_file}")

        print("\n🏗️  步骤3: 测试类实例化...")

        # 导入主类
        print("  - 导入主类...")
        from camera_calibration_modern import ModernCalibrationGUI
        print("  ✅ 主类导入成功")

        print("  - 创建GUI实例...")
        app = ModernCalibrationGUI()
        print("  ✅ GUI实例创建成功")

        print("\n🎯 步骤4: 测试运行...")
        print("  - 准备运行应用...")
        # 这里我们不实际运行主循环，只是测试到实例化完成
        print("  ✅ 应用准备完成")

        print("\n🎉 调试完成!")
        print("✅ 所有导入和实例化都成功")
        print("✅ 问题可能在run()方法或主循环中")

        return True

    except Exception as e:
        print(f"\n❌ 调试过程中发现错误:")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        print("\n🔍 详细错误信息:")
        traceback.print_exc()
        return False

def test_specific_imports():
    """测试可能有问题的特定导入"""
    print("\n🔧 测试特定导入...")

    try:
        # 测试atexit
        import atexit
        print("✅ atexit导入成功")

        # 测试threading (如果使用)
        import threading
        print("✅ threading导入成功")

        # 测试time
        import time
        print("✅ time导入成功")

        # 测试pathlib
        from pathlib import Path
        print("✅ pathlib导入成功")

        # 测试json
        import json
        print("✅ json导入成功")

        # 测试math
        import math
        print("✅ math导入成功")

    except Exception as e:
        print(f"❌ 特定导入失败: {e}")
        return False

    return True

def check_tkinter_display():
    """检查tkinter显示环境"""
    print("\n🖥️  检查tkinter显示环境...")

    try:
        import tkinter as tk

        # 测试基本的tkinter功能
        root = tk.Tk()
        root.title("Test Window")
        root.geometry("200x100")

        # 创建一个简单的标签
        label = tk.Label(root, text="Test")
        label.pack()

        # 测试tkinter版本
        print(f"✅ tkinter版本: {tk.TkVersion}")

        # 测试Tcl版本
        print(f"✅ Tcl版本: {tk.TclVersion}")

        # 销毁测试窗口
        root.destroy()

        print("✅ tkinter显示环境正常")
        return True

    except Exception as e:
        print(f"❌ tkinter显示环境问题: {e}")
        print("💡 可能的解决方案:")
        print("   1. 检查是否有图形界面环境 (DISPLAY变量)")
        print("   2. 如果是SSH连接，确保使用-X或-Y参数")
        print("   3. 尝试设置环境变量: export DISPLAY=:0")
        return False

def main():
    """主调试函数"""
    print("🚀 应用启动问题调试工具")
    print("=" * 60)

    # 记录环境信息
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {sys.platform}")
    print(f"工作目录: {os.getcwd()}")

    # 检查DISPLAY环境变量
    display = os.environ.get('DISPLAY', 'Not set')
    print(f"DISPLAY环境变量: {display}")

    # 执行调试步骤
    success = debug_startup()

    if success:
        # 如果基本调试成功，进一步测试
        test_specific_imports()
        check_tkinter_display()
    else:
        print("\n❌ 基本调试失败，建议检查错误信息")

    print("\n" + "=" * 60)
    print("🔚 调试完成")

if __name__ == "__main__":
    main()
