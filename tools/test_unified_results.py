#!/usr/bin/env python3
"""
测试统一结果窗口功能
"""

import tkinter as tk
from tkinter import ttk
import sys
import os

# 确保能够导入主程序
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from camera_calibration_modern import CameraCalibrationStudio
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def test_unified_results():
    """测试统一结果窗口"""
    print("🧪 开始测试统一结果窗口...")
    
    # 创建应用实例
    app = CameraCalibrationStudio()
    
    # 测试添加不同类型的消息
    test_messages = [
        ("Camera calibration initialized successfully", "SUCCESS"),
        ("Loading calibration images from folder...", "INFO"),
        ("Warning: No calibration data found", "WARNING"),
        ("Error: Failed to open camera device", "ERROR"),
        ("Debug: Processing image 1/20", "DEBUG")
    ]
    
    # 添加测试消息
    print("📝 添加测试消息...")
    for message, category in test_messages:
        app.add_result_message(message, category)
        print(f"  ✅ 添加 {category}: {message[:50]}...")
    
    # 测试清空功能
    def test_clear():
        print("🧹 测试清空功能...")
        app.clear_unified_results()
        app.add_result_message("Results cleared - test completed!", "INFO")
    
    # 添加测试按钮
    test_frame = tk.Frame(app.root, bg='#f0f0f0')
    test_frame.pack(fill='x', padx=10, pady=5)
    
    ttk.Label(test_frame, text="🧪 测试控制面板:", 
             font=('TkDefaultFont', 10, 'bold')).pack(side='left')
    
    ttk.Button(test_frame, text="Clear Results", 
              command=test_clear).pack(side='right', padx=(0, 5))
    
    ttk.Button(test_frame, text="Add Test Messages", 
              command=lambda: [app.add_result_message(msg, cat) for msg, cat in test_messages]).pack(side='right', padx=(0, 5))
    
    print("✅ 测试设置完成")
    print("🚀 启动GUI应用...")
    print("📌 测试内容:")
    print("   - 统一结果窗口应该显示在底部")
    print("   - 可以添加不同类型的消息 (SUCCESS/INFO/WARNING/ERROR/DEBUG)")  
    print("   - 可以清空结果窗口")
    print("   - 消息应该有时间戳和图标")
    print("   - 应该自动滚动到最新消息")
    print()
    print("按Ctrl+C或关闭窗口来退出测试")
    
    # 启动应用
    app.run()

if __name__ == "__main__":
    test_unified_results()