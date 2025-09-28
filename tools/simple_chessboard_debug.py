#!/usr/bin/env python3
"""
简化棋盘格检测调试工具 - 不依赖matplotlib
"""
import cv2
import numpy as np
import os

def debug_chessboard_simple(image_folder, board_size=(9, 6)):
    """简化的棋盘格检测调试"""
    print("🔍 开始调试棋盘格检测...")
    print(f"📂 图片文件夹: {image_folder}")
    print(f"📏 棋盘格尺寸: {board_size[0]}x{board_size[1]}")
    
    # 获取所有jpg文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    print(f"📄 找到 {len(image_files)} 个JPG文件")
    
    detection_results = []
    
    for i, filename in enumerate(image_files[:10], 1):  # 检查前10个文件
        image_path = os.path.join(image_folder, filename)
        print(f"\n🖼️ 检查图片 {i}: {filename}")
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"   ❌ 无法读取图像: {filename}")
            continue
            
        print(f"   📐 图像尺寸: {img.shape[1]}x{img.shape[0]}")
        
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 基本图像统计
        print(f"   📊 图像统计:")
        print(f"      平均亮度: {gray.mean():.1f}")
        print(f"      对比度(标准差): {gray.std():.1f}")
        print(f"      最小值: {gray.min()}")
        print(f"      最大值: {gray.max()}")
        
        # 尝试不同的检测方法
        methods = [
            ("标准", None),
            ("自适应阈值", cv2.CALIB_CB_ADAPTIVE_THRESH),
            ("归一化", cv2.CALIB_CB_NORMALIZE_IMAGE),
            ("组合标志", cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)
        ]
        
        found_any = False
        for method_name, flags in methods:
            print(f"   🔍 尝试 {method_name} 检测...")
            ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
            
            if ret:
                print(f"   ✅ {method_name} 找到棋盘格! 角点数量: {len(corners)}")
                found_any = True
                
                # 保存检测成功的图片
                img_with_corners = cv2.drawChessboardCorners(img.copy(), board_size, corners, ret)
                output_path = os.path.join(image_folder, f"detected_{filename}")
                cv2.imwrite(output_path, img_with_corners)
                print(f"   💾 已保存检测结果到: detected_{filename}")
                
                detection_results.append({
                    'filename': filename,
                    'method': method_name,
                    'detected': True,
                    'corners_count': len(corners)
                })
                break
            else:
                print(f"   ❌ {method_name} 未找到棋盘格")
        
        if not found_any:
            detection_results.append({
                'filename': filename,
                'detected': False,
                'corners_count': 0
            })
    
    # 显示结果统计
    print("\n📊 检测结果统计:")
    detected_count = sum(1 for r in detection_results if r['detected'])
    total_count = len(detection_results)
    print(f"• 检测成功: {detected_count}/{total_count}")
    print(f"• 成功率: {detected_count/total_count*100:.1f}%")
    
    if detected_count == 0:
        print("\n❌ 没有检测到任何棋盘格")
        print("可能的问题:")
        print("1. 图片中没有棋盘格或棋盘格不清晰")
        print("2. 棋盘格尺寸设置不正确")
        print("3. 图像质量太差或对比度不足")
        print("4. 视角角度太大")
        print("5. 光照条件不佳")
    else:
        print(f"\n✅ 成功检测到 {detected_count} 张图片中的棋盘格")
        for result in detection_results:
            if result['detected']:
                print(f"   • {result['filename']}: {result['method']} 方法, {result['corners_count']} 角点")
    
    return detection_results

def test_different_board_sizes(image_path, board_sizes=[(9,6), (8,6), (7,7), (6,6), (10,7), (6,9), (7,5)]):
    """测试不同的棋盘格尺寸"""
    print(f"\n🔧 测试不同棋盘格尺寸: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("   ❌ 无法读取图像")
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用组合标志进行检测
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
    
    for board_size in board_sizes:
        print(f"   测试尺寸 {board_size[0]}x{board_size[1]}...")
        ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if ret:
            print(f"   ✅ 找到棋盘格! 角点数量: {len(corners)}")
            return board_size
        else:
            print(f"   ❌ 未找到")
    
    print("   ❌ 所有尺寸都无法检测到棋盘格")
    return None

def analyze_image_quality(image_path):
    """分析图像质量"""
    print(f"\n🔬 分析图像质量: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("   ❌ 无法读取图像")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算图像质量指标
    mean_brightness = gray.mean()
    contrast = gray.std()
    min_val = gray.min()
    max_val = gray.max()
    
    # 计算拉普拉斯方差（模糊检测）
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f"   📊 质量指标:")
    print(f"      平均亮度: {mean_brightness:.1f} (理想值: 100-150)")
    print(f"      对比度: {contrast:.1f} (理想值: >30)")
    print(f"      动态范围: {min_val}-{max_val} (理想值: 接近 0-255)")
    print(f"      清晰度: {laplacian_var:.1f} (理想值: >100)")
    
    # 质量评估
    if mean_brightness < 80:
        print("   ⚠️ 图像偏暗，建议增加光照")
    elif mean_brightness > 180:
        print("   ⚠️ 图像偏亮，建议减少光照或调整曝光")
    
    if contrast < 25:
        print("   ⚠️ 对比度不足，建议改善光照条件")
    
    if laplacian_var < 100:
        print("   ⚠️ 图像可能模糊，建议检查焦点")
    
    if max_val - min_val < 150:
        print("   ⚠️ 动态范围不足，建议调整拍摄条件")

if __name__ == "__main__":
    # 设置图片文件夹路径
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"
    
    if not os.path.exists(image_folder):
        print(f"❌ 文件夹不存在: {image_folder}")
        exit(1)
    
    # 1. 调试棋盘格检测
    results = debug_chessboard_simple(image_folder)
    
    # 2. 如果没有检测到，测试不同的棋盘格尺寸
    if not any(r['detected'] for r in results):
        print("\n🔄 尝试不同的棋盘格尺寸...")
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        if image_files:
            # 选择一个中间的图片进行测试
            test_image = os.path.join(image_folder, image_files[len(image_files)//2])
            print(f"🔍 使用测试图片: {os.path.basename(test_image)}")
            
            # 分析图像质量
            analyze_image_quality(test_image)
            
            # 测试不同尺寸
            correct_size = test_different_board_sizes(test_image)
            if correct_size:
                print(f"\n✅ 建议使用棋盘格尺寸: {correct_size[0]}x{correct_size[1]}")
                print("🔄 使用建议尺寸重新检测...")
                # 用找到的正确尺寸重新检测
                results_new = debug_chessboard_simple(image_folder, correct_size)
            else:
                print("\n❌ 无法确定正确的棋盘格尺寸")
                print("建议:")
                print("1. 检查图片中是否有清晰的棋盘格")
                print("2. 尝试改善拍摄条件（光照、角度、距离）")
                print("3. 使用更大尺寸的棋盘格")
                print("4. 确保棋盘格边缘清晰可见")
    else:
        print("\n✅ 检测成功！建议查看保存的检测结果图片")