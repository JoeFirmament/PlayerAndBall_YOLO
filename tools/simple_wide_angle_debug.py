#!/usr/bin/env python3
"""
简化的广角摄像头棋盘格检测调试工具
"""
import cv2
import numpy as np
import os

def debug_wide_angle_chessboard(image_path, board_size=(9, 6)):
    """广角摄像头棋盘格检测调试"""
    print(f"🔍 调试广角摄像头棋盘格检测: {os.path.basename(image_path)}")
    print(f"📏 棋盘格尺寸: {board_size[0]}x{board_size[1]}")

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图像")
        return False

    print(f"📐 图像尺寸: {img.shape[1]}x{img.shape[0]}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 显示图像统计信息
    print("📊 图像统计:")
    print(f"   • 平均亮度: {gray.mean():.1f}")
    print(f"   • 对比度: {gray.std():.1f}")
    print(f"   • 像素范围: {gray.min()} - {gray.max()}")

    # 尝试多种检测参数组合
    detection_configs = [
        {
            'name': '标准检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
        },
        {
            'name': '宽松检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH
        },
        {
            'name': '快速检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK
        },
        {
            'name': '无过滤检测',
            'flags': cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        }
    ]

    # 尝试不同的棋盘格尺寸 - 从小到大
    board_sizes_to_try = [
        (5, 4), (6, 4), (7, 5), (6, 5), (8, 6), (9, 6),
        (7, 6), (6, 6), (5, 5), (4, 3), (3, 3)
    ]

    best_result = None
    max_corners = 0

    print("\n🔍 开始检测...")
    for size in board_sizes_to_try:
        if size[0] < 3 or size[1] < 3:
            continue

        print(f"\n📏 测试尺寸 {size[0]}x{size[1]}:")

        for config in detection_configs:
            ret, corners = cv2.findChessboardCorners(gray, size, config['flags'])

            if ret:
                corners_count = len(corners)
                print(f"   ✅ {config['name']}: 找到 {corners_count} 个角点")

                if corners_count > max_corners:
                    max_corners = corners_count
                    best_result = {
                        'size': size,
                        'config': config['name'],
                        'corners': corners,
                        'corners_count': corners_count
                    }

                # 如果找到了足够多的角点，就停止测试
                expected_corners = size[0] * size[1]
                if corners_count >= expected_corners * 0.8:  # 80%的角点
                    print(f"   🎯 检测质量良好 ({corners_count}/{expected_corners})")
                    break
            else:
                print(f"   ❌ {config['name']}: 未找到棋盘格")

        # 如果找到了好的结果，就停止测试其他尺寸
        if best_result and best_result['corners_count'] >= size[0] * size[1] * 0.7:
            break

    # 显示最佳结果
    if best_result:
        print("\n🏆 最佳检测结果:")
        print(f"   • 棋盘格尺寸: {best_result['size'][0]}x{best_result['size'][1]}")
        print(f"   • 检测方法: {best_result['config']}")
        print(f"   • 角点数量: {best_result['corners_count']}")
        expected = best_result['size'][0] * best_result['size'][1]
        print(f"   • 检测率: {best_result['corners_count']/expected*100:.1f}%")

        # 细化角点并显示坐标范围
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, best_result['corners'], (11, 11), (-1, -1), criteria)

        corners_array = np.array(corners_refined)
        x_coords = corners_array[:, 0, 0]
        y_coords = corners_array[:, 0, 1]

        print(f"   • X坐标范围: {x_coords.min():.1f} - {x_coords.max():.1f}")
        print(f"   • Y坐标范围: {y_coords.min():.1f} - {y_coords.max():.1f}")

        # 在图像上绘制角点
        img_with_corners = cv2.drawChessboardCorners(img.copy(), best_result['size'], corners_refined, True)

        # 保存结果图像
        output_path = image_path.replace('.jpg', '_chessboard_detected.jpg')
        cv2.imwrite(output_path, img_with_corners)
        print(f"   • 结果已保存: {output_path}")

        return True, best_result
    else:
        print("\n❌ 未找到任何棋盘格")
        print("💡 建议:")
        print("   1. 检查棋盘格是否清晰可见")
        print("   2. 尝试更小的棋盘格尺寸")
        print("   3. 改善光照条件")
        print("   4. 调整摄像头角度")
        print("   5. 减少畸变（如果可能）")
        return False, None

def analyze_single_image_detailed(image_path):
    """详细分析单张图片"""
    print(f"\n🔬 详细分析图片: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print("❌ 无法读取图像")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 基础图像分析
    print("📊 基础图像分析:")
    print(f"   • 分辨率: {img.shape[1]} x {img.shape[0]}")
    print(f"   • 颜色通道: {img.shape[2] if len(img.shape) > 2 else 1}")
    print(f"   • 数据类型: {img.dtype}")
    
    # 2. 亮度和对比度分析
    mean_brightness = gray.mean()
    contrast = gray.std()
    print(f"   • 平均亮度: {mean_brightness:.1f} (理想: 100-150)")
    print(f"   • 对比度: {contrast:.1f} (理想: >30)")
    
    # 3. 清晰度分析
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"   • 清晰度: {laplacian_var:.1f} (理想: >100)")
    
    # 4. 寻找可能的棋盘格区域
    print("🔍 寻找可能的棋盘格区域...")
    
    # 使用轮廓检测寻找矩形区域
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # 过滤小区域
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # 矩形
                rectangular_contours.append(contour)
    
    print(f"   • 发现 {len(rectangular_contours)} 个矩形区域")
    
    # 分析最大的几个矩形区域
    if rectangular_contours:
        rectangular_contours.sort(key=cv2.contourArea, reverse=True)
        for i, contour in enumerate(rectangular_contours[:3]):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            print(f"   • 矩形 {i+1}: 位置({x}, {y}) 尺寸({w}x{h}) 面积({area:.0f})")

if __name__ == "__main__":
    # 测试文件夹中的一张图片
    image_folder = "/home/orangepi/Qworkspace/yolov8_pose_basketball/tools/test_ground_images"
    
    if not os.path.exists(image_folder):
        print(f"❌ 文件夹不存在: {image_folder}")
        exit(1)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    if not image_files:
        print("❌ 文件夹中没有图片")
        exit(1)
    
    # 选择中间的一张图片进行详细分析
    test_image = os.path.join(image_folder, image_files[len(image_files)//2])
    print(f"🎯 选择测试图片: {os.path.basename(test_image)}")
    
    # 详细分析
    analyze_single_image_detailed(test_image)
    
    # 广角检测
    success, result = debug_wide_angle_chessboard(test_image)
    
    if success:
        print(f"\n🎉 检测成功！建议的棋盘格尺寸: {result['size'][0]}x{result['size'][1]}")
    else:
        print("\n😞 检测失败，可能需要检查棋盘格是否存在于图像中")