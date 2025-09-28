#!/usr/bin/env python3
"""
生成标定过程的调试图像
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

def save_debug_image_with_corners(img, corners, filename, stage="detected"):
    """保存带角点标记的调试图像"""
    debug_img = img.copy()
    
    if corners is not None:
        # 绘制检测到的角点
        cv2.drawChessboardCorners(debug_img, (9, 6), corners, True)
        
        # 添加信息文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, f"Stage: {stage}", (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"Corners: {len(corners)}", (10, 70), font, 1, (0, 255, 0), 2)
        cv2.putText(debug_img, f"File: {filename}", (10, 110), font, 0.7, (0, 255, 0), 2)
    
    # 保存调试图像
    debug_dir = Path("debug_calibration_process")
    debug_dir.mkdir(exist_ok=True)
    
    base_name = Path(filename).stem
    debug_path = debug_dir / f"{base_name}_{stage}_debug.jpg"
    cv2.imwrite(str(debug_path), debug_img)
    print(f"   💾 Saved debug image: {debug_path}")

def assess_image_quality_visual(img_path):
    """评估图像质量并生成可视化结果"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 清晰度检测
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # 2. 边缘检测
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 3. 对比度
    contrast = gray.std()
    
    # 4. 亮度
    brightness = gray.mean()
    
    # 创建质量分析图像
    analysis_img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # 显示原图缩放版本
    h, w = gray.shape
    scale = min(300/w, 200/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    analysis_img[10:10+new_h, 10:10+new_w] = img_resized
    
    # 显示边缘图
    edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    edges_resized = cv2.resize(edges_colored, (new_w, new_h))
    analysis_img[10:10+new_h, 400:400+new_w] = edges_resized
    
    # 添加文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 250
    cv2.putText(analysis_img, "Quality Analysis:", (10, y_pos), font, 0.8, (255, 255, 255), 2)
    y_pos += 40
    cv2.putText(analysis_img, f"Sharpness: {sharpness:.1f}", (10, y_pos), font, 0.6, (0, 255, 0), 1)
    y_pos += 30
    cv2.putText(analysis_img, f"Contrast: {contrast:.1f}", (10, y_pos), font, 0.6, (0, 255, 0), 1)
    y_pos += 30
    cv2.putText(analysis_img, f"Brightness: {brightness:.1f}", (10, y_pos), font, 0.6, (0, 255, 0), 1)
    y_pos += 30
    cv2.putText(analysis_img, f"Edge density: {edge_density*100:.2f}%", (10, y_pos), font, 0.6, (0, 255, 0), 1)
    
    # 质量评分
    score = 10.0
    if sharpness < 50: score -= 3
    elif sharpness < 100: score -= 1
    if contrast < 30: score -= 2
    if brightness < 50 or brightness > 200: score -= 1.5
    if edge_density < 0.05: score -= 1
    
    y_pos += 50
    color = (0, 255, 0) if score >= 6 else (0, 255, 255) if score >= 3 else (0, 0, 255)
    cv2.putText(analysis_img, f"Overall Score: {score:.1f}/10", (10, y_pos), font, 0.8, color, 2)
    
    return analysis_img, score

def process_calibration_images(image_folder):
    """处理标定图像并生成调试信息"""
    print("🔍 Processing calibration images for debugging...")
    
    # 获取图像列表
    image_patterns = ['*.jpg', '*.png', '*.jpeg', '*.JPG']
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(glob.glob(os.path.join(image_folder, pattern)))
    
    if not image_paths:
        print("❌ No images found in the folder!")
        return
    
    print(f"📸 Found {len(image_paths)} images")
    
    # 创建调试目录
    debug_dir = Path("debug_calibration_process")
    debug_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        print(f"\n🖼️ Processing {i}/{len(image_paths)}: {filename}")
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 生成质量分析图像
        analysis_img, quality_score = assess_image_quality_visual(img_path)
        if analysis_img is not None:
            analysis_path = debug_dir / f"{Path(filename).stem}_quality_analysis.jpg"
            cv2.imwrite(str(analysis_path), analysis_img)
            print(f"   📊 Quality score: {quality_score:.2f}/10 - Saved: {analysis_path}")
        
        # 尝试检测棋盘格
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        
        if ret:
            print(f"   ✅ Chessboard detected with {len(corners)} corners")
            save_debug_image_with_corners(img, corners, filename, "original")
            
            # 精化角点
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            save_debug_image_with_corners(img, corners_refined, filename, "refined")
            
            results.append((filename, quality_score, "SUCCESS", len(corners)))
        else:
            print(f"   ❌ Failed to detect chessboard")
            # 保存失败的图像用于分析
            save_debug_image_with_corners(img, None, filename, "failed")
            results.append((filename, quality_score, "FAILED", 0))
    
    # 生成总结报告
    print(f"\n📋 SUMMARY REPORT:")
    print("=" * 70)
    success_count = len([r for r in results if r[2] == "SUCCESS"])
    print(f"Total images: {len(results)}")
    print(f"Successful detections: {success_count}")
    print(f"Success rate: {success_count/len(results)*100:.1f}%")
    print(f"\n📂 Debug images saved to: {debug_dir.absolute()}")
    
    # 保存详细报告
    report_path = debug_dir / "calibration_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Ground Calibration Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Successful detections: {success_count}\n")
        f.write(f"Success rate: {success_count/len(results)*100:.1f}%\n\n")
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")
        for filename, score, status, corners in results:
            f.write(f"{filename:<30} | Score: {score:5.2f} | {status:<8} | Corners: {corners}\n")
    
    print(f"📝 Detailed report saved to: {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_calibration_images.py <image_folder>")
        print("Example: python debug_calibration_images.py /path/to/calibration/images")
        sys.exit(1)
    
    image_folder = sys.argv[1]
    if not os.path.exists(image_folder):
        print(f"❌ Folder does not exist: {image_folder}")
        sys.exit(1)
    
    process_calibration_images(image_folder)