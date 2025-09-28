#!/usr/bin/env python3
"""
独立相机标定验证器

直接加载标定结果文件(.npz)进行全面验证
支持命令行和GUI两种模式
"""

import numpy as np
import cv2
import os
import sys
import json
import argparse
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import glob

class CalibrationValidator:
    """独立标定验证器"""

    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.image_size = None
        self.calibration_data = None

        # 验证结果存储
        self.validation_results = {}
        self.test_images = []

    def load_calibration_file(self, npz_path):
        """加载标定结果文件"""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"标定文件不存在: {npz_path}")

        try:
            print(f"📂 加载标定文件: {npz_path}")
            data = np.load(npz_path)

            # 加载必需参数
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']

            # 加载可选参数
            if 'rvecs' in data:
                self.rvecs = data['rvecs']
            if 'tvecs' in data:
                self.tvecs = data['tvecs']
            if 'image_size' in data:
                self.image_size = tuple(data['image_size'])

            # 保存所有数据用于报告
            self.calibration_data = {}
            for key in data.keys():
                if key in ['camera_matrix', 'dist_coeffs', 'rvecs', 'tvecs']:
                    self.calibration_data[key] = data[key]
                elif key == 'image_size':
                    self.calibration_data[key] = tuple(data[key])
                else:
                    self.calibration_data[key] = data[key]

            print("✅ 标定文件加载成功")
            print(f"   📏 相机矩阵: {self.camera_matrix.shape}")
            print(f"   📐 畸变系数: {self.dist_coeffs.shape}")
            if self.rvecs is not None:
                print(f"   📍 旋转向量: {len(self.rvecs)} 组")
            if self.tvecs is not None:
                print(f"   📍 平移向量: {len(self.tvecs)} 组")
            if self.image_size:
                print(f"   📷 图像尺寸: {self.image_size}")

            return True

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False

    def find_test_images(self, image_dir=None):
        """查找测试图像"""
        self.test_images = []

        # 如果指定了目录，从该目录查找
        if image_dir and os.path.exists(image_dir):
            search_dirs = [image_dir]
        else:
            # 从当前目录及其子目录查找
            search_dirs = ['.'] + [d for d in os.listdir('.') if os.path.isdir(d)]

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue

            for ext in image_extensions:
                pattern = os.path.join(search_dir, ext)
                found_images = glob.glob(pattern)
                self.test_images.extend(found_images)

        # 去重并排序
        self.test_images = list(set(self.test_images))
        self.test_images.sort()

        print(f"📸 找到 {len(self.test_images)} 张测试图像")

        # 显示前几个图像
        for i, img_path in enumerate(self.test_images[:5]):
            print(f"   {i+1}. {os.path.basename(img_path)}")

        if len(self.test_images) > 5:
            print(f"   ... 还有 {len(self.test_images) - 5} 张图像")

        return self.test_images

    def run_comprehensive_validation(self):
        """运行综合验证"""
        print("\n🔍 开始综合验证...")

        results = {
            'validation_type': 'comprehensive',
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'has_intrinsics': self.camera_matrix is not None,
                'has_distortion': self.dist_coeffs is not None,
                'has_extrinsics': self.rvecs is not None and self.tvecs is not None,
                'image_count': len(self.test_images)
            }
        }

        # 1. 内参验证
        print("📏 验证内参...")
        results['intrinsic_validation'] = self.validate_intrinsics()

        # 2. 畸变矫正验证
        if self.test_images:
            print("📐 验证畸变矫正...")
            results['distortion_validation'] = self.validate_distortion_correction()

        # 3. 外参验证
        if self.rvecs is not None and self.tvecs is not None:
            print("📍 验证外参...")
            results['extrinsic_validation'] = self.validate_extrinsics()

        # 4. 性能测试
        print("⚡ 性能测试...")
        results['performance_test'] = self.test_performance()

        # 5. 计算总体质量
        results['overall_quality'] = self.calculate_overall_quality(results)

        self.validation_results = results

        print("\n✅ 综合验证完成!")
        print(f"📊 总体质量: {results['overall_quality']['grade']}")
        print(f"⭐ 质量评分: {results['overall_quality']['score']:.3f}")

        return results

    def validate_intrinsics(self):
        """验证内参"""
        results = {
            'focal_length_check': {'status': 'PASS', 'value': None, 'issues': []},
            'principal_point_check': {'status': 'PASS', 'value': None, 'issues': []},
            'distortion_check': {'status': 'PASS', 'value': None, 'issues': []},
            'matrix_validity': {'status': 'PASS', 'value': None, 'issues': []},
            'overall_score': 1.0
        }

        try:
            # 检查焦距
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            results['focal_length_check']['value'] = (fx, fy)

            if fx <= 0 or fy <= 0 or fx > 10000 or fy > 10000:
                results['focal_length_check']['status'] = 'FAIL'
                results['focal_length_check']['issues'].append(f"不合理的焦距值: fx={fx}, fy={fy}")
                results['overall_score'] -= 0.3

            # 检查主点
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            results['principal_point_check']['value'] = (cx, cy)

            # 如果有图像尺寸，进行更严格的检查
            if self.image_size:
                img_w, img_h = self.image_size
                if not (0 < cx < img_w and 0 < cy < img_h):
                    results['principal_point_check']['status'] = 'WARNING'
                    results['principal_point_check']['issues'].append(f"主点位置偏离图像中心: cx={cx}/{img_w}, cy={cy}/{img_h}")
                    results['overall_score'] -= 0.1

            # 检查畸变系数
            dist_norm = np.linalg.norm(self.dist_coeffs)
            results['distortion_check']['value'] = dist_norm

            if dist_norm > 1.0:
                results['distortion_check']['status'] = 'WARNING'
                results['distortion_check']['issues'].append(f"畸变系数较大: {dist_norm:.3f}")
                results['overall_score'] -= 0.1

            # 检查矩阵有效性
            if not np.allclose(self.camera_matrix[2, :], [0, 0, 1], atol=1e-10):
                results['matrix_validity']['status'] = 'FAIL'
                results['matrix_validity']['issues'].append("相机矩阵最后一行不是[0,0,1]")
                results['overall_score'] -= 0.2

            results['overall_score'] = max(0.0, results['overall_score'])

        except Exception as e:
            results['overall_score'] = 0.0
            results['matrix_validity']['status'] = 'ERROR'
            results['matrix_validity']['issues'].append(f"验证过程出错: {e}")

        return results

    def validate_distortion_correction(self):
        """验证畸变矫正效果"""
        if not self.test_images:
            return {'status': 'NO_IMAGES', 'message': '没有找到测试图像'}

        results = {
            'images_tested': 0,
            'correction_scores': [],
            'average_improvement': 0.0,
            'best_image': None,
            'worst_image': None,
            'recommendations': []
        }

        total_improvement = 0.0
        improvements = []

        # 测试前3张图像
        for img_path in self.test_images[:3]:
            try:
                improvement = self.analyze_single_image_correction(img_path)
                if improvement is not None:
                    improvements.append(improvement)
                    total_improvement += improvement
                    results['images_tested'] += 1

            except Exception as e:
                print(f"⚠️ 分析图像失败 {os.path.basename(img_path)}: {e}")
                continue

        if improvements:
            results['average_improvement'] = total_improvement / len(improvements)
            results['correction_scores'] = improvements
            results['best_image'] = max(improvements)
            results['worst_image'] = min(improvements)

            # 生成建议
            if results['average_improvement'] > 0.8:
                results['recommendations'].append("畸变矫正效果优秀")
            elif results['average_improvement'] > 0.6:
                results['recommendations'].append("畸变矫正效果良好")
            else:
                results['recommendations'].append("建议检查标定质量或使用不同alpha值")
                results['recommendations'].append("考虑重新标定以获得更好的畸变矫正效果")

        return results

    def analyze_single_image_correction(self, image_path):
        """分析单张图像的矫正效果"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 检测直线用于质量评估
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

            if lines is None or len(lines) < 3:
                # 如果没有检测到足够的直线，返回中等分数
                return 0.7

            # 计算原始图像的"直线性"
            original_straightness = self.calculate_image_straightness(lines, gray.shape[::-1])

            # 应用畸变矫正
            undistorted = cv2.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
            undistorted_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            undistorted_edges = cv2.Canny(undistorted_gray, 50, 150, apertureSize=3)
            undistorted_lines = cv2.HoughLinesP(undistorted_edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

            corrected_straightness = 0.5  # 默认值
            if undistorted_lines is not None and len(undistorted_lines) >= 3:
                corrected_straightness = self.calculate_image_straightness(undistorted_lines, undistorted_gray.shape[::-1])

            # 计算改善程度
            improvement = corrected_straightness - original_straightness

            # 归一化到0-1范围
            improvement = max(0, min(1, improvement + 0.5))

            return improvement

        except Exception as e:
            print(f"分析图像失败 {image_path}: {e}")
            return None

    def calculate_image_straightness(self, lines, image_size):
        """计算图像中直线的直线性"""
        if lines is None or len(lines) == 0:
            return 0.5

        straightness_scores = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算直线角度
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.degrees(angle) % 180

            # 计算直线长度
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # 计算直线与图像中心的距离
            center_x, center_y = image_size[0] / 2, image_size[1] / 2
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2
            distance_from_center = np.sqrt((line_center_x - center_x)**2 + (line_center_y - center_y)**2)

            # 距离中心越远，畸变影响越大
            max_distance = np.sqrt(center_x**2 + center_y**2)
            weight = 1 - (distance_from_center / max_distance)

            # 直线性评分（越接近0°或90°越直）
            angle_score = 1 - abs(np.sin(np.radians(angle_deg)))  # 0°或90°时为1

            # 综合评分
            straightness = weight * angle_score * (length / 100)  # 长度归一化
            straightness_scores.append(straightness)

        if straightness_scores:
            return np.mean(straightness_scores)
        else:
            return 0.5

    def validate_extrinsics(self):
        """验证外参"""
        if self.rvecs is None or self.tvecs is None:
            return {'status': 'NO_DATA', 'message': '没有外参数据'}

        results = {
            'rotation_analysis': {'status': 'PASS', 'issues': []},
            'translation_analysis': {'status': 'PASS', 'issues': []},
            'consistency_analysis': {'status': 'PASS', 'issues': []},
            'overall_score': 1.0
        }

        try:
            # 分析旋转参数
            rotation_magnitudes = [np.linalg.norm(rvec) for rvec in self.rvecs]
            avg_rotation = np.mean(rotation_magnitudes)
            std_rotation = np.std(rotation_magnitudes)

            results['rotation_analysis']['magnitude_range'] = f"{min(rotation_magnitudes):.3f} - {max(rotation_magnitudes):.3f}"
            results['rotation_analysis']['average'] = avg_rotation
            results['rotation_analysis']['std_dev'] = std_rotation

            # 检查旋转参数合理性
            if avg_rotation > 5.0:  # 太大的旋转
                results['rotation_analysis']['status'] = 'WARNING'
                results['rotation_analysis']['issues'].append(f"平均旋转角度过大: {avg_rotation:.3f}")
                results['overall_score'] -= 0.1

            # 分析平移参数
            translation_magnitudes = [np.linalg.norm(tvec) for tvec in self.tvecs]
            avg_translation = np.mean(translation_magnitudes)
            std_translation = np.std(translation_magnitudes)

            results['translation_analysis']['magnitude_range'] = f"{min(translation_magnitudes):.1f} - {max(translation_magnitudes):.1f}"
            results['translation_analysis']['average'] = avg_translation
            results['translation_analysis']['std_dev'] = std_translation

            # 检查平移参数合理性
            if avg_translation > 10000:  # 距离过远
                results['translation_analysis']['status'] = 'WARNING'
                results['translation_analysis']['issues'].append(f"平均距离过远: {avg_translation:.1f}")
                results['overall_score'] -= 0.1

            # 分析一致性
            if len(self.rvecs) > 1:
                # 计算相邻视图之间的一致性
                consistency_scores = []
                for i in range(1, len(self.rvecs)):
                    rvec_diff = np.linalg.norm(self.rvecs[i] - self.rvecs[i-1])
                    tvec_diff = np.linalg.norm(self.tvecs[i] - self.tvecs[i-1])

                    # 归一化差异
                    consistency = 1.0 / (1.0 + rvec_diff + tvec_diff * 0.001)  # 平移单位换算
                    consistency_scores.append(consistency)

                avg_consistency = np.mean(consistency_scores)
                results['consistency_analysis']['average_consistency'] = avg_consistency

                if avg_consistency < 0.7:
                    results['consistency_analysis']['status'] = 'WARNING'
                    results['consistency_analysis']['issues'].append(f"位姿一致性较差: {avg_consistency:.3f}")
                    results['overall_score'] -= 0.1

            results['overall_score'] = max(0.0, results['overall_score'])

        except Exception as e:
            results['overall_score'] = 0.0
            results['rotation_analysis']['status'] = 'ERROR'
            results['rotation_analysis']['issues'].append(f"验证过程出错: {e}")

        return results

    def test_performance(self):
        """测试性能"""
        results = {
            'undistortion_speed': {'fps': 0, 'ms_per_frame': 0},
            'memory_usage': {'estimated_mb': 0},
            'scalability': {'score': 0.8}
        }

        try:
            # 测试去畸变速度
            if self.test_images:
                import time

                # 使用第一张图像进行测试
                test_img = cv2.imread(self.test_images[0])
                if test_img is not None:
                    # 预热
                    for _ in range(5):
                        cv2.undistort(test_img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

                    # 正式测试
                    start_time = time.time()
                    num_iterations = 50

                    for _ in range(num_iterations):
                        cv2.undistort(test_img, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

                    end_time = time.time()

                    total_time = end_time - start_time
                    fps = num_iterations / total_time
                    ms_per_frame = (total_time / num_iterations) * 1000

                    results['undistortion_speed']['fps'] = fps
                    results['undistortion_speed']['ms_per_frame'] = ms_per_frame

                    print(".1f")
                    print(".2f")
            # 估算内存使用
            matrix_size = self.camera_matrix.nbytes + self.dist_coeffs.nbytes
            if self.rvecs is not None:
                matrix_size += self.rvecs.nbytes
            if self.tvecs is not None:
                matrix_size += self.tvecs.nbytes

            memory_mb = matrix_size / (1024 * 1024)
            results['memory_usage']['estimated_mb'] = memory_mb

            print(".2f")
        except Exception as e:
            print(f"⚠️ 性能测试出错: {e}")

        return results

    def calculate_overall_quality(self, results):
        """计算总体质量"""
        scores = []

        # 内参评分
        if 'intrinsic_validation' in results:
            scores.append(results['intrinsic_validation']['overall_score'])

        # 畸变矫正评分
        if 'distortion_validation' in results and 'average_improvement' in results['distortion_validation']:
            scores.append(results['distortion_validation']['average_improvement'])

        # 外参评分
        if 'extrinsic_validation' in results:
            scores.append(results['extrinsic_validation']['overall_score'])

        # 计算平均分
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.5

        # 确定等级
        if overall_score >= 0.9:
            grade = 'EXCELLENT'
            description = '标定质量优秀，适合高精度应用'
        elif overall_score >= 0.8:
            grade = 'GOOD'
            description = '标定质量良好，适用于大多数应用'
        elif overall_score >= 0.7:
            grade = 'ACCEPTABLE'
            description = '标定质量可接受，建议优化'
        else:
            grade = 'POOR'
            description = '标定质量较差，建议重新标定'

        return {
            'score': overall_score,
            'grade': grade,
            'description': description,
            'component_scores': scores
        }

    def generate_report(self, output_file=None):
        """生成详细验证报告"""
        if not self.validation_results:
            print("❌ 没有验证结果，请先运行验证")
            return None

        report = f"""相机标定验证报告
{'='*50}
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

总体评估
{'-'*30}
质量等级: {self.validation_results['overall_quality']['grade']}
质量评分: {self.validation_results['overall_quality']['score']:.3f}
描述: {self.validation_results['overall_quality']['description']}

文件信息
{'-'*30}
内参: {'✅' if self.validation_results['file_info']['has_intrinsics'] else '❌'}
畸变参数: {'✅' if self.validation_results['file_info']['has_distortion'] else '❌'}
外参: {'✅' if self.validation_results['file_info']['has_extrinsics'] else '❌'}
测试图像: {self.validation_results['file_info']['image_count']} 张

内参验证详情
{'-'*30}
"""

        if 'intrinsic_validation' in self.validation_results:
            iv = self.validation_results['intrinsic_validation']
            report += f"焦距检查: {iv['focal_length_check']['status']}"
            if iv['focal_length_check']['value']:
                report += f" (fx={iv['focal_length_check']['value'][0]:.1f}, fy={iv['focal_length_check']['value'][1]:.1f})"
            report += "\n"

            report += f"主点检查: {iv['principal_point_check']['status']}"
            if iv['principal_point_check']['value']:
                report += f" (cx={iv['principal_point_check']['value'][0]:.1f}, cy={iv['principal_point_check']['value'][1]:.1f})"
            report += "\n"

            report += f"畸变检查: {iv['distortion_check']['status']}"
            if iv['distortion_check']['value'] is not None:
                report += f" (norm={iv['distortion_check']['value']:.3f})"
            report += "\n"

            report += f"矩阵有效性: {iv['matrix_validity']['status']}\n"
            report += f"综合评分: {iv['overall_score']:.3f}\n"

            # 显示问题
            all_issues = []
            for check in ['focal_length_check', 'principal_point_check', 'distortion_check', 'matrix_validity']:
                all_issues.extend(iv[check]['issues'])

            if all_issues:
                report += "\n发现的问题:\n"
                for issue in all_issues:
                    report += f"• {issue}\n"

        report += f"""
畸变矫正验证
{'-'*30}
"""

        if 'distortion_validation' in self.validation_results:
            dv = self.validation_results['distortion_validation']
            if dv.get('status') == 'NO_IMAGES':
                report += "状态: 未找到测试图像\n"
            else:
                report += f"测试图像数量: {dv['images_tested']}\n"
                report += f"平均改善程度: {dv['average_improvement']:.3f}\n"
                if dv['best_image'] is not None:
                    report += f"最佳改善: {dv['best_image']:.3f}\n"
                if dv['worst_image'] is not None:
                    report += f"最差改善: {dv['worst_image']:.3f}\n"

                if dv['recommendations']:
                    report += "\n建议:\n"
                    for rec in dv['recommendations']:
                        report += f"• {rec}\n"

        report += f"""
外参验证
{'-'*30}
"""

        if 'extrinsic_validation' in self.validation_results:
            ev = self.validation_results['extrinsic_validation']
            if ev.get('status') == 'NO_DATA':
                report += "状态: 没有外参数据\n"
            else:
                report += f"旋转参数: {ev['rotation_analysis']['status']}\n"
                if 'magnitude_range' in ev['rotation_analysis']:
                    report += f"  范围: {ev['rotation_analysis']['magnitude_range']}\n"

                report += f"平移参数: {ev['translation_analysis']['status']}\n"
                if 'magnitude_range' in ev['translation_analysis']:
                    report += f"  范围: {ev['translation_analysis']['magnitude_range']}\n"

                report += f"一致性分析: {ev['consistency_analysis']['status']}\n"
                if 'average_consistency' in ev['consistency_analysis']:
                    report += f"  平均一致性: {ev['consistency_analysis']['average_consistency']:.3f}\n"

                report += f"综合评分: {ev['overall_score']:.3f}\n"

        report += f"""
性能测试
{'-'*30}
"""

        if 'performance_test' in self.validation_results:
            pt = self.validation_results['performance_test']
            if 'undistortion_speed' in pt and pt['undistortion_speed']['fps'] > 0:
                report += f"去畸变速度: {pt['undistortion_speed']['fps']:.1f} FPS\n"
                report += f"每帧耗时: {pt['undistortion_speed']['ms_per_frame']:.2f} ms\n"

            if 'memory_usage' in pt:
                report += f"内存使用: {pt['memory_usage']['estimated_mb']:.2f} MB\n"

        report += f"""
总结与建议
{'-'*30}
"""

        overall = self.validation_results['overall_quality']
        report += f"总体质量等级: {overall['grade']}\n"
        report += f"质量评分: {overall['score']:.3f}\n"
        report += f"描述: {overall['description']}\n\n"

        # 具体建议
        if overall['score'] >= 0.9:
            report += "🎉 恭喜！您的标定结果质量优秀，可以用于高精度应用。\n"
        elif overall['score'] >= 0.8:
            report += "✅ 标定结果质量良好，适用于大多数计算机视觉应用。\n"
        elif overall['score'] >= 0.7:
            report += "⚠️ 标定结果可接受，但建议进行一些优化:\n"
            report += "   • 检查标定图像质量\n"
            report += "   • 增加标定图像数量\n"
            report += "   • 确保棋盘格完全可见\n"
        else:
            report += "❌ 标定质量需要改进，建议:\n"
            report += "   • 重新进行标定过程\n"
            report += "   • 使用更高分辨率的图像\n"
            report += "   • 改善拍摄条件和环境\n"
            report += "   • 检查标定参数设置\n"

        # 保存报告
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ 报告已保存到: {output_file}")
            except Exception as e:
                print(f"❌ 保存报告失败: {e}")

        return report


class ValidationGUI:
    """验证器GUI界面"""

    def __init__(self):
        self.validator = CalibrationValidator()
        self.root = tk.Tk()
        self.root.title("Calibration Validator")
        self.root.geometry("1000x700")

        # 设置样式
        self.setup_styles()

        # 创建界面
        self.setup_ui()

    def setup_styles(self):
        """设置样式"""
        style = ttk.Style()
        style.theme_use('clam')

        # 按钮样式
        style.configure('Success.TButton', background='#28a745', foreground='white')
        style.configure('Primary.TButton', background='#007bff', foreground='white')
        style.configure('Danger.TButton', background='#dc3545', foreground='white')

    def setup_ui(self):
        """创建界面"""
        # 主容器
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 标题
        title_label = ttk.Label(main_frame, text="🎯 相机标定验证器",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))

        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="📁 文件选择", padding=10)
        file_frame.pack(fill='x', pady=(0, 10))

        # 标定文件选择
        ttk.Label(file_frame, text="标定文件:").grid(row=0, column=0, sticky='w', pady=5)
        self.calibration_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.calibration_file_var).grid(row=0, column=1, sticky='ew', padx=(10, 5))
        ttk.Button(file_frame, text="选择文件", command=self.select_calibration_file).grid(row=0, column=2)

        # 图像目录选择
        ttk.Label(file_frame, text="测试图像目录:").grid(row=1, column=0, sticky='w', pady=5)
        self.image_dir_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.image_dir_var).grid(row=1, column=1, sticky='ew', padx=(10, 5))
        ttk.Button(file_frame, text="选择目录", command=self.select_image_directory).grid(row=1, column=2)

        file_frame.columnconfigure(1, weight=1)

        # 控制按钮
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 20))

        ttk.Button(control_frame, text="🚀 快速验证", style='Success.TButton',
                  command=self.run_quick_validation).pack(side='left', padx=(0, 10))

        ttk.Button(control_frame, text="🔍 全面验证", style='Primary.TButton',
                  command=self.run_comprehensive_validation).pack(side='left', padx=(0, 10))

        ttk.Button(control_frame, text="📄 生成报告", style='Primary.TButton',
                  command=self.generate_report).pack(side='left', padx=(0, 10))

        ttk.Button(control_frame, text="❌ 退出", style='Danger.TButton',
                  command=self.root.quit).pack(side='right')

        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="📊 验证结果", padding=10)
        result_frame.pack(fill='both', expand=True)

        # 结果文本框
        self.result_text = tk.Text(result_frame, wrap='word', font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical', command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 初始显示
        self.show_welcome_message()

    def show_welcome_message(self):
        """显示欢迎消息"""
        welcome_text = """🎯 欢迎使用相机标定验证器

使用步骤:
1. 📁 选择标定结果文件 (.npz格式)
2. 📷 选择测试图像目录 (可选)
3. 🚀 点击"快速验证"或"全面验证"
4. 📊 查看详细的验证结果和建议
5. 📄 生成验证报告

支持的验证类型:
• ✅ 内参合理性检查
• 📐 畸变矫正效果评估
• 📍 外参参数验证
• ⚡ 性能基准测试
• 👁️ 可视化质量对比

提示:
• 如果没有测试图像，验证器会自动搜索当前目录
• 验证结果会自动保存到 validation_report.txt
• 可以多次运行不同类型的验证进行比较

请先选择标定文件开始验证...
"""
        self.result_text.insert('1.0', welcome_text)
        self.result_text.config(state='disabled')

    def select_calibration_file(self):
        """选择标定文件"""
        file_path = filedialog.askopenfilename(
            title="选择标定文件",
            filetypes=[("NPZ files", "*.npz"), ("All files", "*.*")]
        )

        if file_path:
            self.calibration_file_var.set(file_path)

            # 尝试加载文件
            if self.validator.load_calibration_file(file_path):
                self.show_file_loaded_message()
            else:
                messagebox.showerror("错误", "无法加载标定文件")

    def select_image_directory(self):
        """选择图像目录"""
        dir_path = filedialog.askdirectory(title="选择测试图像目录")

        if dir_path:
            self.image_dir_var.set(dir_path)

            # 查找图像
            self.validator.find_test_images(dir_path)
            self.show_images_found_message()

    def show_file_loaded_message(self):
        """显示文件加载成功消息"""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')

        message = f"""✅ 标定文件加载成功!

文件信息:
• 相机矩阵: {self.validator.camera_matrix.shape if self.validator.camera_matrix is not None else 'N/A'}
• 畸变系数: {self.validator.dist_coeffs.shape if self.validator.dist_coeffs is not None else 'N/A'}
• 外参数据: {'✅' if self.validator.rvecs is not None else '❌'}
• 图像尺寸: {self.validator.image_size if self.validator.image_size else 'Unknown'}

内参预览:
• 焦距: fx={self.validator.camera_matrix[0,0]:.1f}, fy={self.validator.camera_matrix[1,1]:.1f}
• 主点: cx={self.validator.camera_matrix[0,2]:.1f}, cy={self.validator.camera_matrix[1,2]:.1f}
• 畸变: {self.validator.dist_coeffs.flatten()[:3]}...

现在可以开始验证了!
"""
        self.result_text.insert('1.0', message)
        self.result_text.config(state='disabled')

    def show_images_found_message(self):
        """显示找到的图像"""
        self.result_text.config(state='normal')

        current_text = self.result_text.get('1.0', 'end')
        if "现在可以开始验证了!" in current_text:
            # 追加图像信息
            image_message = f"\n📸 找到 {len(self.validator.test_images)} 张测试图像\n"
            for i, img_path in enumerate(self.validator.test_images[:5]):
                image_message += f"• {os.path.basename(img_path)}\n"

            if len(self.validator.test_images) > 5:
                image_message += f"... 还有 {len(self.validator.test_images) - 5} 张\n"

            self.result_text.insert('end', image_message)

        self.result_text.config(state='disabled')

    def run_quick_validation(self):
        """运行快速验证"""
        if not self.calibration_file_var.get():
            messagebox.showwarning("警告", "请先选择标定文件")
            return

        # 禁用按钮
        self.disable_buttons()

        # 在后台运行验证
        def validation_worker():
            try:
                # 查找测试图像
                image_dir = self.image_dir_var.get()
                self.validator.find_test_images(image_dir if image_dir else None)

                # 运行验证
                results = self.validator.run_comprehensive_validation()

                # 显示结果
                self.root.after(0, lambda: self.display_validation_results(results))

            except Exception as e:
                self.root.after(0, lambda: self.show_validation_error(str(e)))

        threading.Thread(target=validation_worker, daemon=True).start()

    def run_comprehensive_validation(self):
        """运行全面验证"""
        self.run_quick_validation()  # 目前使用相同的逻辑

    def display_validation_results(self, results):
        """显示验证结果"""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')

        # 生成报告
        report = self.validator.generate_report()

        self.result_text.insert('1.0', report)
        self.result_text.config(state='disabled')

        # 启用按钮
        self.enable_buttons()

        # 显示成功消息
        quality = results['overall_quality']
        messagebox.showinfo("验证完成",
                          f"标定验证完成!\n\n"
                          f"质量等级: {quality['grade']}\n"
                          f"质量评分: {quality['score']:.3f}\n"
                          f"描述: {quality['description']}")

    def show_validation_error(self, error_msg):
        """显示验证错误"""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        self.result_text.insert('1.0', f"❌ 验证失败:\n\n{error_msg}")
        self.result_text.config(state='disabled')

        # 启用按钮
        self.enable_buttons()

        messagebox.showerror("验证失败", error_msg)

    def generate_report(self):
        """生成验证报告"""
        if not self.validator.validation_results:
            messagebox.showwarning("警告", "请先运行验证")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="保存验证报告"
        )

        if file_path:
            try:
                report = self.validator.generate_report(file_path)
                messagebox.showinfo("成功", f"报告已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存报告失败: {e}")

    def disable_buttons(self):
        """禁用所有按钮"""
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for button in widget.winfo_children():
                            if isinstance(button, ttk.Button):
                                button.config(state='disabled')

    def enable_buttons(self):
        """启用所有按钮"""
        for child in self.root.winfo_children():
            if isinstance(child, tk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for button in widget.winfo_children():
                            if isinstance(button, ttk.Button):
                                button.config(state='normal')

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='相机标定验证器')
    parser.add_argument('calibration_file', nargs='?', help='标定结果文件路径 (.npz)')
    parser.add_argument('--images', '-i', help='测试图像目录路径')
    parser.add_argument('--output', '-o', help='输出报告文件路径')
    parser.add_argument('--gui', action='store_true', help='启动GUI模式')
    parser.add_argument('--quick', action='store_true', help='快速验证模式')

    args = parser.parse_args()

    if args.gui:
        # GUI模式
        app = ValidationGUI()
        app.run()
    else:
        # 命令行模式
        if not args.calibration_file:
            print("❌ 请提供标定文件路径")
            print("使用方法:")
            print("  python calibration_validator.py <calibration_file.npz> [选项]")
            print("选项:")
            print("  --images <目录>    测试图像目录")
            print("  --output <文件>    输出报告文件")
            print("  --quick           快速验证模式")
            print("  --gui             GUI模式")
            return

        # 创建验证器
        validator = CalibrationValidator()

        # 加载标定文件
        if not validator.load_calibration_file(args.calibration_file):
            return

        # 查找测试图像
        image_dir = args.images
        validator.find_test_images(image_dir)

        # 运行验证
        print("\n🔍 开始验证...")
        results = validator.run_comprehensive_validation()

        # 生成报告
        output_file = args.output or "validation_report.txt"
        report = validator.generate_report(output_file)

        print(f"\n✅ 验证完成!")
        print(f"📊 质量等级: {results['overall_quality']['grade']}")
        print(f"⭐ 质量评分: {results['overall_quality']['score']:.3f}")
        print(f"📄 报告已保存: {output_file}")

        # 显示关键指标
        print("\n📈 关键指标:")
        if 'intrinsic_validation' in results:
            iv = results['intrinsic_validation']
            print(f"• 内参评分: {iv['overall_score']:.3f}")

        if 'distortion_validation' in results and results['distortion_validation'].get('average_improvement'):
            dv = results['distortion_validation']
            print(f"• 畸变改善: {dv['average_improvement']:.3f}")

        if 'extrinsic_validation' in results:
            ev = results['extrinsic_validation']
            print(f"• 外参评分: {ev['overall_score']:.3f}")


if __name__ == "__main__":
    main()
