#!/usr/bin/env python3
"""
多格式标定文件管理器

支持保存和加载多种格式的标定文件：
- NPZ (NumPy) - 原生格式，包含所有数据类型
- JSON - 人类可读，跨平台兼容
- XML - OpenCV标准格式，C++兼容

文件名格式: YYYYMMDD_HHMMSS_calibration.{ext}
"""

import numpy as np
import json
import xml.etree.ElementTree as ET
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import base64
import warnings

class CalibrationFileManager:
    """多格式标定文件管理器"""

    SUPPORTED_FORMATS = ['npz', 'json', 'xml']

    def __init__(self):
        self.current_data = {}
        self.file_prefix = ""

    def generate_timestamp_prefix(self) -> str:
        """生成时间戳前缀"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def prepare_calibration_data(self, calibration_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备标定数据，确保数据格式适合保存

        参数:
        calibration_data: 原始标定数据字典

        返回:
        处理后的数据字典
        """
        processed_data = {}

        for key, value in calibration_data.items():
            if isinstance(value, np.ndarray):
                # NumPy数组转换为列表
                processed_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                # 列表或元组中的NumPy数组也需要转换
                if value and isinstance(value[0], np.ndarray):
                    processed_data[key] = [arr.tolist() for arr in value]
                else:
                    processed_data[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                processed_data[key] = self.prepare_calibration_data(value)
            else:
                # 其他类型直接保存
                processed_data[key] = value

        return processed_data

    def save_calibration_multi_format(self, calibration_data: Dict[str, Any],
                                    output_dir: str = "./calibration_results",
                                    formats: list = None) -> Dict[str, str]:
        """
        保存标定结果为多种格式

        参数:
        calibration_data: 标定数据字典
        output_dir: 输出目录
        formats: 要保存的格式列表，默认保存所有支持格式

        返回:
        保存的文件路径字典 {format: filepath}
        """
        if formats is None:
            formats = self.SUPPORTED_FORMATS

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳前缀
        timestamp = self.generate_timestamp_prefix()
        self.file_prefix = f"{timestamp}_calibration"

        # 准备数据
        processed_data = self.prepare_calibration_data(calibration_data)
        processed_data['save_timestamp'] = timestamp
        processed_data['file_formats'] = formats

        saved_files = {}

        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                warnings.warn(f"不支持的格式: {fmt}，跳过")
                continue

            filename = f"{self.file_prefix}.{fmt}"
            filepath = os.path.join(output_dir, filename)

            try:
                if fmt == 'npz':
                    self._save_npz(calibration_data, filepath)
                elif fmt == 'json':
                    self._save_json(processed_data, filepath)
                elif fmt == 'xml':
                    self._save_xml(processed_data, filepath)

                saved_files[fmt] = filepath
                print(f"✅ 保存 {fmt.upper()} 格式: {filename}")

            except Exception as e:
                print(f"❌ 保存 {fmt.upper()} 格式失败: {e}")
                continue

        return saved_files

    def _save_npz(self, data: Dict[str, Any], filepath: str):
        """保存为NPZ格式"""
        # NPZ格式可以直接保存NumPy数组
        np.savez(filepath, **data)

    def _save_json(self, data: Dict[str, Any], filepath: str):
        """保存为JSON格式"""
        # 处理特殊数据类型
        json_data = self._prepare_for_json(data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def _save_xml(self, data: Dict[str, Any], filepath: str):
        """保存为XML格式（OpenCV兼容）"""
        root = ET.Element("opencv_storage")

        # 添加基本信息
        ET.SubElement(root, "save_timestamp").text = data.get('save_timestamp', '')
        ET.SubElement(root, "file_formats").text = str(data.get('file_formats', []))

        # 保存相机矩阵
        if 'camera_matrix' in data:
            self._add_matrix_to_xml(root, "camera_matrix", data['camera_matrix'])

        # 保存畸变系数
        if 'dist_coeffs' in data:
            self._add_matrix_to_xml(root, "distortion_coefficients", data['dist_coeffs'])

        # 保存外参
        if 'rvecs' in data and data['rvecs']:
            rvecs_elem = ET.SubElement(root, "rotation_vectors")
            for i, rvec in enumerate(data['rvecs']):
                self._add_matrix_to_xml(rvecs_elem, f"rvec_{i}", rvec)

        if 'tvecs' in data and data['tvecs']:
            tvecs_elem = ET.SubElement(root, "translation_vectors")
            for i, tvec in enumerate(data['tvecs']):
                self._add_matrix_to_xml(tvecs_elem, f"tvec_{i}", tvec)

        # 保存其他元数据
        metadata_fields = ['calibration_date', 'board_params', 'image_size',
                          'per_view_errors', 'successful_image_indices',
                          'total_images_processed', 'successful_images_count']

        for field in metadata_fields:
            if field in data:
                if isinstance(data[field], (list, tuple)):
                    ET.SubElement(root, field).text = str(data[field])
                elif isinstance(data[field], dict):
                    field_elem = ET.SubElement(root, field)
                    for k, v in data[field].items():
                        ET.SubElement(field_elem, k).text = str(v)
                else:
                    ET.SubElement(root, field).text = str(data[field])

        # 保存XML文件
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding='utf-8', xml_declaration=True)

    def _add_matrix_to_xml(self, parent, name: str, matrix: list):
        """向XML添加矩阵元素"""
        matrix_elem = ET.SubElement(parent, name)
        matrix_elem.set("type_id", "opencv-matrix")

        # 确定矩阵尺寸
        if isinstance(matrix, list) and matrix:
            if isinstance(matrix[0], list):
                rows = len(matrix)
                cols = len(matrix[0])
            else:
                rows = 1
                cols = len(matrix)
        else:
            rows = cols = 0

        ET.SubElement(matrix_elem, "rows").text = str(rows)
        ET.SubElement(matrix_elem, "cols").text = str(cols)
        ET.SubElement(matrix_elem, "dt").text = "f"  # float
        ET.SubElement(matrix_elem, "data").text = self._matrix_to_string(matrix)

    def _matrix_to_string(self, matrix: list) -> str:
        """将矩阵转换为OpenCV XML格式的字符串"""
        if not matrix:
            return ""

        # 展平矩阵为一行
        flat_data = []
        if isinstance(matrix[0], list):
            for row in matrix:
                flat_data.extend(row)
        else:
            flat_data = matrix

        # 转换为字符串，每个元素之间用空格分隔
        return " ".join(f"{x:.6f}" for x in flat_data)

    def _prepare_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """准备数据用于JSON序列化"""
        json_data = {}

        for key, value in data.items():
            if isinstance(value, (np.ndarray, list)):
                # 确保所有数值都是Python原生类型
                if isinstance(value, np.ndarray):
                    value = value.tolist()

                # 处理嵌套列表
                if isinstance(value, list) and value and isinstance(value[0], list):
                    # 可能是多维数组
                    json_data[key] = value
                else:
                    json_data[key] = value
            elif isinstance(value, dict):
                json_data[key] = self._prepare_for_json(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                json_data[key] = value
            else:
                # 其他类型转换为字符串
                json_data[key] = str(value)

        return json_data

    def load_calibration_file(self, filepath: str) -> Tuple[Dict[str, Any], str]:
        """
        加载标定文件，支持多种格式自动检测

        参数:
        filepath: 文件路径

        返回:
        (标定数据字典, 文件格式)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        # 根据文件扩展名确定格式
        _, ext = os.path.splitext(filepath)
        format_type = ext.lower().lstrip('.')

        if format_type not in self.SUPPORTED_FORMATS:
            # 尝试自动检测格式
            format_type = self._detect_format(filepath)

        try:
            if format_type == 'npz':
                data = self._load_npz(filepath)
            elif format_type == 'json':
                data = self._load_json(filepath)
            elif format_type == 'xml':
                data = self._load_xml(filepath)
            else:
                raise ValueError(f"不支持的文件格式: {format_type}")

            print(f"✅ 成功加载 {format_type.upper()} 格式文件: {os.path.basename(filepath)}")
            return data, format_type

        except Exception as e:
            raise RuntimeError(f"加载文件失败: {e}")

    def _detect_format(self, filepath: str) -> str:
        """自动检测文件格式"""
        try:
            # 尝试读取文件头几个字节
            with open(filepath, 'rb') as f:
                header = f.read(100)

            # 检查是否是NPZ文件
            if header.startswith(b'\x93NUMPY'):
                return 'npz'

            # 检查是否是XML文件
            header_str = header.decode('utf-8', errors='ignore')
            if '<?xml' in header_str or '<opencv_storage' in header_str:
                return 'xml'

            # 尝试解析为JSON
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    json.load(f)
                return 'json'
            except:
                pass

        except:
            pass

        raise ValueError("无法自动检测文件格式")

    def _load_npz(self, filepath: str) -> Dict[str, Any]:
        """加载NPZ格式文件"""
        data = np.load(filepath)
        result = {}

        for key in data.keys():
            if key in ['camera_matrix', 'dist_coeffs', 'rvecs', 'tvecs']:
                result[key] = data[key]
            else:
                result[key] = data[key].item() if data[key].ndim == 0 else data[key]

        return result

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """加载JSON格式文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换回NumPy数组
        if 'camera_matrix' in data:
            data['camera_matrix'] = np.array(data['camera_matrix'])
        if 'dist_coeffs' in data:
            data['dist_coeffs'] = np.array(data['dist_coeffs'])
        if 'rvecs' in data:
            data['rvecs'] = [np.array(rvec) for rvec in data['rvecs']]
        if 'tvecs' in data:
            data['tvecs'] = [np.array(tvec) for tvec in data['tvecs']]

        return data

    def _load_xml(self, filepath: str) -> Dict[str, Any]:
        """加载XML格式文件"""
        tree = ET.parse(filepath)
        root = tree.getroot()

        data = {}

        # 解析相机矩阵
        camera_elem = root.find("camera_matrix")
        if camera_elem is not None:
            data['camera_matrix'] = self._parse_matrix_from_xml(camera_elem)

        # 解析畸变系数
        dist_elem = root.find("distortion_coefficients")
        if dist_elem is not None:
            data['dist_coeffs'] = self._parse_matrix_from_xml(dist_elem)

        # 解析旋转向量
        rvecs_elem = root.find("rotation_vectors")
        if rvecs_elem is not None:
            data['rvecs'] = []
            for rvec_elem in rvecs_elem:
                if rvec_elem.tag.startswith("rvec_"):
                    data['rvecs'].append(self._parse_matrix_from_xml(rvec_elem))

        # 解析平移向量
        tvecs_elem = root.find("translation_vectors")
        if tvecs_elem is not None:
            data['tvecs'] = []
            for tvec_elem in tvecs_elem:
                if tvec_elem.tag.startswith("tvec_"):
                    data['tvecs'].append(self._parse_matrix_from_xml(tvec_elem))

        # 解析元数据
        metadata_fields = ['calibration_date', 'board_params', 'image_size',
                          'per_view_errors', 'save_timestamp', 'file_formats']

        for field in metadata_fields:
            elem = root.find(field)
            if elem is not None and elem.text:
                try:
                    # 尝试解析为Python对象
                    data[field] = eval(elem.text)
                except:
                    data[field] = elem.text

        return data

    def _parse_matrix_from_xml(self, matrix_elem) -> np.ndarray:
        """从XML元素解析矩阵"""
        rows_elem = matrix_elem.find("rows")
        cols_elem = matrix_elem.find("cols")
        data_elem = matrix_elem.find("data")

        if rows_elem is None or cols_elem is None or data_elem is None:
            return np.array([])

        rows = int(rows_elem.text)
        cols = int(cols_elem.text)
        data_text = data_elem.text.strip()

        # 解析数据
        values = [float(x) for x in data_text.split()]

        # 重塑为矩阵
        if rows == 1:
            return np.array(values).reshape(1, -1)
        else:
            return np.array(values).reshape(rows, cols)

    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """获取文件信息"""
        if not os.path.exists(filepath):
            return {}

        stat = os.stat(filepath)
        filename = os.path.basename(filepath)
        name_parts = filename.split('_')

        info = {
            'filename': filename,
            'path': filepath,
            'size_kb': stat.st_size / 1024,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'format': os.path.splitext(filename)[1].lstrip('.').lower()
        }

        # 解析时间戳
        if len(name_parts) >= 2:
            timestamp_str = f"{name_parts[0]}_{name_parts[1]}"
            try:
                info['timestamp'] = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S").isoformat()
            except:
                pass

        return info

    def list_calibration_files(self, directory: str = ".") -> list:
        """列出目录中的所有标定文件"""
        calibration_files = []

        if not os.path.exists(directory):
            return calibration_files

        for filename in os.listdir(directory):
            if filename.endswith(('.npz', '.json', '.xml')):
                filepath = os.path.join(directory, filename)
                if 'calibration' in filename.lower():
                    file_info = self.get_file_info(filepath)
                    calibration_files.append(file_info)

        # 按修改时间排序
        calibration_files.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return calibration_files

    def convert_format(self, input_file: str, output_format: str,
                      output_dir: str = None) -> str:
        """
        转换标定文件格式

        参数:
        input_file: 输入文件路径
        output_format: 输出格式 ('npz', 'json', 'xml')
        output_dir: 输出目录，默认使用输入文件目录

        返回:
        输出文件路径
        """
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的输出格式: {output_format}")

        # 加载原始文件
        data, input_format = self.load_calibration_file(input_file)

        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
        os.makedirs(output_dir, exist_ok=True)

        # 生成输出文件名
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{input_name}_converted.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        # 保存为新格式
        if output_format == 'npz':
            self._save_npz(data, output_path)
        elif output_format == 'json':
            processed_data = self._prepare_for_json(data)
            self._save_json(processed_data, output_path)
        elif output_format == 'xml':
            processed_data = self._prepare_for_json(data)
            self._save_xml(processed_data, output_path)

        print(f"✅ 格式转换完成: {input_format.upper()} -> {output_format.upper()}")
        print(f"   输出文件: {output_filename}")

        return output_path


def main():
    """主函数 - 命令行工具"""
    import argparse

    parser = argparse.ArgumentParser(description='多格式标定文件管理器')
    parser.add_argument('action', choices=['save', 'load', 'convert', 'list'],
                       help='操作类型')
    parser.add_argument('--input', '-i', help='输入文件路径')
    parser.add_argument('--output', '-o', help='输出文件路径或目录')
    parser.add_argument('--format', '-f', choices=['npz', 'json', 'xml'],
                       help='输出格式')
    parser.add_argument('--data', '-d', help='标定数据文件 (NPZ格式，用于save操作)')

    args = parser.parse_args()

    manager = CalibrationFileManager()

    try:
        if args.action == 'save':
            if not args.data:
                print("❌ save 操作需要指定 --data 参数")
                return

            # 加载要保存的数据
            save_data = np.load(args.data)
            data_dict = {key: save_data[key] for key in save_data.keys()}

            # 保存为多种格式
            output_dir = args.output or "./calibration_results"
            saved_files = manager.save_calibration_multi_format(data_dict, output_dir)

            print("\n📁 保存的文件:")
            for fmt, filepath in saved_files.items():
                print(f"  • {fmt.upper()}: {filepath}")

        elif args.action == 'load':
            if not args.input:
                print("❌ load 操作需要指定 --input 参数")
                return

            # 加载标定文件
            data, format_type = manager.load_calibration_file(args.input)

            print(f"\n📋 加载的标定数据 ({format_type.upper()}):")
            print(f"  • 相机矩阵: {data.get('camera_matrix', 'N/A')}")
            print(f"  • 畸变系数: {data.get('dist_coeffs', 'N/A')}")
            print(f"  • 外参数量: {len(data.get('rvecs', []))}")

        elif args.action == 'convert':
            if not args.input or not args.format:
                print("❌ convert 操作需要指定 --input 和 --format 参数")
                return

            output_path = manager.convert_format(args.input, args.format, args.output)
            print(f"✅ 转换完成: {output_path}")

        elif args.action == 'list':
            directory = args.input or "."
            files = manager.list_calibration_files(directory)

            print(f"\n📂 标定文件列表 ({directory}):")
            for file_info in files:
                print(f"  • {file_info['filename']} ({file_info['format'].upper()}) - {file_info['size_kb']:.1f} KB")

    except Exception as e:
        print(f"❌ 操作失败: {e}")


if __name__ == "__main__":
    main()
