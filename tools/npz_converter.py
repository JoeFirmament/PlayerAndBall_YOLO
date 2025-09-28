#!/usr/bin/env python3
"""
相机标定npz文件格式转换工具

支持将npz格式转换为其他常用格式：
- JSON格式
- YAML格式
- XML格式
- 二进制格式
- MATLAB .mat格式
"""

import numpy as np
import json
import yaml
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

class CalibrationConverter:
    """标定文件格式转换器"""

    def __init__(self):
        self.supported_formats = {
            'json': self.to_json,
            'yaml': self.to_yaml,
            'xml': self.to_xml,
            'binary': self.to_binary,
            'mat': self.to_matlab
        }

    def convert(self, npz_path, output_format, output_path=None):
        """转换npz文件格式"""
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        if output_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {output_format}. "
                           f"Supported: {list(self.supported_formats.keys())}")

        # 加载npz文件
        data = np.load(npz_path)

        # 生成输出路径
        if output_path is None:
            base_name = Path(npz_path).stem
            output_path = f"{base_name}.{output_format}"

        # 执行转换
        converter = self.supported_formats[output_format]
        converter(data, output_path)

        print(f"Converted {npz_path} to {output_path}")
        return output_path

    def to_json(self, data, output_path):
        """转换为JSON格式"""
        json_data = {}

        for key in data.keys():
            value = data[key]

            # 处理numpy数组
            if isinstance(value, np.ndarray):
                if value.ndim == 0:  # 标量
                    json_data[key] = float(value)
                elif value.ndim == 1:  # 向量
                    json_data[key] = value.tolist()
                elif value.ndim == 2:  # 矩阵
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value.tolist()
            else:
                # 处理其他类型
                try:
                    json_data[key] = value.tolist() if hasattr(value, 'tolist') else value
                except:
                    json_data[key] = str(value)

        # 添加元数据
        json_data['_metadata'] = {
            'converted_from': 'npz',
            'converted_at': datetime.now().isoformat(),
            'converter_version': '1.0'
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

    def to_yaml(self, data, output_path):
        """转换为YAML格式"""
        yaml_data = {}

        for key in data.keys():
            value = data[key]

            # 处理numpy数组
            if isinstance(value, np.ndarray):
                if value.ndim == 0:  # 标量
                    yaml_data[key] = float(value)
                else:
                    yaml_data[key] = value.tolist()
            else:
                try:
                    yaml_data[key] = value.tolist() if hasattr(value, 'tolist') else value
                except:
                    yaml_data[key] = str(value)

        # 添加元数据
        yaml_data['_metadata'] = {
            'converted_from': 'npz',
            'converted_at': datetime.now().isoformat(),
            'converter_version': '1.0'
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

    def to_xml(self, data, output_path):
        """转换为XML格式"""
        root = ET.Element("calibration_data")
        root.set("converted_from", "npz")
        root.set("converted_at", datetime.now().isoformat())

        for key in data.keys():
            value = data[key]

            element = ET.SubElement(root, "parameter")
            element.set("name", key)

            if isinstance(value, np.ndarray):
                if value.ndim == 0:  # 标量
                    element.set("type", "scalar")
                    element.text = str(float(value))
                elif value.ndim == 1:  # 向量
                    element.set("type", "vector")
                    element.set("size", str(len(value)))
                    for i, v in enumerate(value):
                        item = ET.SubElement(element, "item")
                        item.set("index", str(i))
                        item.text = str(float(v))
                elif value.ndim == 2:  # 矩阵
                    element.set("type", "matrix")
                    element.set("rows", str(value.shape[0]))
                    element.set("cols", str(value.shape[1]))
                    for i in range(value.shape[0]):
                        row = ET.SubElement(element, "row")
                        row.set("index", str(i))
                        for j in range(value.shape[1]):
                            col = ET.SubElement(row, "col")
                            col.set("index", str(j))
                            col.text = str(float(value[i, j]))
            else:
                element.set("type", "other")
                element.text = str(value)

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    def to_binary(self, data, output_path):
        """转换为自定义二进制格式"""
        with open(output_path, 'wb') as f:
            # 写入魔数
            f.write(b'CALIB\x00\x01')

            # 写入参数数量
            f.write(len(data).to_bytes(4, byteorder='little'))

            for key, value in data.items():
                # 写入键名
                key_bytes = key.encode('utf-8')
                f.write(len(key_bytes).to_bytes(4, byteorder='little'))
                f.write(key_bytes)

                # 写入数据类型和形状信息
                if isinstance(value, np.ndarray):
                    f.write(b'ARRAY')
                    f.write(value.ndim.to_bytes(4, byteorder='little'))

                    # 写入形状
                    for dim in value.shape:
                        f.write(dim.to_bytes(8, byteorder='little'))

                    # 写入数据类型
                    dtype_str = str(value.dtype)
                    dtype_bytes = dtype_str.encode('utf-8')
                    f.write(len(dtype_bytes).to_bytes(4, byteorder='little'))
                    f.write(dtype_bytes)

                    # 写入数组数据
                    if value.dtype == np.float64:
                        f.write(value.astype(np.float32).tobytes())
                    else:
                        f.write(value.tobytes())
                else:
                    f.write(b'SCALAR')
                    value_str = str(value)
                    value_bytes = value_str.encode('utf-8')
                    f.write(len(value_bytes).to_bytes(4, byteorder='little'))
                    f.write(value_bytes)

    def to_matlab(self, data, output_path):
        """转换为MATLAB .mat格式"""
        try:
            import scipy.io
        except ImportError:
            raise ImportError("scipy is required for MATLAB format conversion. "
                            "Install with: pip install scipy")

        # 转换numpy数组为MATLAB兼容格式
        matlab_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                matlab_data[key] = value
            else:
                matlab_data[key] = np.array([value])

        scipy.io.savemat(output_path, matlab_data)
        print(f"Saved MATLAB file: {output_path}")

    def list_formats(self):
        """列出支持的格式"""
        return list(self.supported_formats.keys())

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Convert camera calibration npz files to other formats')
    parser.add_argument('input', help='Input npz file path')
    parser.add_argument('format', help='Output format (json, yaml, xml, binary, mat)')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-l', '--list', action='store_true', help='List supported formats')

    args = parser.parse_args()

    converter = CalibrationConverter()

    if args.list:
        print("Supported formats:")
        for fmt in converter.list_formats():
            print(f"  - {fmt}")
        return

    if not args.input or not args.format:
        parser.print_help()
        return

    try:
        output_path = converter.convert(args.input, args.format, args.output)
        print(f"Conversion completed: {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
