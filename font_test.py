#!/usr/bin/env python3
"""
字体兼容性测试脚本
用于验证跨平台字体配置是否正常工作
"""

import tkinter as tk
import sys
import os
import warnings

# 环境配置
warnings.filterwarnings("ignore", category=UserWarning)
if sys.platform == "darwin":
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

class FontCompatibilityTest:
    """字体兼容性测试"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("字体兼容性测试")
        self.root.geometry("800x600")

        # 设置颜色（使用现代化配色）
        self.colors = {
            'bg': '#f8f9fa',
            'card': '#ffffff',
            'border': '#e9ecef',
            'text': '#212529',
            'text_muted': '#6c757d',
            'primary': '#6c757d'
        }

        self.root.configure(bg=self.colors['bg'])

        # 设置字体回退机制
        self.setup_font_fallbacks()
        self.setup_ui()

    def is_font_available(self, font_name):
        """检查字体是否可用"""
        try:
            test_font = (font_name, 10)
            test_label = tk.Label(self.root, text="test", font=test_font)
            test_label.destroy()
            return True
        except:
            return False

    def get_available_font(self, font_list):
        """从字体列表中获取第一个可用的字体"""
        for font_name in font_list:
            if self.is_font_available(font_name):
                return font_name
        return font_list[-1]  # 如果都没有，返回最后一个

    def setup_font_fallbacks(self):
        """设置跨平台字体回退机制"""
        if sys.platform == "win32":
            self.fonts = {
                'primary': ['Arial', 'Helvetica', 'sans-serif'],
                'mono': ['Courier New', 'Consolas', 'monospace']
            }
        elif sys.platform == "darwin":
            self.fonts = {
                'primary': ['Arial', 'Helvetica Neue', 'Helvetica', 'sans-serif'],
                'mono': ['Courier New', 'Monaco', 'monospace']
            }
        else:
            self.fonts = {
                'primary': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif'],
                'mono': ['Courier New', 'DejaVu Sans Mono', 'Liberation Mono', 'monospace']
            }

        # 获取系统中实际可用的字体
        self.primary_font = self.get_available_font(self.fonts['primary'])
        self.mono_font = self.get_available_font(self.fonts['mono'])

        print(f"🖥️  操作系统: {sys.platform}")
        print(f"🎨 主字体: {self.primary_font}")
        print(f"📝 等宽字体: {self.mono_font}")

        # 测试所有候选字体
        print("\n📋 字体可用性测试:")
        for category, font_list in self.fonts.items():
            print(f"\n{category.upper()} 字体:")
            for font_name in font_list:
                available = "✅" if self.is_font_available(font_name) else "❌"
                print(f"  {available} {font_name}")

    def create_card(self, parent, title=None):
        """创建现代化卡片"""
        card = tk.Frame(parent, bg=self.colors['card'], relief='flat', bd=0)
        card.configure(highlightbackground=self.colors['border'], highlightthickness=1)

        if title:
            header = tk.Frame(card, bg=self.colors['card'])
            header.pack(fill='x', padx=20, pady=(20, 15))
            title_label = tk.Label(header, text=title, font=(self.primary_font, 16, 'bold'),
                                 bg=self.colors['card'], fg=self.colors['text'])
            title_label.pack(side='left')

        content = tk.Frame(card, bg=self.colors['card'])
        content.pack(fill='both', expand=True, padx=20, pady=(0, 20))

        return card, content

    def setup_ui(self):
        """设置用户界面"""
        # 主容器
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill='both', expand=True, padx=20, pady=20)

        # 标题
        title_frame = tk.Frame(main_container, bg=self.colors['bg'])
        title_frame.pack(fill='x', pady=(0, 20))

        title_label = tk.Label(title_frame, text="字体兼容性测试",
                             font=(self.primary_font, 24, 'bold'),
                             bg=self.colors['bg'], fg=self.colors['text'])
        title_label.pack(anchor='w')

        subtitle_label = tk.Label(title_frame,
                                text=f"当前系统: {sys.platform} | 主字体: {self.primary_font} | 等宽字体: {self.mono_font}",
                                font=(self.primary_font, 12),
                                bg=self.colors['bg'], fg=self.colors['text_muted'])
        subtitle_label.pack(anchor='w', pady=(5, 0))

        # 字体测试卡片
        test_card, test_content = self.create_card(main_container, "🎨 字体显示测试")
        test_card.pack(fill='both', expand=True)

        # 创建测试内容
        test_items = [
            ("标题字体", (self.primary_font, 18, 'bold'), "这是标题字体测试"),
            ("正文字体", (self.primary_font, 12), "这是正文字体测试，包含中文字符"),
            ("等宽字体", (self.mono_font, 11), "这是等宽字体测试: ABC abc 123"),
            ("小字体", (self.primary_font, 9), "这是小字体测试（用于标签等）"),
            ("按钮字体", (self.primary_font, 11, 'bold'), "这是按钮字体测试"),
        ]

        for i, (name, font_config, sample_text) in enumerate(test_items):
            # 创建测试行
            test_frame = tk.Frame(test_content, bg=self.colors['card'])
            test_frame.pack(fill='x', pady=(0, 15))

            # 字体名称
            name_label = tk.Label(test_frame, text=f"{name}:", font=(self.primary_font, 11, 'bold'),
                                bg=self.colors['card'], fg=self.colors['text'], width=12, anchor='w')
            name_label.pack(side='left', padx=(0, 15))

            # 示例文字
            sample_label = tk.Label(test_frame, text=sample_text, font=font_config,
                                  bg=self.colors['card'], fg=self.colors['text'])
            sample_label.pack(side='left', expand=True, fill='x')

            # 字体信息
            font_info = f"{font_config[0]} {font_config[1]}"
            if len(font_config) > 2:
                font_info += f" {font_config[2]}"

            info_label = tk.Label(test_frame, text=font_info, font=(self.mono_font, 9),
                                bg=self.colors['border'], fg=self.colors['text_muted'],
                                padx=8, pady=2)
            info_label.pack(side='right', padx=(10, 0))

        # 状态信息
        status_card, status_content = self.create_card(main_container, "📊 系统状态")
        status_card.pack(fill='x', pady=(20, 0))

        # 系统信息
        info_items = [
            ("操作系统", sys.platform),
            ("Python版本", sys.version.split()[0]),
            ("Tkinter版本", tk.TkVersion),
            ("检测到的主字体", self.primary_font),
            ("检测到的等宽字体", self.mono_font),
        ]

        for label, value in info_items:
            info_frame = tk.Frame(status_content, bg=self.colors['card'])
            info_frame.pack(fill='x', pady=(0, 8))

            label_widget = tk.Label(info_frame, text=f"{label}:", font=(self.primary_font, 11),
                                  bg=self.colors['card'], fg=self.colors['text'], width=15, anchor='w')
            label_widget.pack(side='left', padx=(0, 15))

            value_widget = tk.Label(info_frame, text=str(value), font=(self.mono_font, 11),
                                  bg=self.colors['card'], fg=self.colors['primary'])
            value_widget.pack(side='left')

    def run(self):
        """运行测试"""
        self.root.mainloop()


if __name__ == "__main__":
    print("🚀 启动字体兼容性测试...")
    try:
        app = FontCompatibilityTest()
        app.run()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
