# 等宽英文字体配置文件
# 专门为避免中文字体显示问题设计的配置
# 平台: linux

FONT_CONFIG = {
    'platform': 'linux',
    'fonts': {
        'mono': ['DejaVu Sans Mono', 'Liberation Mono', 'Courier New', 'monospace']
    },
    'recommended': {
        'button_font': ('DejaVu Sans Mono', 11, 'bold'),
        'button_secondary_font': ('DejaVu Sans Mono', 10),
        'entry_font': ('DejaVu Sans Mono', 11),
        'mono_font': ('DejaVu Sans Mono', 10),
        'title_font': ('DejaVu Sans Mono', 24, 'bold'),
        'subtitle_font': ('DejaVu Sans Mono', 14),
        'card_title_font': ('DejaVu Sans Mono', 16, 'bold'),
        'info_font': ('DejaVu Sans Mono', 11),
        'muted_font': ('DejaVu Sans Mono', 10),
        'tag_font': ('DejaVu Sans Mono', 9)
    }
}

# 使用方法:
# from font_config import FONT_CONFIG
# button_font = FONT_CONFIG['recommended']['button_font']
