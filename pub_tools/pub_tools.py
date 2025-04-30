"""
此模块包含了所有项目都需要使用的公共工具库函数。
"""

import platform
import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import logging
import pub_tools.logging_config
logger = logging.getLogger(__name__)

def set_system_font():
    if platform.system() == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Microsoft/SimHei.ttf'
        ]
    else:  # Linux/Windows
        font_paths = [
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
        ]
    
    # 尝试找到可用的中文字体
    for path in font_paths:
        if os.path.exists(path):
            return path
            
    # 如果找不到预定义的字体，使用系统默认中文字体
    font_path = [f for f in fm.findSystemFonts() if 'noto' in f.lower() or 'ping' in f.lower() or 'heiti' in f.lower()]
    try:
        if font_path:
            plt.rcParams['font.family'] = ['sans-serif']
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        else:
            # 使用内置的中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        logger.info(f'当前设置字体: {font_prop}')
    except Exception as e:
        logger.error(f"设置中文字体时出错: {str(e)}")
        # 使用备选方案
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']