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

def get_system_font_path():
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
            # choose first available fallback font
            chosen = font_path[0]
            return chosen
        else:
            # no fallback file; return None
            return None
    except Exception as e:
        logger.error(f"设置中文字体时出错: {str(e)}")
        return None
    return None