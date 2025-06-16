"""
日志配置，使用简单的日志配置
"""

import os
import logging
import logging.config
import yaml
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

def setup_logger(name='zhejiang_market_crawler'):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
    
    Returns:
        logger: 配置好的日志记录器
    """
    # 获取日志记录器
    logger = logging.getLogger(name)
    
    # 如果日志记录器已经有处理器，说明已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 创建日志目录
    log_dir = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置日志文件路径
    log_file = os.path.join(log_dir, f'{name}.log')
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    
    # 设置日志级别
    logger.setLevel(logging.DEBUG)
    
    return logger 