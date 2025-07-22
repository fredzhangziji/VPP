"""
配置文件，包含数据库连接信息和其他配置
"""

import os
import yaml
from pathlib import Path
from pub_tools.const import DB_CONFIG_ZHEJIANG_MARKET

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, 'data')

# 数据库配置
DB_CONFIG = DB_CONFIG_ZHEJIANG_MARKET

# 目标数据表名
TARGET_TABLE = 'power_market_data'

# 请求间隔时间（秒）
REQUEST_INTERVAL = 2

# 爬虫并发数（如果使用并发）
CONCURRENCY = 3

# 请求超时时间（秒）
REQUEST_TIMEOUT = 30

# 最大重试次数
MAX_RETRIES = 3

# 加载自定义配置（如果存在）
def load_custom_config():
    config_path = os.path.join(ROOT_DIR, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            custom_config = yaml.safe_load(f)
            if custom_config:
                # 更新数据库配置
                if 'database' in custom_config:
                    DB_CONFIG.update(custom_config['database'])
                
                # 更新其他配置
                global REQUEST_INTERVAL, CONCURRENCY, REQUEST_TIMEOUT, MAX_RETRIES, TARGET_TABLE
                if 'request_interval' in custom_config:
                    REQUEST_INTERVAL = custom_config['request_interval']
                if 'concurrency' in custom_config:
                    CONCURRENCY = custom_config['concurrency']
                if 'request_timeout' in custom_config:
                    REQUEST_TIMEOUT = custom_config['request_timeout']
                if 'max_retries' in custom_config:
                    MAX_RETRIES = custom_config['max_retries']
                if 'target_table' in custom_config:
                    TARGET_TABLE = custom_config['target_table']

# 临时硬编码API Cookie配置 - 所有爬虫共用一个Cookie
API_COOKIE = "Huyi3DKbjTb1O=606_J_RlMbS8w.zV8fD9nR2.4F4.QULap7xzd50sC3JIof9xK2TPxJCTG4zbhGsVMrLvDCBl2w29OGag9lP8sfMq; BIGipServerxyddljy_newmain_20080=1822890412.28750.0000; Gray-Tag=79793033; ClientTag=OUTNET_BROWSE; CurrentRoute=/dashboard; X-Token=undefined; Admin-Token=137858cc97bd45d57d65e792baa6f8b0c5559f5a2f3e7b52639101bc6d104fe6906fb31c01650246bf7993a87629709b.5b89315c3ff76c62c74840c408d1222b99091b85; X-Ticket=137858cc97bd45d57d65e792baa6f8b0c5559f5a2f3e7b52639101bc6d104fe6906fb31c01650246bf7993a87629709b.5b89315c3ff76c62c74840c408d1222b99091b85"

def get_api_cookie():
    """
    获取API Cookie
    
    Returns:
        cookie: API Cookie字符串
    """
    # 尝试从数据库获取最新的Cookie（未实现）
    # TODO: 从数据库获取最新的Cookie
    
    # 如果获取失败，则使用默认的Cookie
    return API_COOKIE

# 初始化时加载自定义配置
load_custom_config() 