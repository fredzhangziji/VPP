"""
集中式日志配置，配合logging.yaml文件使用。
"""

from logging.config import dictConfig
import os
import yaml
import sys

# 拿到绝对路径
if hasattr(sys.modules['__main__'], '__file__'):
    main_file = os.path.abspath(sys.modules['__main__'].__file__)
    base_dir = os.path.dirname(main_file)
else:
    base_dir = os.getcwd()

# yaml配置文件必须和当前日志配置脚本在同级目录下
cfg_path = os.path.join(os.path.dirname(__file__), 'logging.yaml')
with open(cfg_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)

for handler in config.get('handlers', {}).values():
    fn = handler.get('filename')
    if fn and not os.path.isabs(fn):
        log_path = os.path.join(base_dir, fn)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        handler['filename'] = log_path

dictConfig(config)