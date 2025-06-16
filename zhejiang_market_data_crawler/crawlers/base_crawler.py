"""
爬虫基类，定义通用方法和接口
"""

import os
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from utils.logger import setup_logger
from utils.http_client import get, post, download_file
from utils.db_helper import save_to_db
from utils.config import DATA_DIR

class BaseCrawler(ABC):
    """爬虫基类，定义通用方法和接口"""
    
    def __init__(self, name):
        """
        初始化爬虫
        
        Args:
            name: 爬虫名称，用于日志记录和临时文件命名
        """
        self.name = name
        self.logger = setup_logger(f'crawler.{name}')
        self.temp_dir = os.path.join(DATA_DIR, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    @abstractmethod
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含数据的DataFrame
        """
        pass
    
    @abstractmethod
    def parse_response(self, response):
        """
        解析响应
        
        Args:
            response: 响应对象
        
        Returns:
            df: 包含解析结果的DataFrame
        """
        pass
    
    @abstractmethod
    def transform_data(self, data):
        """
        转换数据
        
        Args:
            data: 原始数据
        
        Returns:
            df: 转换后的DataFrame
        """
        pass
    
    def save_to_db(self, df, update_columns=None):
        """
        保存数据到数据库
        
        Args:
            df: 包含数据的DataFrame
            update_columns: 当记录已存在时要更新的列，默认为None（更新所有列）
        
        Returns:
            success: 是否保存成功
        """
        if df.empty:
            self.logger.warning("没有数据需要保存")
            return False
        
        try:
            return save_to_db(df, update_columns=update_columns)
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            return False
    
    def run(self, start_date=None, end_date=None, update_columns=None):
        """
        运行爬虫
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
            update_columns: 当记录已存在时要更新的列，默认为None（更新所有列）
        
        Returns:
            success: 是否运行成功
        """
        try:
            self.logger.info(f"开始运行爬虫: {self.name}")
            self.logger.info(f"时间范围: {start_date or '无限制'} 至 {end_date or '无限制'}")
            
            # 获取数据
            df = self.fetch_data(start_date, end_date)
            
            if df is not None and not df.empty:
                # 保存数据到数据库
                self.save_to_db(df, update_columns=update_columns)
                self.logger.info(f"爬虫 {self.name} 运行完成，共获取 {len(df)} 条数据")
                return True
            else:
                self.logger.warning(f"爬虫 {self.name} 未获取到数据")
                return False
        except Exception as e:
            self.logger.error(f"爬虫 {self.name} 运行失败: {e}", exc_info=True)
            return False
    
    def get_temp_file_path(self, filename):
        """
        获取临时文件路径
        
        Args:
            filename: 文件名
        
        Returns:
            path: 临时文件路径
        """
        return os.path.join(self.temp_dir, f"{self.name}_{filename}")
    
    def format_date(self, date_obj=None):
        """
        格式化日期
        
        Args:
            date_obj: 日期对象，默认为当前日期
        
        Returns:
            date_str: 格式化后的日期字符串，格式为YYYY-MM-DD
        """
        if date_obj is None:
            date_obj = datetime.now()
        
        if isinstance(date_obj, str):
            try:
                date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
            except ValueError:
                self.logger.error(f"日期格式错误: {date_obj}，应为YYYY-MM-DD")
                return None
        
        return date_obj.strftime('%Y-%m-%d') 