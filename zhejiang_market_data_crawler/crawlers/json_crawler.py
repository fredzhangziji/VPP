"""
JSON爬虫，用于处理JSON格式的数据
"""

import json
from abc import abstractmethod
from .base_crawler import BaseCrawler
from utils.http_client import get, post

class JSONCrawler(BaseCrawler):
    """JSON爬虫，用于处理JSON格式的数据"""
    
    def __init__(self, name):
        """
        初始化JSON爬虫
        
        Args:
            name: 爬虫名称
        """
        super().__init__(name)
    
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含数据的DataFrame
        """
        # 获取请求参数
        url, method, headers, params, data = self.get_request_params(start_date, end_date)
        
        try:
            # 发送请求
            if method.upper() == 'GET':
                response = get(url, params=params, headers=headers)
            else:
                response = post(url, data=data, json=params, headers=headers)
            
            # 解析响应
            json_data = self.parse_response(response)
            
            # 转换数据
            return self.transform_data(json_data)
        except Exception as e:
            self.logger.error(f"获取数据失败: {e}", exc_info=True)
            return None
    
    @abstractmethod
    def get_request_params(self, start_date=None, end_date=None):
        """
        获取请求参数
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            url: 请求URL
            method: 请求方法，GET或POST
            headers: 请求头
            params: 请求参数（GET）或JSON数据（POST）
            data: 表单数据（POST）
        """
        pass
    
    def parse_response(self, response):
        """
        解析响应
        
        Args:
            response: 响应对象
        
        Returns:
            json_data: JSON数据
        """
        try:
            return response.json()
        except json.JSONDecodeError as e:
            self.logger.error(f"解析JSON响应失败: {e}")
            self.logger.debug(f"响应内容: {response.text[:500]}...")
            raise 