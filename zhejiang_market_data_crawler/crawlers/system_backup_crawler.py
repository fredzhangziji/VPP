"""
系统备用爬虫
用于抓取浙江电力市场的系统备用数据
"""

import json
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import TARGET_TABLE, get_api_cookie

class SystemBackupCrawler(JSONCrawler):
    """系统备用爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化系统备用爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('system_backup')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        
        # 数据库字段名
        self.field_name = "system_reserve"  # 系统备用(MW)
    
    def transform_data(self, json_data, query_date=None):
        """
        转换JSON数据为DataFrame
        
        Args:
            json_data: JSON格式的数据
            query_date: 查询日期，格式为YYYY-MM-DD
            
        Returns:
            df: 包含转换后数据的DataFrame
        """
        # 检查JSON数据是否为空
        if not json_data:
            self.logger.error("JSON数据为空")
            return pd.DataFrame()
        
        # 检查JSON数据是否为字典类型
        if not isinstance(json_data, dict):
            self.logger.error(f"JSON数据不是字典类型: {type(json_data)}")
            return pd.DataFrame()
        
        # 检查JSON数据是否包含status字段
        if 'status' not in json_data:
            self.logger.error("JSON数据不包含status字段")
            return pd.DataFrame()
        
        # 检查状态是否正常
        if json_data.get('status') != 0:
            self.logger.error(f"API返回错误状态: {json_data.get('status')}")
            message = json_data.get('message', 'Unknown error')
            self.logger.error(f"错误信息: {message}")
            return pd.DataFrame()
        
        # 获取数据列表
        data = json_data.get('data', {})
        data_list = data.get('list', [])
        
        if not data_list:
            self.logger.warning("没有找到数据")
            return pd.DataFrame()
        
        self.logger.info(f"找到 {len(data_list)} 条系统备用记录")
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 遍历找到浙江电网的数据
        for item_data in data_list:
            # 创建时间点和系统备用值列表
            timestamps = []
            backup_values = []
            
            # 解析v1到v96的数据，每个数据点对应15分钟
            for i in range(1, 97):
                key = f"v{i}"
                if key in item_data and item_data[key] is not None:
                    # 正确计算时间戳：v1对应00:15，v96对应下一天00:00
                    hour = (i * 15) // 60  # 小时部分
                    minute = (i * 15) % 60  # 分钟部分
                    
                    # 如果是v96，对应的是下一天的00:00
                    if i == 96:
                        timestamp = base_date + timedelta(days=1)
                        timestamp = timestamp.replace(hour=0, minute=0)
                    else:
                        timestamp = base_date.replace(hour=hour, minute=minute)
                    
                    # 添加到列表
                    timestamps.append(timestamp)
                    backup_values.append(float(item_data[key]))
            
            # 如果有数据，返回DataFrame
            if timestamps:
                df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: backup_values
                })
                self.logger.info(f"成功解析浙江电网的数据，共 {len(timestamps)} 条记录")
                return df
        
        # 如果没有找到浙江电网的数据
        self.logger.warning("未能找到浙江电网的系统备用数据")
        return pd.DataFrame()
    
    def get_request_params(self, start_date=None, end_date=None):
        """
        获取请求参数
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            dict: 包含请求所需的所有参数的字典
        """
        # 默认获取当天的数据
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        # 如果未指定结束日期，则使用开始日期
        if not end_date:
            end_date = start_date
        
        # 请求URL - 系统备用数据的URL
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/schedule/sysBackup"
        
        # 请求头
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
            "ClientTag": "OUTNET_BROWSE",
            "Connection": "keep-alive",
            "Content-Type": "application/json;charset=UTF-8",
            "CurrentRoute": "/pxf-settlement-outnetpub-phbzj/columnHomeLeftMenuNew",
            "Host": "zjpx.com.cn",
            "Origin": "https://zjpx.com.cn",
            "Referer": "https://zjpx.com.cn/pxf-settlement-outnetpub-phbzj/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "sec-ch-ua": "\"Google Chrome\";v=\"137\", \"Chromium\";v=\"137\", \"Not/A)Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"macOS\""
        }
        
        # 请求参数
        payload = {
            "pageInfo": {
                "pageSize": 10,
                "pageNum": 1
            },
            "data": {
                "queryDate": start_date,
                "zjNumber": "0200089",  # 系统备用使用的zjNumber
                "measType": None
            }
        }
        
        # 返回请求参数字典
        return {
            "url": url,
            "method": "POST",
            "headers": headers,
            "params": None,
            "data": json.dumps(payload),
            "cookies": self.cookie
        }
    
    def send_request(self, url, method, headers=None, params=None, data=None, cookies=None):
        """
        发送HTTP请求
        
        Args:
            url: 请求URL
            method: 请求方法（GET或POST）
            headers: 请求头
            params: URL参数
            data: 请求体
            cookies: Cookie
        
        Returns:
            response: 响应对象
        """
        try:
            # 处理cookies参数
            cookies_dict = None
            if cookies:
                if isinstance(cookies, str):
                    # 将字符串cookies转换为字典
                    cookies_dict = {}
                    for item in cookies.split(';'):
                        if '=' in item:
                            name, value = item.strip().split('=', 1)
                            cookies_dict[name] = value
                else:
                    cookies_dict = cookies
            
            if method.upper() == "GET":
                response = get(url, params=params, headers=headers, cookies=cookies_dict)
            else:
                response = post(url, data=data, json=params, headers=headers, cookies=cookies_dict)
            
            return response
        except Exception as e:
            self.logger.error(f"请求失败: {e}", exc_info=True)
            raise
    
    def parse_response(self, response, query_date=None):
        """
        解析响应
        
        Args:
            response: 响应对象
            query_date: 查询日期
        
        Returns:
            json_data: 解析后的JSON数据
        """
        try:
            json_data = response.json()
            
            # 检查状态
            if json_data.get('status') != 0:
                self.logger.error(f"API返回错误状态: {json_data.get('status')}")
                message = json_data.get('message', 'Unknown error')
                self.logger.error(f"错误信息: {message}")
            
            return json_data
        except Exception as e:
            self.logger.error(f"解析响应失败: {e}", exc_info=True)
            self.logger.debug(f"响应内容: {response.text[:500]}...")
            raise
    
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含数据的DataFrame
        """
        # 默认获取当天的数据
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        # 如果未指定结束日期，则使用开始日期
        if not end_date:
            end_date = start_date
        
        # 存储所有日期的数据
        all_data = []
        
        # 解析开始和结束日期
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 对每个日期获取数据
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            self.logger.info(f"获取系统备用数据: {date_str}")
            
            # 获取请求参数
            request_params = self.get_request_params(date_str, date_str)
            
            # 发送请求
            try:
                response = self.send_request(
                    request_params["url"],
                    request_params["method"],
                    headers=request_params["headers"],
                    params=request_params["params"],
                    data=request_params["data"],
                    cookies=request_params["cookies"]
                )
                
                # 解析响应
                json_data = self.parse_response(response, query_date=date_str)
                
                # 转换数据
                df = self.transform_data(json_data, query_date=date_str)
                
                # 如果获取到数据，则添加到结果中
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"成功获取 {date_str} 的系统备用数据，共 {len(df)} 条记录")
                else:
                    self.logger.warning(f"未找到 {date_str} 的系统备用数据")
                
            except Exception as e:
                self.logger.error(f"获取 {date_str} 的系统备用数据失败: {e}", exc_info=True)
            
            # 进入下一天
            current_date += timedelta(days=1)
        
        # 合并所有数据
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"共获取 {len(result)} 条系统备用数据记录")
            return result
        else:
            self.logger.warning("未获取到任何系统备用数据")
            return pd.DataFrame() 