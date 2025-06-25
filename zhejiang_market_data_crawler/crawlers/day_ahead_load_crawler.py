"""
日前负荷预测爬虫
用于抓取浙江电力市场的日前负荷预测数据
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import TARGET_TABLE, get_api_cookie

class DayAheadLoadCrawler(JSONCrawler):
    """日前负荷预测爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化日前负荷预测爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('day_ahead_load')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        
        # 浙江电网的负荷预测字段名
        self.field_name = "day_ahead_load_forecast"  # 全省负荷预测
    
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
        
        self.logger.info(f"找到 {len(data_list)} 条负荷预测记录")
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 遍历找到浙江电网的数据
        for city_data in data_list:
            city_name = city_data.get('devName')
            
            # 只处理浙江电网的数据
            if city_name == "浙江电网":                
                # 创建时间点和负荷预测值列表
                timestamps = []
                forecast_values = []
                
                # 解析v1到v96的数据，每个数据点对应15分钟
                for i in range(1, 97):
                    key = f"v{i}"
                    if key in city_data and city_data[key] is not None:
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
                        forecast_values.append(float(city_data[key]))
                
                # 如果有数据，返回DataFrame
                if timestamps:
                    df = pd.DataFrame({
                        'date_time': timestamps,
                        self.field_name: forecast_values
                    })
                    self.logger.info(f"成功解析浙江电网的数据，共 {len(timestamps)} 条记录")
                    return df
        
        # 如果没有找到浙江电网的数据
        self.logger.warning("未能找到浙江电网的负荷预测数据")
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
        
        # 请求URL
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/supplyAndDemand/dailyLoad"
        
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
                "zjNumber": "0200003",
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
            method: 请求方法，GET或POST
            headers: 请求头
            params: 请求参数
            data: 请求数据
            cookies: 请求cookies
            
        Returns:
            response_text: 响应文本
        """
        try:
            self.logger.info(f"正在请求 {url}")
            
            # 确保cookies是字典类型
            cookie_dict = {}
            if cookies:
                if isinstance(cookies, str):
                    # 如果是字符串，解析成字典
                    cookie_pairs = cookies.split(';')
                    for pair in cookie_pairs:
                        if '=' in pair:
                            key, value = pair.strip().split('=', 1)
                            cookie_dict[key] = value
                elif isinstance(cookies, dict):
                    # 如果已经是字典，直接使用
                    cookie_dict = cookies
            
            # 发送请求
            if method.upper() == 'GET':
                response = get(url, params=params, headers=headers, cookies=cookie_dict)
            else:  # POST
                response = post(url, data=data, params=params, headers=headers, cookies=cookie_dict)
            
            # 检查响应状态码
            if response.status_code != 200:
                self.logger.error(f"请求失败: 状态码 {response.status_code}")
                self.logger.debug(f"响应内容: {response.text[:500]}...")
                return None
            
            # 获取响应文本
            return response.text
        except Exception as e:
            self.logger.error(f"请求异常: {e}")
            return None
    
    def parse_response(self, response, query_date=None):
        """
        解析响应数据
        
        Args:
            response: 响应数据
            query_date: 查询日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含解析后数据的DataFrame
        """
        # 如果未指定查询日期，则使用当天的日期
        if not query_date:
            query_date = datetime.now().strftime('%Y-%m-%d')
        
        # 检查响应内容
        if not response:
            self.logger.error("响应为空")
            return pd.DataFrame()
        
        # 记录原始响应内容的前1000个字符
        self.logger.debug(f"原始响应内容: {response[:1000]}")
        
        try:
            # 解析JSON响应
            json_data = json.loads(response)
            self.logger.debug("成功解析JSON响应")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析错误: {e}")
            self.logger.debug(f"无法解析的响应内容: {response[:1000]}")
            return pd.DataFrame()
        
        # 使用transform_data方法解析数据
        return self.transform_data(json_data, query_date=query_date)
    
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取指定日期范围内的数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含所有数据的DataFrame
        """
        # 如果未指定开始日期，则使用当天的日期
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        # 如果未指定结束日期，则使用开始日期
        if not end_date:
            end_date = start_date
        
        # 将日期字符串转换为datetime对象
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 创建一个空的DataFrame来存储所有数据
        all_data = pd.DataFrame()
        
        # 记录缺失数据的日期
        missing_dates = []
        
        # 遍历日期范围
        current_dt = start_dt
        while current_dt <= end_dt:
            # 获取当前日期的字符串表示
            current_date = current_dt.strftime('%Y-%m-%d')
            
            # 获取请求参数
            request_params = self.get_request_params(current_date)
            
            # 发送请求
            response = self.send_request(**request_params)
            
            # 如果请求失败，继续下一个日期
            if not response:
                self.logger.warning(f"获取 {current_date} 的数据请求失败，可能是网络问题或服务器维护")
                missing_dates.append(current_date)
                current_dt += timedelta(days=1)
                continue
            
            # 解析响应数据
            df = self.parse_response(response, current_date)
            
            # 如果解析成功，添加到总数据中
            if not df.empty:
                self.logger.info(f"成功获取 {current_date} 的数据，共 {len(df)} 条记录")
                all_data = pd.concat([all_data, df], ignore_index=True)
            else:
                self.logger.warning(f"未获取到 {current_date} 的负荷预测数据，可能是官方尚未发布该日数据")
                missing_dates.append(current_date)
            
            # 增加一天
            current_dt += timedelta(days=1)
        
        # 如果有缺失的日期，输出汇总信息
        if missing_dates:
            self.logger.warning(f"以下日期的数据获取失败或为空: {', '.join(missing_dates)}")
        
        # 如果所有日期都失败，则返回空的DataFrame
        if all_data.empty:
            self.logger.error(f"所有日期 {start_date} 至 {end_date} 的数据获取失败")
            return pd.DataFrame()
        
        # 获取成功，返回合并后的数据
        self.logger.info(f"总共获取了 {len(all_data)} 条数据")
        
        # 对数据进行排序，确保时间戳是有序的
        all_data = all_data.sort_values(by='date_time')
        
        return all_data