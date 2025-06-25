"""
日前市场出清负荷侧电价爬虫
用于抓取浙江电力市场的日前市场出清负荷侧电价数据
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import TARGET_TABLE, get_api_cookie

class DayAheadPriceCrawler(JSONCrawler):
    """日前市场出清负荷侧电价爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化日前市场出清负荷侧电价爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('day_ahead_price')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        
        # 日前市场出清负荷侧电价字段名
        self.field_name = "day_ahead_price"  # 日前市场出清负荷侧电价(元/MWh)
    
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
        
        self.logger.info(f"找到 {len(data_list)} 条价格记录")
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 遍历找到价格数据
        for item_data in data_list:
            # 创建时间点和价格值列表
            timestamps = []
            price_values = []
            
            # 解析v2, v4, v6...v96的数据，每个数据点对应15分钟
            for i in range(1, 97):
                if i % 2 == 0:  # 只处理偶数索引，即v2, v4, v6...v96
                    key = f"v{i}"
                    if key in item_data and item_data[key] is not None:
                        # 计算时间戳：v2对应00:30，v4对应01:00，以此类推
                        hour = ((i // 2) * 30) // 60  # 小时部分
                        minute = ((i // 2) * 30) % 60  # 分钟部分
                        
                        # 如果是v96，对应的是下一天的00:00
                        if i == 96:
                            timestamp = base_date + timedelta(days=1)
                            timestamp = timestamp.replace(hour=0, minute=0)
                        else:
                            timestamp = base_date.replace(hour=hour, minute=minute)
                        
                        # 添加到列表
                        timestamps.append(timestamp)
                        price_values.append(float(item_data[key]))
            
            # 如果有数据，返回DataFrame
            if timestamps:
                df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: price_values
                })
                self.logger.info(f"成功解析浙江电网的价格数据，共 {len(timestamps)} 条记录")
                return df
        
        # 如果没有找到浙江电网的数据
        self.logger.warning("未能找到浙江电网的日前市场出清负荷侧电价数据")
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
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/marketQuery/findRqsccqfhcdj"
        
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
                "zjNumber": "0200068",  # 日前市场出清负荷侧电价的zjNumber
                "measType": "98109002"  # 日前市场出清负荷侧电价的measType
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
            json_data: JSON数据
        """
        # 如果响应是字符串，尝试解析为JSON
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                self.logger.error(f"解析JSON响应失败: {e}")
                self.logger.debug(f"响应内容: {response[:500]}...")
                return None
        
        # 如果响应是响应对象，尝试获取JSON
        try:
            return response.json()
        except Exception as e:
            self.logger.error(f"获取响应的JSON失败: {e}")
            self.logger.debug(f"响应内容: {response.text[:500]}...")
            return None
    
    def format_date(self, date_str):
        """
        格式化日期字符串
        
        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
            
        Returns:
            formatted_date: 格式化后的日期字符串，格式为YYYY-MM-DD
        """
        if not date_str:
            return datetime.now().strftime('%Y-%m-%d')
        
        try:
            # 尝试解析日期字符串
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            self.logger.error(f"无效的日期格式: {date_str}，应为YYYY-MM-DD")
            return None
            
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含数据的DataFrame
        """
        # 格式化日期
        start_date = self.format_date(start_date)
        
        # 如果未指定结束日期，则与开始日期相同
        if not end_date:
            end_date = start_date
        else:
            end_date = self.format_date(end_date)
        
        if not start_date or not end_date:
            self.logger.error("无效的日期格式")
            return pd.DataFrame()
        
        # 创建日期序列
        start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 计算日期差
        date_diff = (end_date_obj - start_date_obj).days + 1
        self.logger.info(f"共需要处理 {date_diff} 天的数据")
        
        # 如果日期范围过大，给出警告
        if date_diff > 100:
            self.logger.warning(f"日期范围过大 ({date_diff} 天)，可能需要较长时间")
        
        # 存储所有数据的DataFrame
        all_data = pd.DataFrame()
        
        # 逐日获取数据
        current_date_obj = start_date_obj
        while current_date_obj <= end_date_obj:
            current_date = current_date_obj.strftime('%Y-%m-%d')
            self.logger.info(f"正在处理日期: {current_date}")
            
            # 获取请求参数
            request_params = self.get_request_params(current_date)
            
            # 发送请求
            response_text = self.send_request(
                request_params['url'],
                request_params['method'],
                request_params['headers'],
                request_params['params'],
                request_params['data'],
                request_params['cookies']
            )
            
            if response_text:
                # 解析响应
                json_data = self.parse_response(response_text)
                
                # 转换数据
                df = self.transform_data(json_data, current_date)
                
                if not df.empty:
                    # 将数据添加到总DataFrame
                    if all_data.empty:
                        all_data = df
                    else:
                        all_data = pd.concat([all_data, df], ignore_index=True)
            
            # 移动到下一天
            current_date_obj += timedelta(days=1)
        
        self.logger.info(f"数据获取完成，共 {len(all_data)} 条记录")
        return all_data