"""
实时市场出清总电量爬虫
用于抓取浙江电力市场的实时市场出清总电量数据
"""

import json
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import TARGET_TABLE, get_api_cookie
from utils.db_helper import save_to_db

class SpotClearedVolumeCrawler(JSONCrawler):
    """实时市场出清总电量爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化实时市场出清总电量爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('spot_cleared_volume')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        
        # 实时市场出清总电量字段名
        self.field_name = "spot_cleared_volume"  # 实时市场出清总电量
    
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
        
        self.logger.info(f"找到 {len(data_list)} 条实时市场出清总电量记录")
        
        # 调试: 输出第一个数据项的键
        if data_list:
            item_keys = list(data_list[0].keys())
            self.logger.debug(f"数据项包含以下键: {item_keys}")
            
            # 检查v1到v96是否存在
            v_keys = [f"v{i}" for i in range(1, 97) if f"v{i}" in data_list[0]]
            self.logger.debug(f"找到的v键: {v_keys}")
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 处理每个数据项
        for item_data in data_list:
            # 创建时间点和实时市场出清总电量值列表
            timestamps = []
            volume_values = []
            
            # 先检查v1到v96这些键的值
            v_keys = []
            for i in range(1, 97):
                key = f"v{i}"
                if key in item_data and item_data[key] is not None:
                    v_keys.append(key)
            
            # 输出找到的v键供调试
            self.logger.debug(f"找到 {len(v_keys)} 个有效的v键")
            if v_keys:
                self.logger.debug(f"v键示例: {v_keys[:10]}...")
                
                # 输出这些键对应的值类型和示例
                for key in v_keys[:5]:
                    value = item_data[key]
                    self.logger.debug(f"键 {key} 的值类型: {type(value)}, 值: {value}")
            
            # 处理v1到v96的数据
            # 根据用户确认的数据格式：
            # - v2对应00:30，v3为null，v4对应01:00，v5为null，v6对应01:30...
            # - 只有偶数v有有效值，奇数v为null
            # - v94对应23:30，v96对应次日00:00
            for i in range(2, 97, 2):  # 只处理偶数v（v2, v4, v6...v96）
                key = f"v{i}"
                if key in item_data and item_data[key] is not None:
                    # 特殊情况：v96对应次日00:00
                    if i == 96:
                        hour = 0
                        minute = 0
                        day_offset = 1
                    else:
                        # 按照用户确认的映射关系：
                        # v2->00:30, v4->01:00, v6->01:30...v94->23:30
                        
                        # 计算时间点
                        idx = i // 2  # v2->1, v4->2, v6->3...
                        
                        if idx % 2 == 1:  # 奇数索引 v2,v6,v10...
                            hour = (idx - 1) // 2
                            minute = 30
                        else:  # 偶数索引 v4,v8,v12...
                            hour = idx // 2
                            minute = 0
                        
                        # 验证计算结果（添加额外的日志）
                        self.logger.debug(f"计算时间点: v{i} -> idx={idx}, 结果: {hour:02d}:{minute:02d}")
                        
                        day_offset = 0
                    
                    # 打印调试信息
                    self.logger.debug(f"键 {key}: i={i}, hour={hour}, minute={minute}, day_offset={day_offset}")
                    
                    # 创建带有正确日期和时间的时间戳
                    timestamp = base_date + timedelta(days=day_offset)
                    timestamp = timestamp.replace(hour=hour, minute=minute)
                    
                    # 添加到列表
                    timestamps.append(timestamp)
                    volume_values.append(float(item_data[key]))
            
            # 调试: 输出时间戳和值的详细信息
            if timestamps:
                debug_data = pd.DataFrame({
                    'index': range(1, len(timestamps) + 1),
                    'date_time': timestamps,
                    self.field_name: volume_values
                })
                self.logger.debug(f"解析后的数据点详情:\n{debug_data}")
            
            # 如果有数据，返回DataFrame
            if timestamps:
                df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: volume_values
                })
                self.logger.info(f"成功解析数据，共 {len(timestamps)} 条记录")
                return df
        
        # 如果没有找到实时市场出清总电量数据
        self.logger.warning("未能找到实时市场出清总电量数据")
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
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/marketQuery/findTbDisclosureDevGeneratorH5MospowerYearPageData"
        
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
                "zjNumber": "0200078",
                "measType": "98249011"
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
            response: 响应对象
        """
        try:
            # 检查cookies参数
            if cookies and isinstance(cookies, str):
                # 如果cookies是字符串，创建一个简单的字典
                cookies_dict = {'Cookie': cookies}
                cookies = cookies_dict
            
            if method.upper() == 'GET':
                response = get(url, params=params, headers=headers, cookies=cookies)
            else:
                response = post(url, data=data, json=params, headers=headers, cookies=cookies)
            
            return response
        except Exception as e:
            self.logger.error(f"发送请求失败: {e}")
            raise
    
    def parse_response(self, response, query_date=None):
        """
        解析响应
        
        Args:
            response: 响应对象
            query_date: 查询日期，格式为YYYY-MM-DD
            
        Returns:
            json_data: JSON格式的数据
        """
        try:
            # 解析JSON响应
            json_data = response.json()
            
            # 检查响应状态
            if json_data.get('status') != 0:
                self.logger.error(f"API返回错误状态: {json_data.get('status')}")
                message = json_data.get('message', 'Unknown error')
                self.logger.error(f"错误信息: {message}")
            
            return json_data
        except json.JSONDecodeError as e:
            self.logger.error(f"解析JSON响应失败: {e}")
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
        try:
            # 格式化日期
            start_date = self.format_date(start_date)
            end_date = self.format_date(end_date) if end_date else start_date
            
            self.logger.info(f"获取 {start_date} 至 {end_date} 的实时市场出清总电量数据")
            
            # 生成日期序列
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            date_range = [(start_dt + timedelta(days=i)).strftime('%Y-%m-%d') 
                          for i in range((end_dt - start_dt).days + 1)]
            
            # 用于存储所有数据的DataFrame
            all_data = pd.DataFrame()
            
            # 遍历日期序列
            for query_date in date_range:
                # 获取请求参数
                request_params = self.get_request_params(query_date)
                
                # 发送请求
                response = self.send_request(
                    url=request_params["url"],
                    method=request_params["method"],
                    headers=request_params["headers"],
                    params=request_params["params"],
                    data=request_params["data"],
                    cookies=request_params["cookies"]
                )
                
                # 解析响应
                json_data = self.parse_response(response, query_date)
                
                # 转换数据
                df = self.transform_data(json_data, query_date)
                
                # 如果获取到数据，添加到总DataFrame
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)
                else:
                    self.logger.warning(f"未能获取 {query_date} 的数据")
            
            # 检查是否有数据
            if all_data.empty:
                self.logger.warning(f"未能获取 {start_date} 至 {end_date} 的数据")
                return pd.DataFrame()
            
            # 返回所有数据
            self.logger.info(f"成功获取 {len(all_data)} 条实时市场出清总电量数据")
            return all_data
        except Exception as e:
            self.logger.error(f"获取数据失败: {e}", exc_info=True)
            return pd.DataFrame()
    
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
            # 确保要更新的列不包含主键列
            if update_columns is None:
                update_columns = [col for col in df.columns if col != 'date_time']
            
            # 设置目标表名
            table_name = self.target_table
            
            # 保存到数据库
            self.logger.info(f"保存 {len(df)} 条记录到表 {table_name}")
            
            # 调用基类的保存方法
            return super().save_to_db(df, update_columns=update_columns)
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}", exc_info=True)
            return False

async def crawl_spot_cleared_volume_for_date(date_str, target_table=None, cookie=None):
    """
    爬取指定日期的实时市场出清总电量数据
    
    Args:
        date_str: 日期字符串，格式为YYYY-MM-DD
        target_table: 目标数据表名，默认使用config.py中的配置
        cookie: API请求的Cookie，如果提供则使用此Cookie
        
    Returns:
        success: 是否成功爬取数据
    """
    try:
        # 创建爬虫实例
        crawler = SpotClearedVolumeCrawler(target_table=target_table, cookie=cookie)
        
        # 获取数据
        df = crawler.fetch_data(date_str)
        
        # 如果有数据，保存到数据库
        if not df.empty:
            update_columns = [col for col in df.columns if col != 'date_time']
            return crawler.save_to_db(df, update_columns=update_columns)
        else:
            return False
    except Exception as e:
        logger = setup_logger('crawler.spot_cleared_volume')
        logger.error(f"爬取 {date_str} 的实时市场出清总电量数据失败: {e}", exc_info=True)
        return False

async def run_historical_crawl(start_date, end_date, target_table=None, cookie=None):
    """
    运行历史数据爬取
    
    Args:
        start_date: 开始日期，格式为YYYY-MM-DD
        end_date: 结束日期，格式为YYYY-MM-DD
        target_table: 目标数据表名，默认使用config.py中的配置
        cookie: API请求的Cookie，如果提供则使用此Cookie
        
    Returns:
        success: 是否成功爬取所有数据
    """
    logger = setup_logger('crawler.spot_cleared_volume')
    logger.info(f"开始爬取 {start_date} 至 {end_date} 的实时市场出清总电量历史数据")
    
    try:
        # 创建爬虫实例
        crawler = SpotClearedVolumeCrawler(target_table=target_table, cookie=cookie)
        
        # 获取数据
        df = crawler.fetch_data(start_date, end_date)
        
        # 如果有数据，保存到数据库
        if not df.empty:
            update_columns = [col for col in df.columns if col != 'date_time']
            success = crawler.save_to_db(df, update_columns=update_columns)
            logger.info(f"历史数据爬取{'成功' if success else '失败'}")
            return success
        else:
            logger.warning("未获取到历史数据")
            return False
    except Exception as e:
        logger.error(f"爬取历史数据失败: {e}", exc_info=True)
        return False

async def run_daily_crawl(target_table=None, retry_days=3, cookie=None):
    """
    运行每日数据爬取
    
    Args:
        target_table: 目标数据表名，默认使用config.py中的配置
        retry_days: 重试天数，为确保数据完整性，会重新爬取最近几天的数据
        cookie: API请求的Cookie，如果提供则使用此Cookie
        
    Returns:
        success: 是否成功爬取所有数据
    """
    logger = setup_logger('crawler.spot_cleared_volume')
    logger.info("开始执行实时市场出清总电量每日爬取任务")
    
    try:
        # 获取当前日期
        today = datetime.now().strftime('%Y-%m-%d')
        
        # 计算重试开始日期
        retry_start_date = (datetime.now() - timedelta(days=retry_days)).strftime('%Y-%m-%d')
        
        # 创建爬虫实例
        crawler = SpotClearedVolumeCrawler(target_table=target_table, cookie=cookie)
        
        # 获取数据
        df = crawler.fetch_data(retry_start_date, today)
        
        # 如果有数据，保存到数据库
        if not df.empty:
            update_columns = [col for col in df.columns if col != 'date_time']
            success = crawler.save_to_db(df, update_columns=update_columns)
            logger.info(f"每日数据爬取{'成功' if success else '失败'}")
            return success
        else:
            logger.warning("未获取到每日数据")
            return False
    except Exception as e:
        logger.error(f"每日数据爬取失败: {e}", exc_info=True)
        return False 