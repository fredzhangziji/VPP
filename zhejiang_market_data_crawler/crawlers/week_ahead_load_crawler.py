"""
周前负荷预测爬虫
用于抓取浙江电力市场的周前负荷预测数据
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

class WeekAheadLoadCrawler(JSONCrawler):
    """周前负荷预测爬虫"""
    
    def __init__(self, target_table=None, field_name='week_ahead_load_forecast', cookie=None):
        """
        初始化周前负荷预测爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            field_name: 字段名，默认为week_ahead_load_forecast
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('week_ahead_load')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.field_name = field_name
        self.cookie = cookie or get_api_cookie()
    
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
            params: 请求参数
            data: 请求数据
            cookies: 请求cookies
        """
        # 默认获取当天的数据
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        
        # 如果未指定结束日期，则使用开始日期
        if not end_date:
            end_date = start_date
        
        # 请求URL
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/supplyAndDemand/weekLoad"
        
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
        
        # 如果提供了Cookie，则添加到请求头中
        if self.cookie:
            headers["Cookie"] = self.cookie
        
        # 请求参数
        params = None
        
        # 请求数据
        data = {
            "pageInfo": {
                "pageSize": 10,
                "pageNum": 1
            },
            "data": {
                "queryDate": start_date,
                "zjNumber": "0200004",
                "measType": None
            }
        }
        
        # 请求cookies
        cookies = None
        
        # 返回请求参数
        return {
            "url": url,
            "method": "POST",
            "headers": headers,
            "params": params,
            "data": json.dumps(data),
            "cookies": cookies
        }
    
    def transform_data(self, json_data, query_date=None):
        """
        转换JSON数据为DataFrame
        
        Args:
            json_data: JSON数据
            query_date: 查询日期，格式为YYYY-MM-DD
            
        Returns:
            df: 包含解析后数据的DataFrame
        """
        # 检查响应状态
        if not json_data or json_data.get('status') != 0:
            error_msg = json_data.get('message') if json_data else '响应为空'
            self.logger.error(f"API响应错误: {error_msg}")
            return pd.DataFrame()
        
        # 获取数据列表
        data_list = json_data.get('data', {}).get('list', [])
        
        # 如果数据列表为空，返回空DataFrame
        if not data_list:
            self.logger.warning(f"未获取到数据")
            return pd.DataFrame()
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        all_data = []
        
        # 遍历数据列表
        for data in data_list:
            # 创建时间点和负荷预测值列表
            timestamps = []
            forecast_values = []
            
            # 解析v1到v96的数据，每个数据点对应15分钟
            for i in range(1, 97):
                key = f"v{i}"
                if key in data and data[key] is not None:
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
                    try:
                        forecast_values.append(float(data[key]))
                    except (ValueError, TypeError):
                        self.logger.warning(f"无法将 {data[key]} 转换为浮点数，跳过该数据点")
                        continue
            
            # 如果有数据，添加到总数据中
            if timestamps:
                temp_df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: forecast_values
                })
                all_data.append(temp_df)
        
        # 合并所有数据
        if all_data:
            result_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"成功解析数据，共 {len(result_df)} 条记录")
            if not result_df.empty:
                self.logger.debug(f"数据范围: {result_df[self.field_name].min()} 至 {result_df[self.field_name].max()}")
            return result_df
        else:
            self.logger.warning(f"解析数据后得到空结果，可能是数据格式发生变化")
            return pd.DataFrame()
    
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
            
            # 发送请求
            if method.upper() == 'GET':
                response = get(url, params=params, headers=headers, cookies=cookies)
            else:  # POST
                response = post(url, data=data, params=params, headers=headers, cookies=cookies)
            
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
            
            # 检查是否返回的是XML格式
            if response.startswith('<') and '>' in response:
                self.logger.warning("返回的可能是XML格式，尝试使用正则表达式提取数据")
                try:
                    import re
                    # 提取v1到v96的值
                    data = {}
                    for i in range(1, 97):
                        pattern = f"<v{i}>(.*?)</v{i}>"
                        match = re.search(pattern, response)
                        if match:
                            data[f"v{i}"] = float(match.group(1))
                    
                    if data:
                        self.logger.info(f"使用正则表达式成功提取了 {len(data)} 个数据点")
                        # 生成DataFrame
                        timestamps = []
                        forecast_values = []
                        base_date = datetime.strptime(query_date, '%Y-%m-%d')
                        
                        for i in range(1, 97):
                            key = f"v{i}"
                            if key in data:
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
                                forecast_values.append(data[key])
                        
                        # 创建DataFrame
                        df = pd.DataFrame({
                            'date_time': timestamps,
                            self.field_name: forecast_values
                        })
                        return df
                except Exception as ex:
                    self.logger.error(f"尝试使用正则表达式提取数据失败: {ex}")
            
            return pd.DataFrame()
        
        # 检查响应状态
        status = json_data.get('status')
        if status != 0:
            self.logger.error(f"API响应错误: {json_data.get('message')}")
            return pd.DataFrame()
        
        # 调用transform_data方法并传入query_date参数
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
        
        # 如果总数据不为空，记录总数据量
        if not all_data.empty:
            self.logger.info(f"总共获取了 {len(all_data)} 条数据")
            
            # 打印数据样例
            self.logger.debug(f"数据样例:\n{all_data.head(3)}")
        else:
            self.logger.error(f"未能获取到任何数据。请求的日期范围: {start_date} 至 {end_date}")
        
        # 如果有缺失的日期，输出汇总信息
        if missing_dates:
            self.logger.warning(f"以下日期的数据缺失（共 {len(missing_dates)} 天）: {', '.join(missing_dates)}")
        
        # 返回总数据
        return all_data
    
    def _save_latest_data(self, df):
        """
        保存最近获取的数据到临时文件，用于在官方数据不可用时的回退策略
        
        Args:
            df: 包含数据的DataFrame
        """
        # 不再保存缓存，方法保留但不执行任何操作
        pass
    
    def _get_cached_data(self):
        """
        从临时文件获取缓存的数据
        
        Returns:
            df: 包含缓存数据的DataFrame，如果没有缓存则返回空DataFrame
        """
        # 不再使用缓存，直接返回空DataFrame
        return pd.DataFrame()
    
    def save_to_db(self, df, update_columns=None):
        """
        保存数据到数据库
        
        Args:
            df: 包含数据的DataFrame
            update_columns: 当记录已存在时要更新的列，默认为None（更新所有列）
        
        Returns:
            success: 保存是否成功
        """
        try:
            # 保存数据到数据库
            return save_to_db(df, table_name=self.target_table, update_columns=update_columns)
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            return False


async def crawl_week_ahead_load_for_date(date_str, target_table=None, field_name='week_ahead_load_forecast', cookie=None):
    """
    爬取指定日期的周前负荷预测数据
    
    Args:
        date_str: 日期字符串，格式为YYYY-MM-DD
        target_table: 目标数据表名，默认使用config.py中的配置
        field_name: 字段名，默认为week_ahead_load_forecast
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        success: 爬取是否成功
    """
    # 创建爬虫实例
    crawler = WeekAheadLoadCrawler(target_table=target_table, field_name=field_name, cookie=cookie)
    
    # 获取数据
    df = crawler.fetch_data(date_str)
    
    # 检查是否成功获取数据
    if df.empty:
        return False
    
    # 保存到数据库
    return crawler.save_to_db(df, update_columns=[field_name])


async def run_historical_crawl(target_table=None, field_name='week_ahead_load_forecast', cookie=None):
    """
    运行历史数据爬取，获取过去一年的数据
    
    Args:
        target_table: 目标数据表名，默认使用config.py中的配置
        field_name: 字段名，默认为week_ahead_load_forecast
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        success: 爬取是否成功
    """
    # 创建爬虫实例
    crawler = WeekAheadLoadCrawler(target_table=target_table, field_name=field_name, cookie=cookie)
    
    # 计算开始日期（一年前）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 格式化日期
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # 获取数据
    df = crawler.fetch_data(start_date_str, end_date_str)
    
    # 检查是否成功获取数据
    if df.empty:
        return False
    
    # 保存到数据库
    return crawler.save_to_db(df, update_columns=[field_name])


async def run_daily_crawl(target_table=None, field_name='week_ahead_load_forecast', retry_days=3, cookie=None):
    """
    运行每日数据爬取，获取当天的数据，如果失败则尝试获取前几天的数据
    
    Args:
        target_table: 目标数据表名，默认使用config.py中的配置
        field_name: 字段名，默认为week_ahead_load_forecast
        retry_days: 如果当天数据不存在，尝试获取前几天的数据
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        success: 爬取是否成功
    """
    # 从环境变量获取retry_days，如果设置了的话
    import os
    env_retry_days = os.environ.get('RETRY_DAYS')
    if env_retry_days:
        try:
            retry_days = int(env_retry_days)
        except ValueError:
            # 如果转换失败，使用默认值
            pass
    
    # 创建爬虫实例
    crawler = WeekAheadLoadCrawler(target_table=target_table, field_name=field_name, cookie=cookie)
    
    # 获取当天的日期
    today = datetime.now()
    
    # 尝试获取当天数据
    logger = setup_logger('crawler.week_ahead_load')
    logger.info(f"尝试获取当天 ({today.strftime('%Y-%m-%d')}) 的周前负荷预测数据")
    
    df = crawler.fetch_data(today.strftime('%Y-%m-%d'))
    
    # 如果当天数据为空，尝试获取前几天的数据
    if df.empty and retry_days > 0:
        logger.info(f"当天数据不存在，尝试获取前 {retry_days} 天的数据")
        
        for i in range(1, retry_days + 1):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            logger.info(f"尝试获取 {date} 的数据")
            
            df = crawler.fetch_data(date)
            
            if not df.empty:
                logger.info(f"成功获取到 {date} 的数据，使用此数据作为当前最新数据")
                break
    
    # 检查是否成功获取数据
    if df.empty:
        logger.error(f"无法获取最近 {retry_days + 1} 天的周前负荷预测数据，爬取失败")
        return False
    
    # 保存到数据库
    return crawler.save_to_db(df, update_columns=[field_name])