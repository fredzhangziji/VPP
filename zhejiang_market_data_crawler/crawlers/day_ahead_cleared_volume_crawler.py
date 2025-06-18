"""
日前市场出清总电量爬虫
用于抓取浙江电力市场的日前市场出清总电量数据
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
import logging

class DayAheadClearedVolumeCrawler(JSONCrawler):
    """日前市场出清总电量爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化日前市场出清总电量爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('day_ahead_cleared_volume')
        self.logger = setup_logger(f'crawler.{self.name}', level=logging.DEBUG)
        self.target_table = target_table or TARGET_TABLE
        
        # 处理cookie参数
        if cookie:
            if isinstance(cookie, str):
                self.logger.debug("Cookie是字符串格式，将在请求时直接作为header使用")
                self.cookie_str = cookie
                self.cookie_dict = None
            elif isinstance(cookie, dict):
                self.logger.debug("Cookie是字典格式")
                self.cookie_str = None
                self.cookie_dict = cookie
            else:
                self.logger.warning(f"Cookie格式不支持: {type(cookie)}，将不使用cookie")
                self.cookie_str = None
                self.cookie_dict = None
        else:
            cookie_from_config = get_api_cookie()
            if isinstance(cookie_from_config, str):
                self.logger.debug("从配置获取的Cookie是字符串格式")
                self.cookie_str = cookie_from_config
                self.cookie_dict = None
            elif isinstance(cookie_from_config, dict):
                self.logger.debug("从配置获取的Cookie是字典格式")
                self.cookie_str = None
                self.cookie_dict = cookie_from_config
            else:
                self.logger.warning(f"从配置获取的Cookie格式不支持: {type(cookie_from_config)}，将不使用cookie")
                self.cookie_str = None
                self.cookie_dict = None
        
        # 数据库字段名
        self.field_name = "day_ahead_cleared_volume"  # 日前市场出清总电量(MWh)
    
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
        
        # 记录JSON数据的类型和部分内容
        self.logger.debug(f"响应数据类型: {type(json_data)}")
        if isinstance(json_data, dict):
            self.logger.debug(f"响应键: {list(json_data.keys())}")
        
        # 检查JSON数据是否包含status字段
        if 'status' not in json_data:
            self.logger.error("JSON数据不包含status字段")
            self.logger.debug(f"JSON数据内容(前100字符): {str(json_data)[:100]}...")
            return pd.DataFrame()
        
        # 检查状态是否正常
        if json_data.get('status') != 0:
            self.logger.error(f"API返回错误状态: {json_data.get('status')}")
            message = json_data.get('message', 'Unknown error')
            self.logger.error(f"错误信息: {message}")
            return pd.DataFrame()
        
        # 获取数据列表
        try:
            data = json_data.get('data', {})
            if not isinstance(data, dict):
                self.logger.error(f"data不是字典类型: {type(data)}")
                self.logger.debug(f"data内容: {data}")
                return pd.DataFrame()
                
            data_list = data.get('list', [])
            
            if not data_list:
                self.logger.warning("没有找到数据")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取数据列表时出错: {e}")
            return pd.DataFrame()
        
        self.logger.info(f"找到 {len(data_list)} 条日前市场出清总电量记录")
        
        # 使用查询日期作为基准日期
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            # 如果没有提供查询日期，则使用当前日期
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 创建次日凌晨时间点（用于v96的处理）
        next_day = base_date + timedelta(days=1)
        
        # 遍历数据
        for item_data in data_list:
            # 记录每个数据项的可用键，便于调试
            if len(item_data.keys()) > 10:  # 如果键很多，只记录一部分
                self.logger.debug(f"数据项部分键: {list(item_data.keys())[:10]}...")
            else:
                self.logger.debug(f"数据项键: {list(item_data.keys())}")
            
            # 创建时间点和出清总电量值列表
            timestamps = []
            volume_values = []
            
            try:
                # 新的解析逻辑：处理v2, v4, v6, ..., v96的数据
                # 从实际响应看，这些数据对应每小时的半点和整点数据
                for i in range(2, 97, 2):
                    key = f"v{i}"
                    if key in item_data and item_data[key] is not None:
                        # 记录找到的数据，便于调试
                        self.logger.debug(f"找到数据 {key}={item_data[key]}")
                        
                        # 特殊处理v96（对应24:00，应转换为次日00:00）
                        if i == 96:
                            self.logger.debug("处理v96数据（24:00），转换为次日00:00")
                            timestamp = next_day.replace(hour=0, minute=0)
                        else:
                            # 计算时间点：v2对应00:30，v4对应01:00，v6对应01:30...
                            hour = (i // 2) // 2  # 小时部分 (0-23)
                            minute = 30 if i % 4 == 2 else 0  # 分钟部分，v2,v6,v10...对应30分，v4,v8,v12...对应00分
                            
                            # 记录计算的时间点，便于调试
                            self.logger.debug(f"计算时间点: {key} -> {hour:02d}:{minute:02d}")
                            
                            # 确保小时和分钟值在有效范围内
                            if 0 <= hour <= 23 and minute in (0, 30):
                                timestamp = base_date.replace(hour=hour, minute=minute)
                            else:
                                self.logger.warning(f"计算出的时间点无效: {hour}:{minute}，跳过数据 {key}={item_data[key]}")
                                continue
                        
                        # 添加到列表
                        timestamps.append(timestamp)
                        volume_values.append(float(item_data[key]))
                            
                # 如果没有找到v2-v96格式的数据，尝试解析v0005-v2355格式的数据
                if not timestamps:
                    self.logger.info("未找到v2-v96格式的数据，尝试解析v0005-v2355格式的数据")
                    for hour in range(0, 24):
                        for minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                            key = f"v{hour:02d}{minute:02d}"
                            if key in item_data and item_data[key] is not None:
                                # 记录找到的数据，便于调试
                                self.logger.debug(f"找到数据 {key}={item_data[key]}")
                                
                                # 创建时间点
                                timestamp = base_date.replace(hour=hour, minute=minute)
                                
                                # 添加到列表
                                timestamps.append(timestamp)
                                volume_values.append(float(item_data[key]))
                                
                    # 检查v2400（对应24:00）
                    if "v2400" in item_data and item_data["v2400"] is not None:
                        self.logger.debug(f"找到数据 v2400={item_data['v2400']}")
                        timestamp = next_day.replace(hour=0, minute=0)
                        timestamps.append(timestamp)
                        volume_values.append(float(item_data["v2400"]))
            except Exception as e:
                self.logger.error(f"处理数据时出错: {str(e)}")
                import traceback
                self.logger.debug(f"详细错误信息: {traceback.format_exc()}")
                continue
            
            # 如果有数据，返回DataFrame
            if timestamps:
                df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: volume_values
                })
                self.logger.info(f"成功解析日前市场出清总电量数据，共 {len(timestamps)} 条记录")
                return df
        
        # 如果没有找到数据
        self.logger.warning("未能找到日前市场出清总电量数据")
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
        
        # 如果有cookie字符串，添加到headers中
        if hasattr(self, 'cookie_str') and self.cookie_str:
            headers['Cookie'] = self.cookie_str
        
        # 请求参数
        payload = {
            "pageInfo": {
                "pageSize": 10,
                "pageNum": 1
            },
            "data": {
                "queryDate": start_date,
                "zjNumber": "0200067",
                "measType": "98149011"
            }
        }
        
        # 返回请求参数字典
        return {
            "url": url,
            "method": "POST",
            "headers": headers,
            "params": None,
            "data": json.dumps(payload),
            "cookies": self.cookie_dict if hasattr(self, 'cookie_dict') else None
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
        self.logger.info(f"发送{method}请求: {url}")
        
        # 打印请求数据，但隐藏敏感信息
        if data:
            data_obj = json.loads(data)
            self.logger.info(f"请求数据: {data_obj}")
        
        # 检查cookie参数类型
        if cookies:
            if not isinstance(cookies, dict):
                self.logger.warning(f"传入的cookies不是字典类型: {type(cookies)}，将忽略cookies参数")
                cookies = None
        
        try:
            if method.upper() == 'GET':
                response = get(url, headers=headers, params=params, cookies=cookies)
            elif method.upper() == 'POST':
                # 在这里直接传递cookies参数
                response = post(url, headers=headers, data=data, cookies=cookies)
            else:
                self.logger.error(f"不支持的请求方法: {method}")
                return None
            
            # 检查响应状态码
            if response.status_code != 200:
                self.logger.error(f"请求失败，状态码: {response.status_code}")
                self.logger.error(f"响应内容: {response.text}")
                return None
            
            # 打印响应内容前100个字符，用于调试
            self.logger.debug(f"响应内容(前100字符): {response.text[:100]}...")
            
            # 检查响应类型
            content_type = response.headers.get('Content-Type', '')
            self.logger.debug(f"响应Content-Type: {content_type}")
            
            # 返回响应对象
            return response
        except Exception as e:
            self.logger.error(f"请求异常: {str(e)}")
            # 打印详细异常信息
            import traceback
            self.logger.debug(f"详细异常信息: {traceback.format_exc()}")
            return None
    
    def parse_response(self, response, query_date=None):
        """
        解析响应
        
        Args:
            response: 响应对象
            query_date: 查询日期，格式为YYYY-MM-DD
        
        Returns:
            data: 解析后的数据
        """
        if not response:
            self.logger.error("响应对象为空")
            return None
        
        try:
            # 打印响应文本
            self.logger.debug(f"响应文本(前100字符): {response.text[:100]}...")
            
            # 解析JSON响应
            try:
                json_data = response.json()
                self.logger.debug(f"成功解析响应为JSON，类型: {type(json_data)}")
                if isinstance(json_data, str):
                    self.logger.debug(f"响应是JSON字符串: {json_data[:100]}...")
                    try:
                        # 尝试再次解析JSON字符串
                        json_data = json.loads(json_data)
                        self.logger.info("成功将嵌套的JSON字符串解析为对象")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"无法解析嵌套的JSON字符串: {e}")
            except Exception as e:
                self.logger.error(f"解析响应为JSON失败: {e}")
                # 打印详细异常信息
                import traceback
                self.logger.debug(f"详细异常信息: {traceback.format_exc()}")
                self.logger.debug(f"响应内容(前200字符): {response.text[:200]}...")
                return None
            
            # 转换为DataFrame
            df = self.transform_data(json_data, query_date)
            
            return df
        except Exception as e:
            self.logger.error(f"解析响应失败: {e}")
            # 打印详细异常信息
            import traceback
            self.logger.debug(f"详细异常信息: {traceback.format_exc()}")
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
        # 如果未提供start_date，则使用当前日期
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        else:
            start_date = self.format_date(start_date)
        
        # 如果未提供end_date，则使用start_date
        if not end_date:
            end_date = start_date
        else:
            end_date = self.format_date(end_date)
        
        if not start_date or not end_date:
            self.logger.error("日期格式错误")
            return pd.DataFrame()
        
        self.logger.info(f"开始获取日前市场出清总电量数据，时间范围: {start_date} 至 {end_date}")
        
        # 计算日期范围
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 确保end_datetime不早于start_datetime
        if end_datetime < start_datetime:
            self.logger.warning(f"结束日期 {end_date} 早于开始日期 {start_date}，将交换这两个日期")
            start_datetime, end_datetime = end_datetime, start_datetime
            start_date, end_date = end_date, start_date
        
        date_range = []
        current_datetime = start_datetime
        while current_datetime <= end_datetime:
            date_range.append(current_datetime.strftime('%Y-%m-%d'))
            current_datetime += timedelta(days=1)
        
        self.logger.info(f"将获取以下日期的数据: {date_range}")
        
        all_data = []
        
        # 遍历日期范围
        for date_str in date_range:
            self.logger.info(f"获取 {date_str} 的数据")
            
            # 获取请求参数
            params = self.get_request_params(date_str)
            
            # 发送请求
            response = self.send_request(
                params['url'], 
                params['method'], 
                headers=params['headers'], 
                params=params['params'], 
                data=params['data'], 
                cookies=params['cookies']
            )
            
            # 解析响应
            df = self.parse_response(response, date_str)
            
            if df is not None and not df.empty:
                all_data.append(df)
                self.logger.info(f"成功获取 {date_str} 的数据，共 {len(df)} 条记录")
            else:
                self.logger.warning(f"获取 {date_str} 的数据失败")
        
        # 合并所有数据
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"总共获取 {len(combined_df)} 条记录")
            return combined_df
        else:
            self.logger.warning("未获取到任何数据")
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
        
        self.logger.info(f"将 {len(df)} 条记录保存到表 {self.target_table}")
        
        try:
            # 确保数据类型正确
            df[self.field_name] = df[self.field_name].astype(float)
            
            # 保存到数据库
            success = save_to_db(df, self.target_table, update_columns=update_columns)
            
            if success:
                self.logger.info("数据保存成功")
            else:
                self.logger.error("数据保存失败")
            
            return success
        except Exception as e:
            self.logger.error(f"保存数据时发生异常: {e}")
            return False


# 异步函数，用于爬取单一日期的数据
async def crawl_day_ahead_cleared_volume_for_date(date_str, target_table=None, cookie=None):
    """
    异步爬取指定日期的日前市场出清总电量数据
    
    Args:
        date_str: 日期字符串，格式为YYYY-MM-DD
        target_table: 目标数据表名，默认使用config.py中的配置
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        df: 包含数据的DataFrame
    """
    logger = setup_logger('crawler.day_ahead_cleared_volume.async')
    logger.info(f"异步爬取 {date_str} 的日前市场出清总电量数据")
    
    crawler = DayAheadClearedVolumeCrawler(target_table=target_table, cookie=cookie)
    df = crawler.fetch_data(date_str)
    
    if df is not None and not df.empty:
        logger.info(f"成功爬取 {date_str} 的数据，共 {len(df)} 条记录")
        
        # 保存到数据库
        update_columns = [col for col in df.columns if col != 'date_time']
        crawler.save_to_db(df, update_columns=update_columns)
        
        return df
    else:
        logger.warning(f"爬取 {date_str} 的数据失败")
        return pd.DataFrame()

# 异步函数，用于爬取历史数据
async def run_historical_crawl(start_date, end_date, target_table=None, cookie=None):
    """
    异步爬取指定日期范围的历史数据
    
    Args:
        start_date: 开始日期，格式为YYYY-MM-DD
        end_date: 结束日期，格式为YYYY-MM-DD
        target_table: 目标数据表名，默认使用config.py中的配置
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        success: 是否成功爬取所有数据
    """
    logger = setup_logger('crawler.day_ahead_cleared_volume.historical')
    logger.info(f"异步爬取历史数据，时间范围: {start_date} 至 {end_date}")
    
    # 计算日期范围
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    date_range = []
    
    current_datetime = start_datetime
    while current_datetime <= end_datetime:
        date_range.append(current_datetime.strftime('%Y-%m-%d'))
        current_datetime += timedelta(days=1)
    
    logger.info(f"将爬取以下日期的数据: {date_range}")
    
    success_count = 0
    
    # 遍历日期范围
    for date_str in date_range:
        df = await crawl_day_ahead_cleared_volume_for_date(date_str, target_table, cookie)
        
        if not df.empty:
            success_count += 1
        
        # 避免请求过于频繁
        await asyncio.sleep(1)
    
    success_rate = success_count / len(date_range) if date_range else 0
    logger.info(f"历史数据爬取完成，成功率: {success_rate:.2%} ({success_count}/{len(date_range)})")
    
    return success_rate == 1.0

# 异步函数，用于每日定时爬取数据
async def run_daily_crawl(target_table=None, retry_days=3, cookie=None):
    """
    异步爬取当天和过去几天的数据
    
    Args:
        target_table: 目标数据表名，默认使用config.py中的配置
        retry_days: 重试天数，默认为3天
        cookie: API请求的Cookie，如果提供则使用此Cookie
    
    Returns:
        success: 是否成功爬取所有数据
    """
    logger = setup_logger('crawler.day_ahead_cleared_volume.daily')
    logger.info(f"开始每日定时爬取，将尝试获取当天和过去 {retry_days} 天的数据")
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=retry_days)
    
    # 格式化日期
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # 调用历史爬取函数
    return await run_historical_crawl(start_date_str, end_date_str, target_table, cookie)