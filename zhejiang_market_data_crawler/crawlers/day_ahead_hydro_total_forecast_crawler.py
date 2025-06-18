"""
水电总出力预测爬虫
用于抓取浙江电力市场的水电总出力预测数据
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

class DayAheadHydroTotalForecastCrawler(JSONCrawler):
    """水电总出力预测爬虫"""
    
    def __init__(self, target_table=None, cookie=None):
        """
        初始化水电总出力预测爬虫
        
        Args:
            target_table: 目标数据表名
            cookie: API请求的Cookie
        """
        super().__init__('day_ahead_hydro_total_forecast')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        self.field_name = "day_ahead_hydro_total_forecast"  # 水电总出力预测(MW)

    def transform_data(self, json_data, query_date=None):
        """
        转换JSON数据为DataFrame
        
        Args:
            json_data: JSON格式的数据
            query_date: 查询日期
            
        Returns:
            DataFrame: 包含转换后数据的DataFrame
        """
        if not json_data:
            self.logger.error("JSON数据为空")
            return pd.DataFrame()
            
        if not isinstance(json_data, dict):
            self.logger.error(f"JSON数据不是字典类型: {type(json_data)}")
            return pd.DataFrame()
            
        if 'status' not in json_data:
            self.logger.error("JSON数据不包含status字段")
            return pd.DataFrame()
            
        if json_data.get('status') != 0:
            self.logger.error(f"API返回错误状态: {json_data.get('status')}")
            message = json_data.get('message', 'Unknown error')
            self.logger.error(f"错误信息: {message}")
            return pd.DataFrame()
            
        data = json_data.get('data', {})
        data_list = data.get('list', [])
        
        if not data_list:
            self.logger.warning("没有找到数据")
            return pd.DataFrame()
            
        self.logger.info(f"找到 {len(data_list)} 条水电总出力预测记录")
        
        if query_date:
            base_date = datetime.strptime(query_date, '%Y-%m-%d')
        else:
            base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for item_data in data_list:
            timestamps = []
            forecast_values = []
            
            for i in range(1, 97):
                key = f"v{i}"
                if key in item_data and item_data[key] is not None:
                    hour = (i * 15) // 60
                    minute = (i * 15) % 60
                    
                    if i == 96:
                        timestamp = base_date + timedelta(days=1)
                        timestamp = timestamp.replace(hour=0, minute=0)
                    else:
                        timestamp = base_date.replace(hour=hour, minute=minute)
                        
                    timestamps.append(timestamp)
                    forecast_values.append(float(item_data[key]))
                    
            if timestamps:
                df = pd.DataFrame({
                    'date_time': timestamps,
                    self.field_name: forecast_values
                })
                self.logger.info(f"成功解析水电总出力预测数据，共 {len(timestamps)} 条记录")
                return df
                
        self.logger.warning("未能找到水电总出力预测数据")
        return pd.DataFrame()

    def get_request_params(self, start_date=None, end_date=None):
        """
        获取请求参数
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            dict: 请求参数
        """
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
            
        if not end_date:
            end_date = start_date
            
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/operscheduleController/getTbDisclosureConPwrgridFScheduleYearList"
        
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
        
        payload = {
            "pageInfo": {
                "pageSize": 10,
                "pageNum": 1
            },
            "data": {
                "queryDate": start_date,
                "zjNumber": "0200025",
                "measType": "01012212"
            }
        }
        
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
            method: 请求方法
            headers: 请求头
            params: 请求参数
            data: 请求数据
            cookies: 请求cookies
            
        Returns:
            response: HTTP响应
        """
        try:
            cookies_dict = None
            if cookies:
                if isinstance(cookies, str):
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
        解析HTTP响应
        
        Args:
            response: HTTP响应
            query_date: 查询日期
            
        Returns:
            dict: 解析后的JSON数据
        """
        try:
            json_data = response.json()
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
        获取指定时间范围内的数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 包含获取数据的DataFrame
        """
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
            
        if not end_date:
            end_date = start_date
            
        all_data = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            self.logger.info(f"获取水电总出力预测数据: {date_str}")
            request_params = self.get_request_params(date_str, date_str)
            
            try:
                response = self.send_request(
                    request_params["url"],
                    request_params["method"],
                    headers=request_params["headers"],
                    params=request_params["params"],
                    data=request_params["data"],
                    cookies=request_params["cookies"]
                )
                
                json_data = self.parse_response(response, query_date=date_str)
                df = self.transform_data(json_data, query_date=date_str)
                
                if not df.empty:
                    all_data.append(df)
                    self.logger.info(f"成功获取 {date_str} 的水电总出力预测数据，共 {len(df)} 条记录")
                else:
                    self.logger.warning(f"未找到 {date_str} 的水电总出力预测数据")
                    
            except Exception as e:
                self.logger.error(f"获取 {date_str} 的水电总出力预测数据失败: {e}", exc_info=True)
                
            current_date += timedelta(days=1)
            
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"共获取 {len(result)} 条水电总出力预测数据记录")
            return result
        else:
            self.logger.warning("未获取到任何水电总出力预测数据")
            return pd.DataFrame()

    def save_to_db(self, df, update_columns=None):
        """
        保存数据到数据库
        
        Args:
            df: 包含数据的DataFrame
            update_columns: 需要更新的列
            
        Returns:
            bool: 是否保存成功
        """
        if df.empty:
            self.logger.warning("没有数据需要保存")
            return False
            
        try:
            return save_to_db(df, self.target_table, update_columns=update_columns)
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
            return False

async def crawl_day_ahead_hydro_total_forecast_for_date(date_str, target_table=None, cookie=None):
    """
    获取指定日期的水电总出力预测数据
    
    Args:
        date_str: 日期字符串
        target_table: 目标表名
        cookie: API请求Cookie
    
    Returns:
        DataFrame: 包含数据的DataFrame
    """
    crawler = DayAheadHydroTotalForecastCrawler(target_table=target_table, cookie=cookie)
    return crawler.fetch_data(start_date=date_str, end_date=date_str)

async def run_historical_crawl(start_date, end_date, target_table=None, cookie=None):
    """
    批量爬取历史数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        target_table: 目标表名
        cookie: API请求Cookie
    
    Returns:
        bool: 是否成功
    """
    crawler = DayAheadHydroTotalForecastCrawler(target_table=target_table, cookie=cookie)
    df = crawler.fetch_data(start_date=start_date, end_date=end_date)
    
    if not df.empty:
        update_columns = [col for col in df.columns if col != 'date_time']
        return crawler.save_to_db(df, update_columns=update_columns)
        
    return False

async def run_daily_crawl(target_table=None, retry_days=3, cookie=None):
    """
    运行每日爬虫
    
    Args:
        target_table: 目标表名
        retry_days: 重试天数
        cookie: API请求Cookie
        
    Returns:
        bool: 是否成功
    """
    today = datetime.now().strftime('%Y-%m-%d')
    crawler = DayAheadHydroTotalForecastCrawler(target_table=target_table, cookie=cookie)
    crawler.logger.info(f"开始爬取当天({today})的数据")
    today_df = crawler.fetch_data(start_date=today, end_date=today)
    
    success = False
    if not today_df.empty:
        update_columns = [col for col in today_df.columns if col != 'date_time']
        success = crawler.save_to_db(today_df, update_columns=update_columns)
    
    if retry_days > 0:
        crawler.logger.info(f"尝试补充之前 {retry_days} 天的数据")
        
        for i in range(1, retry_days + 1):
            retry_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            crawler.logger.info(f"补充 {retry_date} 的数据")
            retry_df = crawler.fetch_data(start_date=retry_date, end_date=retry_date)
            
            if not retry_df.empty:
                update_columns = [col for col in retry_df.columns if col != 'date_time']
                retry_success = crawler.save_to_db(retry_df, update_columns=update_columns)
                success = success or retry_success
                
    return success 