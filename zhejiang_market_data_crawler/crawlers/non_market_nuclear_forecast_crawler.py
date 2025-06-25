"""
非市场核电出力预测爬虫
用于抓取浙江电力市场的非市场核电出力预测数据
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import TARGET_TABLE, get_api_cookie

class NonMarketNuclearForecastCrawler(JSONCrawler):
    """
    非市场核电出力预测爬虫
    """
    def __init__(self, target_table=None, cookie=None):
        super().__init__('non_market_nuclear_forecast')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table or TARGET_TABLE
        self.cookie = cookie or get_api_cookie()
        self.field_name = "non_market_nuclear_forecast"  # 非市场核电出力预测(日)(MW)

    def transform_data(self, json_data, query_date=None):
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
        self.logger.info(f"找到 {len(data_list)} 条非市场核电出力预测记录")
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
                self.logger.info(f"成功解析非市场核电出力预测数据，共 {len(timestamps)} 条记录")
                return df
        self.logger.warning("未能找到非市场核电出力预测数据")
        return pd.DataFrame()

    def get_request_params(self, start_date=None, end_date=None):
        if not start_date:
            start_date = datetime.now().strftime('%Y-%m-%d')
        if not end_date:
            end_date = start_date
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/tbDisclosureDevGeneratorH5FixpalnYear/findPageData"
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
                "queryMeasType": "98880001",
                "zjNumber": "0200010",
                "measType": "98149025"
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
            self.logger.info(f"获取非市场核电出力预测数据: {date_str}")
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
                    self.logger.info(f"成功获取 {date_str} 的非市场核电出力预测数据，共 {len(df)} 条记录")
                else:
                    self.logger.warning(f"未找到 {date_str} 的非市场核电出力预测数据")
            except Exception as e:
                self.logger.error(f"获取 {date_str} 的非市场核电出力预测数据失败: {e}", exc_info=True)
            current_date += timedelta(days=1)
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            self.logger.info(f"共获取 {len(result)} 条非市场核电出力预测数据记录")
            return result
        else:
            self.logger.warning("未获取到任何非市场核电出力预测数据")
            return pd.DataFrame()