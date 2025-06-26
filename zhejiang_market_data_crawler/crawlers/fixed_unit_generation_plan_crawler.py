"""
固定出力机组发电计划爬虫
用于抓取浙江电力市场的固定出力机组发电计划数据
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from .json_crawler import JSONCrawler
from utils.logger import setup_logger
from utils.http_client import get, post
from utils.config import get_api_cookie
from pub_tools.db_tools import get_db_connection, release_db_connection, upsert_multiple_columns_to_db
import random
import time
import requests

class FixedUnitGenerationPlanCrawler(JSONCrawler):
    """固定出力机组发电计划爬虫"""

    def __init__(self, target_table=None, cookie=None):
        """
        初始化固定出力机组发电计划爬虫
        
        Args:
            target_table: 目标数据表名，默认使用config.py中的配置
            cookie: API请求的Cookie，如果提供则使用此Cookie
        """
        super().__init__('fixed_unit_generation_plan')
        self.logger = setup_logger(f'crawler.{self.name}')
        self.target_table = target_table
        self.cookie = cookie or get_api_cookie()
    
    def transform_data(self, json_data, query_date=None):
        """
        转换JSON数据为DataFrame
        
        Args:
            json_data: JSON格式的数据
            query_date: 查询日期，格式为YYYY-MM-DD
            
        Returns:
            df: 包含转换后数据的DataFrame
        """
        # 检查JSON数据是否有效
        if not json_data or not isinstance(json_data, dict) or 'status' not in json_data:
            self.logger.error("JSON数据无效")
            return pd.DataFrame()
        
        if json_data['status'] != 0:
            error_msg = json_data.get('message', '未知错误')
            self.logger.error(f"API返回错误: {error_msg}")
            return pd.DataFrame()
        
        if 'data' not in json_data:
            self.logger.error("JSON数据不包含data字段")
            return pd.DataFrame()
        
        # 获取数据
        data = json_data['data']
        
        if not data or 'list' not in data:
            self.logger.warning("API返回的数据为空或格式不正确")
            return pd.DataFrame()
        
        # 获取机组列表
        units_list = data['list']
        if not units_list:
            self.logger.warning("机组列表为空")
            return pd.DataFrame()
        
        # 如果没有提供查询日期，则使用当前日期
        if query_date is None:
            query_date = datetime.now().strftime('%Y-%m-%d')
        
        # 将日期字符串转换为datetime对象
        base_date = datetime.strptime(query_date, '%Y-%m-%d')
        
        # 从配置中导入DB_CONFIG
        from utils.config import DB_CONFIG
        
        # 用于存储机组数据的列表
        units_data = []
        # 用于存储机组发电计划数据的列表 
        plan_data = []
        # 用于存储总计划数据的列表
        total_plan_data = []
        
        for unit in units_list:
            unit_id = unit.get('id')
            name = unit.get('name', '')
            publish_name = unit.get('publishName', '')
            
            # 检查是否为浙江总输出数据
            is_total_output = (unit_id == '320000CA00000002' or unit_id == '1013300000' or unit_id == '0101330000' or
                              name == '浙江' or publish_name == '浙江' or name == '浙江电网' or publish_name == '浙江电网')
            
            if is_total_output:
                # 处理总输出数据
                for i in range(1, 97):
                    key = f"v{i}"
                    if key in unit and unit[key] is not None:
                        # 计算对应的时间点
                        hours = (i * 15) // 60
                        minutes = (i * 15) % 60
                        
                        if i == 96:
                            timestamp = base_date + timedelta(days=1)
                            timestamp = timestamp.replace(hour=0, minute=0)
                        else:
                            timestamp = base_date.replace(hour=hours, minute=minutes)
                        
                        total_generation_plan = float(unit[key])
                        total_plan_data.append({
                            'date_time': timestamp,
                            'total_generation_plan': total_generation_plan
                        })
                continue
            
            # 处理普通机组数据
            if not unit_id:
                continue
                
            # 使用name作为机组名称，如果name为空则使用publish_name
            unit_name = name if name and name != 'null' else publish_name
            
            # 如果unit_name仍然为空，则使用unit_id作为名称
            if not unit_name or unit_name == 'null':
                unit_name = f"机组_{unit_id}"
            
            units_data.append({
                'unit_id': unit_id,
                'unit_name': unit_name
            })
            
            # 处理机组发电计划数据
            for i in range(1, 97):
                key = f"v{i}"
                if key in unit and unit[key] is not None:
                    hours = (i * 15) // 60
                    minutes = (i * 15) % 60
                    
                    if i == 96:
                        timestamp = base_date + timedelta(days=1)
                        timestamp = timestamp.replace(hour=0, minute=0)
                    else:
                        timestamp = base_date.replace(hour=hours, minute=minutes)
                    
                    generation_plan = float(unit[key])
                    
                    plan_data.append({
                        'unit_id': unit_id,
                        'date_time': timestamp,
                        'generation_plan': generation_plan
                    })
        
        # 创建DataFrames
        units_df = pd.DataFrame(units_data) if units_data else pd.DataFrame()
        plan_df = pd.DataFrame(plan_data) if plan_data else pd.DataFrame()
        total_plan_df = pd.DataFrame(total_plan_data) if total_plan_data else pd.DataFrame()
        
        try:
            # 获取数据库连接
            engine, _ = get_db_connection(DB_CONFIG)
            
            # 1. 保存机组信息
            if not units_df.empty:
                try:
                    # 保存机组信息
                    upsert_multiple_columns_to_db(engine, units_df, 'fixed_generation_units', None)
                    self.logger.info(f"保存机组信息成功，共 {len(units_df)} 条记录")
                    
                    # 查询现有机组ID，用于过滤计划数据
                    query = "SELECT unit_id FROM fixed_generation_units"
                    existing_units_df = pd.read_sql(query, engine)
                    existing_unit_ids = set(existing_units_df['unit_id'].tolist())
                except Exception as e:
                    self.logger.error(f"保存机组信息失败: {e}")
                    return pd.DataFrame()
            else:
                # 查询现有机组ID
                try:
                    query = "SELECT unit_id FROM fixed_generation_units"
                    existing_units_df = pd.read_sql(query, engine)
                    existing_unit_ids = set(existing_units_df['unit_id'].tolist())
                except Exception as e:
                    self.logger.error(f"查询现有机组ID失败: {e}")
                    return pd.DataFrame()
            
            # 2. 过滤并保存计划数据
            success = False
            if not plan_df.empty:
                # 过滤计划数据，只保留已存在于机组表中的数据
                plan_df = plan_df[plan_df['unit_id'].isin(existing_unit_ids)]
                
                # 分批插入计划数据
                if not plan_df.empty:
                    batch_size = 200
                    total_records = len(plan_df)
                    batches = (total_records + batch_size - 1) // batch_size
                    
                    self.logger.info(f"开始分批插入 {total_records} 条机组发电计划数据，分 {batches} 批进行")
                    
                    insert_success = 0
                    for i in range(batches):
                        start_idx = i * batch_size
                        end_idx = min((i + 1) * batch_size, total_records)
                        batch_df = plan_df.iloc[start_idx:end_idx]
                        
                        try:
                            upsert_multiple_columns_to_db(engine, batch_df, 'fixed_unit_generation_plan', None)
                            insert_success += len(batch_df)
                        except Exception as e:
                            self.logger.error(f"插入第 {i+1}/{batches} 批数据失败: {e}")
                    
                    self.logger.info(f"成功保存 {insert_success} 条机组发电计划数据")
                    if insert_success > 0:
                        success = True
            
            # 3. 保存总计划数据
            if not total_plan_df.empty:
                try:
                    upsert_multiple_columns_to_db(engine, total_plan_df, 'total_fixed_generation_plan', None)
                    self.logger.info(f"成功保存 {len(total_plan_df)} 条总计划数据")
                    success = True
                except Exception as e:
                    self.logger.error(f"保存总计划数据失败: {e}")
            
            # 返回结果
            if success:
                result_df = pd.DataFrame({
                    'date': [query_date],
                    'units_count': [len(units_data)],
                    'plan_points_count': [len(plan_data)],
                    'total_plan_points_count': [len(total_plan_data)]
                })
                return result_df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"处理数据失败: {e}")
            return pd.DataFrame()
        finally:
            # 关闭数据库连接
            if 'engine' in locals() and engine:
                engine.dispose()
    
    def get_request_params(self, date=None, page_num=1):
        """
        获取请求参数
        
        Args:
            date: 查询日期，格式为YYYY-MM-DD
            page_num: 页码，默认为1
            
        Returns:
            dict: 包含请求所需的所有参数的字典
        """
        # 默认获取当天的数据
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # 请求URL
        url = "https://zjpx.com.cn/px-settlement-infpubquery-phbzj/tbDisclosureDevGeneratorH5FixpalnYear/findGDJZPage"
        # 设置API URL
        self.api_url = url
        
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
                "pageNum": page_num
            },
            "data": {
                "queryDate": date,
                "queryMeasType": "98009008",
                "zjNumber": "0200159",
                "measType": ""
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
    
    def send_request(self, url, method, headers=None, params=None, data=None, cookies=None, max_retries=3):
        """
        发送HTTP请求，并自动处理重试
        
        Args:
            url: 请求URL
            method: 请求方法，GET或POST
            headers: 请求头
            params: 请求参数
            data: 请求数据
            cookies: 请求cookies
            max_retries: 最大重试次数，默认3次
            
        Returns:
            response_text: 响应文本
        """
        retry_count = 0
        base_wait_time = 5  # 基础等待时间（秒）
        
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
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    # 指数退避策略，随机化等待时间
                    wait_time = base_wait_time * (2 ** (retry_count - 1)) + random.uniform(0, 3)
                    self.logger.info(f"第 {retry_count} 次重试，等待 {wait_time:.2f} 秒...")
                    time.sleep(wait_time)
                
                # 发送请求
                if method.upper() == 'GET':
                    response = get(url, params=params, headers=headers, cookies=cookie_dict)
                else:  # POST
                    response = post(url, data=data, params=params, headers=headers, cookies=cookie_dict)
                
                # 检查响应状态码
                if response.status_code == 200:
                    return response.text
                elif response.status_code in [500, 501, 502, 503, 504]:
                    self.logger.warning(f"服务器错误: 状态码 {response.status_code}，将进行重试")
                    retry_count += 1
                    continue
                else:
                    self.logger.error(f"请求失败: 状态码 {response.status_code}")
                    return None
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"请求异常: {e}, 重试中...")
                retry_count += 1
                continue
            except Exception as e:
                self.logger.error(f"未预期的异常: {e}")
                return None
        
        self.logger.error(f"重试 {max_retries} 次后仍然失败")
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
                self.logger.error(f"JSON解析错误: {e}")
                return None
        
        # 如果响应已经是JSON对象
        elif hasattr(response, 'json'):
            try:
                return response.json()
            except json.JSONDecodeError as e:
                self.logger.error(f"响应JSON解析错误: {e}")
                return None
        
        # 其他情况
        else:
            self.logger.error(f"无法解析响应类型: {type(response)}")
            return None
    
    def format_date(self, date_str):
        """
        格式化日期字符串
        
        Args:
            date_str: 日期字符串，格式为YYYY-MM-DD
        
        Returns:
            date_obj: 格式化后的日期对象
        """
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            self.logger.error(f"无效的日期格式: {date_str}")
            return None
    
    def fetch_data(self, start_date=None, end_date=None):
        """
        获取固定出力机组发电计划数据
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
        
        Returns:
            df: 包含数据的DataFrame
        """
        # 格式化日期
        if start_date:
            start_date = self.format_date(start_date)
        else:
            start_date = datetime.now()
        
        if end_date:
            end_date = self.format_date(end_date)
        else:
            end_date = start_date
        
        # 确保开始日期不晚于结束日期
        if start_date > end_date:
            self.logger.error(f"开始日期 {start_date} 晚于结束日期 {end_date}")
            return pd.DataFrame()
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame()
        
        # 遍历日期范围
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            self.logger.info(f"获取 {date_str} 的固定出力机组发电计划数据")
            
            try:
                # 获取请求参数
                request_params = self.get_request_params(date_str)
                url = request_params['url']
                method = request_params['method']
                headers = request_params['headers']
                params = request_params['params']
                data = request_params['data']
                cookies = request_params['cookies']
                
                # 第一次请求，获取总记录数和页数
                response_text = self.send_request(url, method, headers, params, data, cookies)
                if not response_text:
                    self.logger.error(f"获取 {date_str} 的数据失败")
                    current_date += timedelta(days=1)
                    continue
                
                json_data = self.parse_response(response_text, date_str)
                if not json_data:
                    self.logger.error(f"解析 {date_str} 的响应失败")
                    current_date += timedelta(days=1)
                    continue

                # 获取总记录数和页数
                data = json_data.get('data', {})
                total = data.get('total', 0)
                pageSize = data.get('pageSize', 10) or 10
                pages = (total + pageSize - 1) // pageSize if total > 0 else 0
                
                self.logger.info(f"找到 {total} 条记录，共 {pages} 页")
                
                # 所有机组数据的列表
                all_units = []
                
                # 第一页的数据已经获取到了
                units_list = data.get('list', [])
                all_units.extend(units_list)
                
                # 如果有多页，继续获取后面的页
                for page_num in range(2, pages + 1):
                    page_params = self.get_request_params(date_str, page_num)
                    page_response = self.send_request(url, method, headers, params, page_params['data'], cookies)
                    
                    if not page_response:
                        self.logger.error(f"获取第 {page_num} 页数据失败")
                        continue
                    
                    page_json = self.parse_response(page_response, date_str)
                    if not page_json:
                        self.logger.error(f"解析第 {page_num} 页响应失败")
                        continue
                    
                    page_data = page_json.get('data', {})
                    page_units = page_data.get('list', [])
                    all_units.extend(page_units)
                
                # 创建包含所有机组数据的JSON
                complete_json = json_data.copy()
                complete_json['data']['list'] = all_units
                
                # 转换数据
                df = self.transform_data(complete_json, date_str)
                
                # 合并结果
                if not df.empty:
                    if result_df.empty:
                        result_df = df
                    else:
                        result_df = pd.concat([result_df, df], ignore_index=True)
                
            except Exception as e:
                self.logger.error(f"处理 {date_str} 的数据时出错: {e}")
            
            # 移至下一天
            current_date += timedelta(days=1)
        
        return result_df
    
    def run(self, start_date=None, end_date=None, update_columns=None):
        """
        运行爬虫
        
        Args:
            start_date: 开始日期，格式为YYYY-MM-DD
            end_date: 结束日期，格式为YYYY-MM-DD
            update_columns: 当记录已存在时要更新的列，默认为None
        
        Returns:
            success: 是否运行成功
        """
        try:
            self.logger.info(f"开始运行爬虫: {self.name}")
            self.logger.info(f"时间范围: {start_date or '今日'} 至 {end_date or '今日'}")
            
            # 获取数据
            df = self.fetch_data(start_date, end_date)
            
            if df is not None and not df.empty:
                self.logger.info(f"爬虫 {self.name} 运行完成，共获取 {len(df)} 条数据")
                return True
            else:
                self.logger.warning(f"爬虫 {self.name} 未获取到数据")
                return False
        except Exception as e:
            self.logger.error(f"爬虫 {self.name} 运行失败: {e}", exc_info=True)
            return False