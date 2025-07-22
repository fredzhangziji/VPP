"""
此模块包含了所有数据爬取过程中所需的功能函数。
"""

import requests
from pub_tools import const, db_tools
from sqlalchemy import select, Table
from sqlalchemy.exc import NoSuchTableError
import pandas as pd
from snowflake import SnowflakeGenerator
import time
import ssl
from requests.adapters import HTTPAdapter
from datetime import date

import logging
import pub_tools.logging_config
logger = logging.getLogger('renewable_data_crawler')

_MAX_RETRY_NUM = 5

def _set_headers(headers: dict) -> dict:
    """为request设置对应的headers

    通过从数据库获取实时Token，拼接并构造headers的'Authorization'和'Cookie'字段。

    入参:
        headers: 字典形式，包括了浏览器的各种信息

    返回:
        设置好的headers
    """
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_USER)
    try:
        vpp_dict_info = Table('vpp_dict_info', metadata, autoload_with=engine)
    except NoSuchTableError:
        logger.error("错误: 数据库中未找到表 vpp_dict_info，请检查数据库配置或表名是否正确。")
        raise
    query = select(vpp_dict_info).where(
        vpp_dict_info.c.info_key == 'NEIMENG_VPP_TOKEN'
    )
    df_token = db_tools.read_from_db(engine, query)
    token_str = str(df_token['info_value'].iloc[0])

    headers['Authorization'] = token_str
    headers['Cookie'] = headers['Cookie'][:6] + token_str + headers['Cookie'][6:]

    db_tools.release_db_connection(engine)
    return headers

def _set_payload(date: str, type: str, city: str) -> dict:
    """为request设置对应的payload

    根据传入的入参拼接payload，告诉服务器具体需要哪个城市、哪一天、什么类型的数据。

    入参:
        date: 字符串形式，表示具体日期。例如: "2025-04-02"。
        type: 字符串形式，表示具体数据类型. 例如: "1".
        city: 字符串形式，表示具体城市. 例如: "'鄂尔多斯','薛家湾'".   

    返回:
        设置好的payload。例如:
        payload = {
            "time": "2025-04-02",
            "name": "1",
            "area": "'鄂尔多斯','薛家湾'"
            }
    """
    payload = {
        'time': date,
        'name': type,
        'area': city
    }
    
    return payload

def _fix_datetime(dt_str):
    """对于24:00的时间戳, 转换为数据库合法的时间戳."""
    date_part, time_part = dt_str.split(" ")
    if time_part == "24:00":
        new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
        return new_date.strftime("%Y-%m-%d") + " 00:00:00"
    else:
        return pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
    
def _preprocess_df(response_data: list, city_name: str, date: str) -> pd.DataFrame:
    """对于从网页上爬取到的Dataframe进行预处理

    合并同城市同时段的数据类型数据为type列; 对数据进行检查并进行去重; 验证数据长度是否合法; 生成唯一id.

    入参:
        response_data: 列表类型, 从服务器爬取到的数据. 
        city_name: 字符串类型, 具体的城市名
        date: 字符串类型, 具体的日期

    返回:
        如果数据有问题, 返回None;
        没问题返回名为df_long的pd.Dataframe.
    """
    df = pd.DataFrame(response_data)
    df['city_name'] = city_name
    df_long = pd.melt(df, 
                      id_vars=['TIME', 'city_name'],
                      value_vars=['YCVALUEPLUS1', 'YCVALUEPLUS2', 'YCVALUEPLUS3'],
                      var_name='type',
                      value_name='value')
    df_long['type'] = df_long['type'].str.extract(r'(\d)').astype(int)
    df_long.rename(columns={'TIME': 'date_time'}, inplace=True)

    df_long['date_time'] = (date + " " + df_long['date_time']).apply(_fix_datetime)

    duplicates = df_long[df_long.duplicated(subset=['city_name', 'date_time', 'type'], keep=False)]
    if not duplicates.empty:
        logger.info("发现重复数据：")
        logger.info(duplicates)
        # 去重，保留第一条记录
        df_long = df_long.drop_duplicates(subset=['city_name', 'date_time', 'type'])
        logger.info("去重完毕。")

    expected_count = 24 * 4 * 3
    if len(df_long) == expected_count:
        logger.info(f"当前数据行数符合预期：{expected_count}")
    else:
        logger.error(f"当前数据行数为：{len(df_long)}，不符合预期，预期应为：{expected_count}")
        return None
    
    gen = SnowflakeGenerator(21)
    df_long['id'] = [next(gen) for _ in range(len(df_long))]

    return df_long

def _process_city_name(city: str) -> str:
    """对于特殊的复合城市名的字符串进行处理"""
    city = city.strip("'")
    if "','" in city:
        city = city.replace("','", "+")
    return city

def validate_date(date_str: str) -> str:
    """验证日期字符串并返回标准化的日期格式
    
    入参:
        date_str: 字符串类型，表示需要验证的日期
        
    返回:
        标准化的YYYY-MM-DD格式日期字符串
        
    异常:
        如果日期格式无效，抛出ValueError
    """
    try:
        # 尝试解析日期字符串，并转换为标准格式
        return pd.to_datetime(date_str).strftime("%Y-%m-%d")
    except Exception as e:
        logger.error(f"日期格式无效: {date_str}")
        raise ValueError(f"日期格式无效: {date_str}") from e

def fetch_multi_day_wind_power_data_for_each_city(url: str, start_date: str, end_date: str = None):
    """对各个城市进行多天风电数据爬取，并将每一天的数据插入数据库。

    从起始日期到结束日期（日期递减），对每个城市依次请求单日风电数据。
    数据请求失败时会进行动态重试，达到重试阈值后放弃当前城市数据。
    成功获取数据后，将调用write_df_into_wind_db将数据写入数据库。
    
    入参:
        url: 字符串类型, API请求的URL地址
        start_date: 字符串类型, 起始日期. 例如: "2025-04-02"
        end_date: 字符串类型, 结束日期. 默认为None，表示使用start_date作为结束日期
    """
    engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)

    headers = _set_headers(const.HEADERS)
    
    # 如果未提供结束日期，则使用起始日期作为结束日期
    if end_date is None:
        end_date = start_date
    
    # 验证并标准化日期格式
    try:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)
    except ValueError as e:
        logger.error(f"日期验证失败: {e}")
        return
    
    cur_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 确保开始日期不早于结束日期
    if cur_date < end_date:
        logger.error(f"起始日期 {start_date} 早于结束日期 {end_date}，将交换顺序")
        cur_date, end_date = end_date, cur_date
    
    while cur_date >= end_date:
        for city in const.REGIONS_FOR_CRAWLER:
            retry_cnt = 0
            delay = 1
            cur_date_str = cur_date.strftime("%Y-%m-%d")
            cur_df = fetch_one_day_wind_power_data_for_city(url, headers, city, cur_date_str)
            while cur_df is None and retry_cnt < _MAX_RETRY_NUM:
                logger.error(f"{city} 的 {cur_date_str} 当日风电数据获取失败，第 {retry_cnt+1} 次重试，等待 {delay} 秒...")
                time.sleep(delay)
                headers = _set_headers(const.HEADERS)
                cur_df = fetch_one_day_wind_power_data_for_city(url, headers, city, cur_date_str)
                retry_cnt += 1
                delay *= 2

            if cur_df is not None:
                write_df_into_wind_db(engine, cur_df, city, cur_date_str)
            else:
                logger.error(f"{city} 的 {cur_date_str} 当日风电数据获取失败，达到最大重试次数，放弃该城市数据。")
                continue

        cur_date -= pd.Timedelta(days=1)

    db_tools.release_db_connection(engine)

def fetch_multi_day_solar_power_data_for_each_city(url: str, start_date: str, end_date: str = None):
    """对各个城市进行多天光伏数据爬取，并将每一天的数据写入数据库。
      
    从起始日期到结束日期（日期递减），对每个城市依次请求单日光伏数据。
    若请求数据失败，则进行动态重试，超过重试阈值后放弃当前城市当日数据。
    成功获取数据后，调用write_df_into_solar_db将数据插入数据库。
        
    入参:
        url: 字符串类型, API请求的URL地址
        start_date: 字符串类型, 起始日期. 例如: "2025-04-02"
        end_date: 字符串类型, 结束日期. 默认为None，表示使用start_date作为结束日期
    """
    engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)

    headers = _set_headers(const.HEADERS)
    
    # 如果未提供结束日期，则使用起始日期作为结束日期
    if end_date is None:
        end_date = start_date
    
    # 验证并标准化日期格式
    try:
        start_date = validate_date(start_date)
        end_date = validate_date(end_date)
    except ValueError as e:
        logger.error(f"日期验证失败: {e}")
        return
    
    cur_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # 确保开始日期不早于结束日期
    if cur_date < end_date:
        logger.error(f"起始日期 {start_date} 早于结束日期 {end_date}，将交换顺序")
        cur_date, end_date = end_date, cur_date
    
    while cur_date >= end_date:
        for city in const.REGIONS_FOR_CRAWLER:
            retry_cnt = 0
            delay = 1
            cur_date_str = cur_date.strftime("%Y-%m-%d")
            cur_df = fetch_one_day_solar_power_data_for_city(url, headers, city, cur_date_str)
            while cur_df is None and retry_cnt < _MAX_RETRY_NUM:
                logger.error(f"{city} 的 {cur_date_str} 当日光伏数据获取失败，第 {retry_cnt+1} 次重试，等待 {delay} 秒...")
                time.sleep(delay)
                headers = _set_headers(const.HEADERS)
                cur_df = fetch_one_day_solar_power_data_for_city(url, headers, city, cur_date_str)
                retry_cnt += 1
                delay *= 2

            if cur_df is not None:
                write_df_into_solar_db(engine, cur_df, city, cur_date_str)
            else:
                logger.error(f"{city} 的 {cur_date_str} 当日光伏数据获取失败，达到最大重试次数，放弃该城市数据。")
                continue

        cur_date -= pd.Timedelta(days=1)

    db_tools.release_db_connection(engine)

def fetch_one_day_wind_power_data_for_city(url: str, headers: dict, city: str, cur_date: str) -> pd.DataFrame:
    """请求获取单日风电数据，并返回预处理后的DataFrame。
      
    根据传入的参数构造payload，发送POST请求获取数据。
    请求成功且返回的数据符合要求后，调用_preprocess_df进行预处理。
    若请求失败或返回数据异常，则返回None。

    入参:
        url: 字符串类型, API请求的URL地址.
        headers: 字典类型, 请求头信息.
        city: 字符串类型, 城市名称.
        cur_date: 字符串类型, 当前日期. 例如: "2025-04-02"

    返回:
        正常情况下返回经预处理后的风电数据DataFrame，否则返回None。
    """
    payload = _set_payload(cur_date, const.WIND_POWER_NAME, city)

    response = get_response(url, headers, payload)
    if response.status_code != 200:
        logger.error("请求失败")
        return None
    
    response_dict = response.json()
    status = response_dict.get('status')
    msg = response_dict.get('msg')
    if status != 1 or msg != '操作成功':
        logger.error("请求内容有误")
        logger.error("状态码:", status)
        logger.error("错误信息:", msg)
        logger.error("Token:", headers["Authorization"])
        return None
    
    response_data = response_dict.get('data')
    if not response_data or len(response_data) == 0:
        logger.error("获取的 response_data 为空！")
        return None
    
    processed_city_name = _process_city_name(city)

    df = _preprocess_df(response_data=response_data, city_name=processed_city_name, date=cur_date)
    if df is None:
        logger.error("预处理数据出现问题。")
        return None
        
    return df

def fetch_one_day_solar_power_data_for_city(url: str, headers: dict, city: str, cur_date: str) -> pd.DataFrame:
    """请求获取单日光伏数据，并返回预处理后的DataFrame。
      
    根据传入的参数构造payload，发送POST请求获取数据。
    请求成功且返回的数据符合要求后，调用_preprocess_df进行预处理。
    若请求失败或返回数据异常，则返回None。

    入参:
        url: 字符串类型, API请求的URL地址.
        headers: 字典类型, 请求头信息.
        city: 字符串类型, 城市名称.
        cur_date: 字符串类型, 当前日期. 例如: "2025-04-02"

    返回:
        正常情况下返回经预处理后的光伏数据DataFrame，否则返回None。
    """
    payload = _set_payload(cur_date, const.SOLAR_POWER_NAME, city)

    response = get_response(url, headers, payload)
    if response.status_code != 200:
        logger.error("请求失败")
        return None
    
    response_dict = response.json()
    status = response_dict.get('status')
    msg = response_dict.get('msg')
    if status != 1 or msg != '操作成功':
        logger.error("请求内容有误")
        logger.error("状态码:", status)
        logger.error("错误信息:", msg)
        logger.error("Token:", headers["Authorization"])
        return None
    
    response_data = response_dict.get('data')
    if not response_data or len(response_data) == 0:
        logger.error("获取的 response_data 为空！")
        return None
    
    processed_city_name = _process_city_name(city)

    df = _preprocess_df(response_data=response_data, city_name=processed_city_name, date=cur_date)
    if df is None:
        logger.error("预处理数据出现问题。")
        return None
        
    return df

def write_df_into_wind_db(engine, df: pd.DataFrame, city_name: str, cur_date_str: str):
    """将风电数据DataFrame写入数据库表中。
      
    调用db_tools.upsert_to_db，将数据写入指定的风电数据表中。

    入参:
        engine: 数据库连接引擎对象
        df: 经处理后的风电数据DataFrame
        city_name: 城市名称，字符串形式（用于日志输出）
    """
    db_tools.upsert_to_db(engine=engine, df=df, table_name=const.NEIMENG_WIND_TABLE_NAME, update_column='value')
    logger.info(f'{cur_date_str} 的 {city_name} 的风电数据已写入DB.')

def write_df_into_solar_db(engine, df: pd.DataFrame, city_name: str, cur_date_str: str):
    """将光伏数据DataFrame写入数据库表中。
      
    调用db_tools.upsert_to_db，将数据写入指定的风电数据表中。

    入参:
        engine: 数据库连接引擎对象
        df: 经处理后的光伏数据DataFrame
        city_name: 城市名称，字符串形式（用于日志输出）
    """
    db_tools.upsert_to_db(engine=engine, df=df, table_name=const.NEIMENG_SOLAR_TABLE_NAME, update_column='value')
    logger.info(f'{cur_date_str} 的 {city_name} 的光伏数据已写入DB.')

class TLSv1_2Adapter(HTTPAdapter):
    """强制 requests 使用 TLSv1.2 协议的适配器"""
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs['ssl_version'] = ssl.PROTOCOL_TLSv1_2
        return super().init_poolmanager(connections, maxsize, block, **pool_kwargs)

def get_response(url: str, headers: dict, data: dict):
    """发送POST请求获取数据响应。

    通过requests.post发送POST请求，支持指定SSL证书验证路径。
    此实现强制使用TLSv1.2协议。

    入参:
        url: 请求的URL地址
        headers: 请求头信息
        data: 请求中发送的payload数据，字典形式

    返回:
        返回requests的响应对象(response)。
    """
    session = requests.Session()
    session.mount('https://', TLSv1_2Adapter())
    response = session.post(url=url, json=data, headers=headers, verify='pem/fullchain_intermediate+root.pem')
    return response

if __name__ == '__main__': 
    pass