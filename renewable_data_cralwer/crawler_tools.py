import requests
from pub_tools import const, db_tools
from datetime import date
from sqlalchemy import create_engine, select, MetaData, Table
from sqlalchemy.exc import NoSuchTableError
import pandas as pd
from snowflake import SnowflakeGenerator
import time
import logging

_MAX_RETRY_NUM = 5

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler('renewable_data_crawler.log', mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def _set_headers(headers: dict) -> dict:
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
    payload = {
        'time': date,
        'name': type,
        'area': city
    }
    
    return payload

def _fix_datetime(dt_str):
    date_part, time_part = dt_str.split(" ")
    if time_part == "24:00":
        new_date = pd.to_datetime(date_part) + pd.Timedelta(days=1)
        return new_date.strftime("%Y-%m-%d") + " 00:00:00"
    else:
        return pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M").strftime("%Y-%m-%d %H:%M:%S")
    
def _preprocess_df(response_data: list, city_name: str, date: str) -> pd.DataFrame:
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
    city = city.strip("'")
    if "','" in city or "','" in city:
        city = city.replace("','", "+")
    return city

def fetch_multi_day_wind_power_data_for_each_city(url: str, start_date: str, end_date: str = '2023-01-01'):
    engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)

    headers = _set_headers(const.HEADERS)
    
    cur_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
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
                write_df_into_wind_db(engine, cur_df, city)
            else:
                logger.error(f"{city} 的 {cur_date_str} 当日风电数据获取失败，达到最大重试次数，放弃该城市数据。")
                continue

        cur_date -= pd.Timedelta(days=1)

    db_tools.release_db_connection(engine)
    return

def fetch_multi_day_solar_power_data_for_each_city(url: str, start_date: str, end_date: str = '2023-01-01'):
    engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)

    headers = _set_headers(const.HEADERS)
    
    cur_date = pd.to_datetime(start_date, format="%Y-%m-%d")
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d")
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
                write_df_into_solar_db(engine, cur_df, city)
            else:
                logger.error(f"{city} 的 {cur_date_str} 当日光伏数据获取失败，达到最大重试次数，放弃该城市数据。")
                continue

        cur_date -= pd.Timedelta(days=1)

    db_tools.release_db_connection(engine)
    return

def fetch_one_day_wind_power_data_for_city(url: str, headers: dict, city: str, cur_date: str) -> pd.DataFrame:
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

def write_df_into_wind_db(engine, df: pd.DataFrame, city_name: str):
    db_tools.upsert_to_db(engine=engine, df=df, table_name=const.NEIMENG_WIND_TABLE_NAME)
    return

def write_df_into_solar_db(engine, df: pd.DataFrame, city_name: str):
    db_tools.upsert_to_db(engine=engine, df=df, table_name=const.NEIMENG_SOLAR_TABLE_NAME)
    return

def get_response(url: str, headers: dict, data: dict):
    response = requests.post(url=url, json=data, headers=headers)
    return response

if __name__ == '__main__': 
    url = 'https://www.imptc.com/api/sctjfxyyc/crqwxxfb/getXnyfdnlycData'

    response_data = fetch_one_day_wind_power_data_for_city(url, const.HEADERS, const.REGIONS_FOR_CRAWLER[0])
    print(type(response_data))
    print(type(response_data[0]))
    print(response_data)

    write_df_into_wind_db(response_data)