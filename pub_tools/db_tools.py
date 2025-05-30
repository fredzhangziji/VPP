"""
此模块包含和数据库相关的功能的所有函数。
"""

import requests
import time
from sqlalchemy import create_engine, select, MetaData, Table, insert
from sqlalchemy.dialects.mysql import insert as mysql_insert
import pandas as pd
from . import const

import logging
import pub_tools.logging_config
logger = logging.getLogger(__name__)

def get_history_weather_data_from_web(pos={'呼和浩特': [111.755509, 40.848423]}, start_time='2022-12-31 17:00:00',
                             end_time='2023-01-02 16:00:00'):
    data_type = 'era5_land'
    url = f"https://api-pro-openet.terraqt.com/v1/{data_type}/point"

    headers = {
        'Content-Type': 'application/json',
        'token': 'lBDMwUjM2UzN2ADMwQWZwEjNzYjM2cDM'
    }

    for city, coord in pos.items():
        longitude, latitude = coord[0], coord[1]
        request = {
            'start_time': start_time,
            'end_time': end_time,
            'lon': longitude,
            'lat': latitude,
            'mete_vars': const.WEATHER_FEATURE
        }

        response = requests.request("POST", url, headers=headers, json=request)
        try:
            response.raise_for_status()
            # 尝试解析 JSON，如失败则捕获异常
            response_json = response.json()
        except Exception as e:
            logger.error(f"City {city} request failed or returned non JSON format data, error info: {e}")
            logger.error("Returned content: ", response.text)
            continue  # 跳过处理该城市

        # 若 response_json 正常，则继续获取数据
        try:
            data = response_json['data']
            values = data['data'][0]['values']
            timestamp = data['timestamp']
        except Exception as e:
            logger.error(f"City {city} returned data format error, error info: {e}")
            continue

        df = pd.DataFrame(values, index=timestamp)
        df.index.name = 'datetime'
        df.columns = data['mete_var']
        df['city'] = city
        df.to_csv('../data/tmp_history_weather_data_for_' + city + '.csv')

        time.sleep(2)

def get_db_connection(db_config):
    db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset=utf8mb4"
    engine = create_engine(db_url, echo=False)
    metadata = MetaData()
    logger.info('Database connections have been created.')

    return engine, metadata

def release_db_connection(engine):
    engine.dispose()
    logger.info("Database connections have been released.")

def read_from_db(engine, query):
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def write_to_db(engine, df, table_name, if_exists):
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    logger.info(f"Data has been written into {table_name}.")

def upsert_to_db(engine, df, table_name, update_column='value'):
    """
    使用 SQLAlchemy 实现 MySQL 的 upsert 操作：
    对于重复的记录（比如以 city_name, date_time, model 为唯一键），只更新指定的字段，
    而其余字段保持原值。

    参数:
        engine: SQLAlchemy engine 对象
        df: 要插入的 DataFrame
        table_name: 目标表名
        update_column: 当发生重复时要更新的列名，默认为 'value'
    """
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    
    # 将 DataFrame 转换为字典列表记录
    data = df.to_dict(orient='records')
    if not data:
        logger.error("没有数据插入")
        return
    
    # 构造插入语句
    stmt = mysql_insert(table).values(data)
    # 当唯一键冲突时，仅更新指定字段为新值    
    upsert_stmt = stmt.on_duplicate_key_update(**{update_column: getattr(stmt.inserted, update_column)})
    
    with engine.begin() as conn:
        conn.execute(upsert_stmt)
    logger.info(f"Upsert 执行完成，数据已写入 {table_name}。")

if __name__ == '__main__':
    history_weather = get_history_weather_data_from_web(const.CITY_POS, '2025-03-26 00:00:00', '2025-03-29 00:00:00')

    # test read data
    query = "SELECT * FROM some_table LIMIT 10;"
    df_sample = read_from_db(query)
    print("sample:")
    print(df_sample.head())
    
    # test write data
    test_df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"]
    })
    write_to_db(test_df, "test_table", if_exists='append')