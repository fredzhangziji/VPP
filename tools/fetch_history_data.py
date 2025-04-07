import requests
import time
from sqlalchemy import create_engine
import pandas as pd

weather_features = [
    't2m',      # temperature at 2 meters
    'ws10m',    # wind speed at 10 meters
    'wd10m',    # wind direction at 10 meters
    'sp'        # surface pressure
]

city_pos = {
    '呼和浩特': [111.755509, 40.848423],
    '包头': [109.846544, 40.662929],
    '乌海': [106.800391, 39.662006],
    '鄂尔多斯+薛家湾': [109.787443, 39.614482],
    '巴彦淖尔': [107.394398, 40.749359],
    '乌兰察布': [113.139468, 41.000748],
    '锡林郭勒': [116.054391, 43.939423],
    '阿拉善盟': [105.735377, 38.858276]
}

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
            'mete_vars': weather_features
        }

        response = requests.request("POST", url, headers=headers, json=request)
        try:
            response.raise_for_status()
            # 尝试解析 JSON，如失败则捕获异常
            response_json = response.json()
        except Exception as e:
            print(f"城市 {city} 请求失败或返回非 JSON 格式数据，错误信息：{e}")
            print("响应内容：", response.text)
            continue  # 跳过处理该城市

        # 若 response_json 正常，则继续获取数据
        try:
            data = response_json['data']
            values = data['data'][0]['values']
            timestamp = data['timestamp']
        except Exception as e:
            print(f"城市 {city} 返回数据结构异常，错误信息：{e}")
            continue

        df = pd.DataFrame(values, index=timestamp)
        df.index.name = 'datetime'
        df.columns = data['mete_var']
        df['city'] = city
        df.to_csv('../data/tmp_history_weather_data_for_' + city + '.csv')

        time.sleep(2)


def get_history_weather_data_from_db():
    return

# 数据库连接配置
DB_CONFIG = {
    'host': '10.5.0.10',
    'user': 'root',
    'password': 'kunyu2023rds',
    'database': 'vpp_service',
    'port': 3306
}

def get_db_connection(db_config=DB_CONFIG):
    db_uri = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset=utf8mb4"
    engine = create_engine(db_uri, echo=False)
    return engine

def read_from_db(query):
    engine = get_db_connection()
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def write_to_db(df, table_name, if_exists='append'):
    engine = get_db_connection()
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    print(f"数据已写入表 {table_name}.")

if __name__ == '__main__':
    history_weather = get_history_weather_data_from_web(city_pos, '2025-03-26 00:00:00', '2025-03-29 00:00:00')

    # test read data
    query = "SELECT * FROM some_table LIMIT 10;"
    df_sample = read_from_db(query)
    print("读取数据示例:")
    print(df_sample.head())
    
    # test write data
    test_df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"]
    })
    write_to_db(test_df, "test_table", if_exists='append')