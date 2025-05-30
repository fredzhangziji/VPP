import requests
import time
from sqlalchemy import create_engine, select, MetaData, Table
import pandas as pd
from . import const

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
            print(f"City {city} request failed or returned non JSON format data, error info: {e}")
            print("Returned content: ", response.text)
            continue  # 跳过处理该城市

        # 若 response_json 正常，则继续获取数据
        try:
            data = response_json['data']
            values = data['data'][0]['values']
            timestamp = data['timestamp']
        except Exception as e:
            print(f"City {city} returned data format error, error info: {e}")
            continue

        df = pd.DataFrame(values, index=timestamp)
        df.index.name = 'datetime'
        df.columns = data['mete_var']
        df['city'] = city
        df.to_csv('../data/tmp_history_weather_data_for_' + city + '.csv')

        time.sleep(2)

# def get_history_weather_data_from_db(engine, query):
#     df_from_db = read_from_db(engine, query)
#     return df_from_db

# def write_prediction_weather_data_to_db(engine, df, table_name, if_exists='append'):
#     write_to_db(engine, df, table_name, if_exists)

def get_db_connection(db_config):
    db_url = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}?charset=utf8mb4"
    engine = create_engine(db_url, echo=False)
    metadata = MetaData()
    print('Database connections have been created.')

    return engine, metadata

def release_db_connection(engine):
    engine.dispose()
    print("Database connections have been released.")

def read_from_db(engine, query):
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def write_to_db(engine, df, table_name, if_exists):
    with engine.connect() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    print(f"Data has been written into {table_name}.")

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