import numpy as np
import pandas as pd
import requests
import pickle
import os
import util.constant
from sklearn.preprocessing import StandardScaler

def fetch_all_data():
    '''
    Fetching all factory power data from database
    
    Parameters:
        null
    
    Returns:
        data in json format
    '''

    payload = {}
    response = requests.post(util.constant.database_url, json=payload)

    if response.status_code == 200:
        data = response.json()
    else:
        util.constant.logger.error('Failed to fetch all data, code: ', response.status_code)

    return data

def fetch_single_factory_data(factory_id, start_time='', end_time=''):
    '''
    Fetching data for a single factory power data from database
    
    Parameters:
        factory_id - the factory id
        start_time - the start timestamp
        end_time - the end timestamp

        1. only start_time is passed: fetch data from start_time to lastest time
        2. only end_time is passed: fetch data from earliest time to end_time
        3. both are passed: fetch data from start_time to end_time
        4. neither is passed: fetch all data
    
    Returns:
        data in json format
    '''

    if factory_id is None:
        util.constant.logger.error('Factory id is none.')
        return None

    if start_time == '':
        start_time = None
    if end_time == '':
        end_time = None

    payload = {'stationId': factory_id, 'startDate': start_time, 'endDate': end_time}
    response = requests.post(util.constant.database_fetch_api_url, json=payload)

    if response.status_code == 200:
        data = response.json()
    else:
        util.constant.logger.error('Failed to fetch data for a single factory, code: ', response.status_code)

    return data

def transfer_response_data(json_data):
    '''
    Transfer json data to Dataframe data

    Parameters:
        json_data - fetched response data from database in json format

    Returns:
        data in Dataframe format
    '''

    data = json_data.get('data')
    if data is None:
        util.constant.logger.error('data is None.')
        return data
    
    data = pd.DataFrame(data)
    data.rename(columns={'stationId': util.constant.STATION_ID}, inplace=True)
    return data

def cleanse_single_factory_data(data):
    '''
    Cleanse fetched data for single factory.

    Parameters:
        data - fetched data from database in DataFrame format

    Returns:
        cleansed data for singel factory
    '''

    if data.empty:
        util.constant.logger.error('Data is empty.')
        return
    
    data = data.sort_values(by=util.constant.STATION_ID)
    if data[util.constant.STATION_ID].iloc[0] != data[util.constant.STATION_ID].iloc[-1]:
        util.constant.logger.error('Input data contains multiple factories.')
        return
    
    data[util.constant.DATETIME] = pd.to_datetime(data[util.constant.DATETIME])
    data.set_index(util.constant.DATETIME, inplace=True)
    data.sort_index(inplace=True)

    # 数据去重
    data = data[~data.index.duplicated(keep='first')]

    # 清除数据+填补数据
    min_date = data.index.min().normalize()
    max_date = data.index.max().normalize() + pd.Timedelta(hours=23)
    complete_time_range = pd.date_range(start=min_date, end=max_date, freq='h')
    missing_timestamp = complete_time_range.difference(data.index)
    if len(missing_timestamp) == 0:
        util.constant.logger.info("The input data has 0 missing timestamp, no need to clean and interpolate.")
        return data

    util.constant.logger.info("The input data has {} missing timestamps.".format(len(missing_timestamp)))
    data_reindexed = data.reindex(complete_time_range)
    data_reindexed['date'] = data_reindexed.index.date
    missing_counts = data_reindexed[util.constant.POWER].isna().groupby(data_reindexed['date']).sum()
    days_to_drop = missing_counts[missing_counts > util.constant.INTERPOLATE_THRESHOLD].index

    data_cleansed = data_reindexed[~data_reindexed['date'].isin(days_to_drop)].copy()
    data_cleansed.drop(columns='date', inplace=True)
    data_cleansed = data_cleansed.infer_objects(copy=False)
    data_cleansed[util.constant.POWER] = data_cleansed[util.constant.POWER].interpolate(method='time')
    data_cleansed[util.constant.STATION_ID] = data_cleansed[util.constant.STATION_ID].bfill()
    data_cleansed[util.constant.STATION_ID] = data_cleansed[util.constant.STATION_ID].ffill()

    # 异常值处理
    mean_power = data_cleansed[util.constant.POWER].mean()
    std_power = data_cleansed[util.constant.POWER].std()

    n_std = 3
    lower_threshold = mean_power - n_std * std_power
    upper_threshold = mean_power + n_std * std_power

    # 替换 power < lower_threshold 或 power > upper_threshold 为 NaN
    data_cleansed[util.constant.POWER] = data_cleansed[util.constant.POWER].mask(
        (data_cleansed[util.constant.POWER] < lower_threshold) | (data_cleansed[util.constant.POWER] > upper_threshold),
        np.nan
    )

    # 插值填补 NaN 值
    first_valid_power = data_cleansed[util.constant.POWER].first_valid_index()
    if first_valid_power is not None:
        data_cleansed.loc[:first_valid_power, util.constant.POWER] = data_cleansed.at[first_valid_power, util.constant.POWER]
    data_cleansed[util.constant.POWER] = data_cleansed[util.constant.POWER].interpolate(method='time')
    
    return data_cleansed

def normalize_data(data):
    '''
    Normalize input data.

    Parameters:
        data - cleansed data

    Returns:
        normalized data

    Output:
        pickle file
    '''

    if data.empty:
        util.constant.logger.error("Input data is empty for normalization.")
        return None

    cur_factory_id = data[util.constant.STATION_ID].iloc[0]
    norm_model_file = 'saved_norm/' + str(cur_factory_id) + '.pickle'
    data = data.drop(columns=[util.constant.STATION_ID])

    # 已有文件
    if os.path.exists(norm_model_file):
        util.constant.logger.info("Already had normalized model.")
        with open(norm_model_file, 'rb') as f:
            saved_model = pickle.load(f)
            normalized_data_np = saved_model.transform(data)

    # 没有文件
    else:
        util.constant.logger.info("No normalized model, creating one.")
        if not os.path.exists('saved_norm/'):
            os.makedirs('saved_norm')

        std_scaler = StandardScaler()
        normalized_data_np = std_scaler.fit_transform(data)
        with open(norm_model_file, 'wb') as f:
            pickle.dump(std_scaler, f)

    normalized_data_df = pd.DataFrame(normalized_data_np, columns=[util.constant.POWER])
    normalized_data_df.index = data.index
    normalized_data_df[util.constant.DATETIME] = normalized_data_df.index

    normalized_data_df[util.constant.STATION_ID] = cur_factory_id
    return normalized_data_df
    
def output_to_file(normalized_data):
    '''
    Output final data to csv
    
    Parameters:
        normalized_data - normalized data

    Output:
        csv file
    '''

    if not os.path.exists('data_cleansing/output/'):
        os.makedirs('data_cleansing/output')

    normalized_data.to_csv('data_cleansing/output/' + 'tmp.csv', index=False)
    return