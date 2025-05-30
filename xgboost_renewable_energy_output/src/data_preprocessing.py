"""
此模块包含了预处理风能和太阳能数据所需的功能函数，用于XGBoost模型的训练和预测。
模块提供从数据库获取历史、预测天气数据和能源输出数据的功能，并进行数据清洗、特征工程和可视化。
"""

import pandas as pd
from sqlalchemy import Table, select
from pub_tools import db_tools, const
import numpy as np
import matplotlib.pyplot as plt

import logging
import pub_tools.logging_config
logger = logging.getLogger('xgboost')

CITIES_FOR_WEATHER = [
    '乌兰察布市',
    '锡林郭勒盟',
    '包头市',
    '巴彦淖尔市',
    '阿拉善盟',
    '呼和浩特市',
    '鄂尔多斯市'
    ]

CITIES_FOR_POWER = [
    '乌兰察布',
    '锡林郭勒',
    '包头',
    '巴彦淖尔',
    '阿拉善盟',
    '呼和浩特',
    '鄂尔多斯+薛家湾',
    ]

CITY_NAME_MAPPING_DICT = {
    '乌兰察布': '乌兰察布市',
    '锡林郭勒': '锡林郭勒盟',
    '包头': '包头市',
    '巴彦淖尔': '巴彦淖尔市',
    '阿拉善盟': '阿拉善盟',
    '呼和浩特': '呼和浩特市',
    '鄂尔多斯+薛家湾': '鄂尔多斯市',
}

def get_history_weather_data_for_city(city: str) -> pd.DataFrame:
    """获取指定城市的指定历史实测天气数据
    
    根据传入的城市, 获取该城市的 t2m, ws100m, wd100m, gust, sp 的实测天气数据.

    入参:
        city: 字符串类型, 表示要获取哪座城市的天气

    返回:
        weather_df: pd.Dataframe类型, 获取到的指定城市的历史实测天气数据
    """
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
    terraqt_weather = Table('terraqt_weather', metadata, autoload_with=engine)
    query = select(terraqt_weather).where(
        (terraqt_weather.c.model == 'gdas_surface') &
        (terraqt_weather.c.region_name == city)
    )

    weather_df = db_tools.read_from_db(engine, query)

    weather_df['ts'] = pd.to_datetime(weather_df['ts'])
    weather_df.rename(columns={'ts': 'datetime'}, inplace=True)
    weather_df.set_index('datetime', inplace=True)
    weather_df.drop(
        columns=[
            'id', 'region_code', 'region_name', 'model', 'ws10m', 'wd10m', 'irra', 'tp', 'lng', 'lat', 'time_fcst'
            ],
        inplace=True
    )

    weather_df.sort_index(inplace=True)

    db_tools.release_db_connection(engine)

    return weather_df

def get_predicted_weather_data_for_city(city: str) -> pd.DataFrame:
    """获取指定城市的指定预测的天气数据
    
    根据传入的城市, 获取该城市的 t2m, ws100m, wd100m, gust, sp 的预测天气数据.

    入参:
        city: 字符串类型, 表示要获取该城市的天气

    返回:
        weather_df: pd.Dataframe类型, 获取到的预测天气数据
    """
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
    terraqt_weather = Table('terraqt_weather', metadata, autoload_with=engine)
    query = select(terraqt_weather).where(
        (terraqt_weather.c.model == 'gfs_surface') &
        (terraqt_weather.c.region_name == city)
    )

    weather_df = db_tools.read_from_db(engine, query)

    weather_df['ts'] = pd.to_datetime(weather_df['ts'])
    weather_df.rename(columns={'ts': 'datetime'}, inplace=True)
    weather_df.set_index('datetime', inplace=True)
    weather_df.drop(
        columns=[
            'id', 'region_code', 'region_name', 'model', 'ws10m', 'wd10m', 'irra', 'tp', 'lng', 'lat', 'time_fcst'
            ],
        inplace=True
    )

    weather_df.sort_index(inplace=True)

    db_tools.release_db_connection(engine)

    return weather_df

def get_history_wind_power_for_city(city: str) -> pd.DataFrame:
    """获取指定城市的实测风电出力数据
    
    根据传入的城市, 获取该城市的实测风电出力数据, 并进行简单的数据处理

    入参:
        city: 字符串类型, 表示要获取哪座城市的天气

    返回:
        output_df: pd.Dataframe类型, 获取到的实测风电出力数据
    """
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
    neimeng_wind_output = Table('neimeng_wind_power', metadata, autoload_with=engine)
    query = select(neimeng_wind_output).where(
        (neimeng_wind_output.c.type == '3') &
        (neimeng_wind_output.c.city_name == city)
    )

    output_df = db_tools.read_from_db(engine, query)

    db_tools.release_db_connection(engine)

    output_df.drop(columns=['type', 'id', 'city_name'], inplace=True)
    output_df.rename(columns={'date_time': 'datetime', 'value': 'wind_output'}, inplace=True)
    output_df['datetime'] = pd.to_datetime(output_df['datetime'])
    output_df.set_index('datetime', inplace=True)
    output_df['wind_output'] = output_df['wind_output'].resample('h', closed='right', label='right').mean()
    output_df.dropna(inplace=True)

    output_df.sort_index(inplace=True)
    return output_df

def get_history_solar_power_for_city(city: str) -> pd.DataFrame:
    """获取指定城市的实测光伏出力数据
    
    根据传入的城市, 获取该城市的实测光伏出力数据, 并进行简单的数据处理

    入参:
        city: 字符串类型, 表示要获取哪座城市的天气

    返回:
        output_df: pd.Dataframe类型, 获取到的实测光伏出力数据
    """
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
    neimeng_solar_output = Table('neimeng_solar_power', metadata, autoload_with=engine)
    query = select(neimeng_solar_output).where(
        (neimeng_solar_output.c.type == '3') &
        (neimeng_solar_output.c.city_name == city)
    )

    output_df = db_tools.read_from_db(engine, query)

    db_tools.release_db_connection(engine)

    output_df.drop(columns=['type', 'id', 'city_name'], inplace=True)
    output_df.rename(columns={'date_time': 'datetime', 'value': 'solar_output'}, inplace=True)
    output_df['datetime'] = pd.to_datetime(output_df['datetime'])
    output_df.set_index('datetime', inplace=True)
    output_df['solar_output'] = output_df['solar_output'].resample('H', closed='right', label='right').mean()
    output_df.dropna(inplace=True)

    output_df.sort_index(inplace=True)
    return output_df

def merge_weather_and_power_df(weather_df: pd.DataFrame, power_df: pd.DataFrame) -> pd.DataFrame:
    """将天气和出力数据进行合并
    
    通过索引（时间戳）将天气数据和能源出力数据进行内连接，并删除合并后的空值记录。
    
    入参:
        weather_df: 包含天气数据的DataFrame
        power_df: 包含能源出力数据的DataFrame
        
    返回:
        合并后的DataFrame，包含天气数据和能源出力
    """
    merged_df = pd.merge(weather_df, power_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.dropna()

    return merged_df

def _wind_season(m):
    """判断月份属于哪个风季
    
    根据月份确定是大风季还是小风季。
    
    入参:
        m: 月份数字
        
    返回:
        'big'表示大风季，'small'表示小风季
    """
    return 'big' if (1 <= m <= 5 or 9 <= m <= 12) else 'small'

def _engineer_weather_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """对输入的天气特征进行工程化处理
    
    根据现有天气数据创建额外的特征，包括风速平方、立方、滚动平均值、风向分量、
    温度变化趋势等衍生特征，增强模型对天气变化模式的捕捉能力。
    
    入参:
        df_input: 包含原始天气特征的DataFrame
        
    返回:
        添加了工程化天气特征的DataFrame
    """
    df = df_input.copy()
    if 'ws100m' in df.columns:
        df['ws100m_squared'] = df['ws100m'] ** 2
        df['ws100m_cubed'] = df['ws100m'] ** 3
        df['ws100m_roll_mean_6hr'] = df['ws100m'].rolling(window=6, min_periods=1).mean()
        df['ws100m_roll_mean_12hr'] = df['ws100m'].rolling(window=12, min_periods=1).mean()
        df['ws100m_roll_max_6hr'] = df['ws100m'].rolling(window=6, min_periods=1).max()

    if 'gust' in df.columns:
        df['gust_squared'] = df['gust'] ** 2
        df['gust_roll_mean_6hr'] = df['gust'].rolling(window=6, min_periods=1).mean()
        df['gust_roll_max_6hr'] = df['gust'].rolling(window=6, min_periods=1).max()

    if 'wd100m' in df.columns and 'ws100m' in df.columns: # Ensure both columns exist
        # wd100m is in degrees, convert to radians for sin/cos
        wd_rad = np.deg2rad(df['wd100m'])
        df['u_wind_100m'] = df['ws100m'] * np.cos(wd_rad)
        df['v_wind_100m'] = df['ws100m'] * np.sin(wd_rad)

    if 't2m' in df.columns:
        df['delta_t2m_3hr'] = df['t2m'].diff(periods=3).fillna(0)
        df['t2m_roll_mean_6hr'] = df['t2m'].rolling(window=6, min_periods=1).mean()

    if 'sp' in df.columns:
        df['pressure_trend'] = df['sp'].diff().fillna(0)
        # Ensure first diff NaN is handled if df is short or at the beginning
        if not df.empty and pd.isna(df['pressure_trend'].iloc[0]) and len(df) == 1:
             df['pressure_trend'].iloc[0] = 0
        elif not df.empty and pd.isna(df['pressure_trend'].iloc[0]): # Check if it's the first row overall
            first_valid_index = df['pressure_trend'].first_valid_index()
            if first_valid_index == df.index[0]: # if it is the first row of the whole series
                 df['pressure_trend'].iloc[0] = 0

    if 'gust' in df.columns and 'ws100m' in df.columns:
        df['gustiness_abs_100m'] = (df['gust'] - df['ws100m']).fillna(0)
        df['gustiness_rel_100m'] = (df['gustiness_abs_100m'] / df['ws100m']).replace([np.inf, -np.inf], 0).fillna(0)
        df['gust_x_ws100m'] = df['gust'] * df['ws100m']

    return df

def _engineer_time_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """对输入数据添加时间相关特征
    
    基于时间索引创建周期性时间特征，如月份、小时的正弦/余弦表示，
    一年中的第几天，一周中的第几天等，可以帮助模型捕捉时间周期性模式。
    
    入参:
        df_input: 包含时间索引的DataFrame
        
    返回:
        添加了时间特征的DataFrame
    """
    df = df_input.copy()
    # Ensure datetime index is available for time feature engineering
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df = df.set_index('datetime', drop=False) # Keep datetime column if needed later
        else:
            # This case should be handled by ensuring df.index is datetime before calling
            logger.warning("Warning: DataFrame index is not DatetimeIndex and 'datetime' column not found for time feature engineering.")
            return df_input # Return original if no time reference

    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['day_of_week'] = df.index.dayofweek # Monday=0, Sunday=6
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['week_of_year'] = df.index.isocalendar().week.astype(int)

    # Existing feature from _add_more_features
    df['wind_season'] = df.index.month.map(_wind_season) # This is already string, will be one-hot encoded later
    return df

def _engineer_interaction_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """创建天气与时间的交互特征
    
    通过组合天气数据和时间特征，生成交互项特征，
    帮助模型捕捉天气因素在不同时间条件下的变化效应。
    
    入参:
        df_input: 包含天气和时间特征的DataFrame
        
    返回:
        添加了交互特征的DataFrame
    """
    df = df_input.copy()
    if 'ws100m' in df.columns and 'hour_sin' in df.columns and 'hour_cos' in df.columns:
        df['ws100m_x_hour_sin'] = df['ws100m'] * df['hour_sin']
        df['ws100m_x_hour_cos'] = df['ws100m'] * df['hour_cos']

    if 't2m' in df.columns and 'month_sin' in df.columns and 'month_cos' in df.columns:
        df['t2m_x_month_sin'] = df['t2m'] * df['month_sin']
        df['t2m_x_month_cos'] = df['t2m'] * df['month_cos']
    
    # Example for gust and wind_season (if wind_season were numerical, or after one-hot encoding)
    # For now, assuming wind_season is categorical and handled later by get_dummies
    # if 'gust' in df.columns and 'wind_season_big' in df.columns: # Assuming 'wind_season_big' is a one-hot encoded column
    #     df['gust_x_wind_season_big'] = df['gust'] * df['wind_season_big']

    return df

def _add_more_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程主函数，综合添加多种特征
    
    调用各个特征工程子函数，依次添加天气特征、时间特征和交互特征。
    
    入参:
        df: 原始数据DataFrame
        
    返回:
        完成特征工程后的DataFrame
    """
    df_engineered = df.copy()
    df_engineered = _engineer_weather_features(df_engineered)
    df_engineered = _engineer_time_features(df_engineered) # df.index should be datetime
    df_engineered = _engineer_interaction_features(df_engineered)

    return df_engineered

def _add_more_features_for_future(df_future_weather: pd.DataFrame, df_historical_with_output: pd.DataFrame) -> pd.DataFrame:
    """为未来天气数据添加特征
    
    针对预测使用场景，为未来天气数据添加特征，但不包含依赖于历史输出的滞后项特征。
    
    入参:
        df_future_weather: 未来天气数据DataFrame
        df_historical_with_output: 包含历史数据和输出的DataFrame，用于特征参考
        
    返回:
        添加了特征的未来天气数据DataFrame
    """
    df = df_future_weather.copy()
    if df.empty:
        return df

    df = _engineer_weather_features(df)
    df = _engineer_time_features(df) # df.index should be datetime
    df = _engineer_interaction_features(df)
    
    return df

def _deduplicate_data(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """对数据进行去重处理
    
    检查并删除重复的数据记录，保留首次出现的记录。
    可以通过subset参数指定用于判断重复的列子集。
    
    入参:
        df: 需要去重的DataFrame
        subset: 用于判断重复的列名列表，默认为None表示使用所有列
        
    返回:
        去重后的DataFrame
    """
    dup_mask = df.duplicated(subset=subset, keep=False)
    if dup_mask.any():
        logger.info('Duplicate rows found:\n%s' % df[dup_mask])
        logger.info(f'Total {len(df[dup_mask])} duplicate rows found')
        df = df.drop_duplicates(subset=subset, keep='first')
        logger.info('Deduplication complete; kept first occurrence of each duplicate row')
    else:
        logger.info('No duplicate rows found')

    return df

def _deal_missing_data(df: pd.DataFrame, threshold_percent: float = 0.05) -> pd.DataFrame:
    """处理缺失数据.

    如果某一天的缺失数据高于阈值百分比, 则丢弃该日数据.

    入参:
        df: 待处理的数据
        threshold_percent: 每天最多允许的缺失小时占比 (目前默认0.1, 即10%)

    返回:
        df_filled: 处理过的数据
    """
    # Ensure index is datetime for reindexing and interpolation
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df = df.set_index('datetime')
        else:
            logger.warning("Warning: Cannot perform missing data handling without a datetime index or 'datetime' column.")
            return df # Or raise an error

    complete_time_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='h'
    )
    initial_missing = complete_time_range.difference(df.index)
    logger.info(f"Initial missing timestamps: {len(initial_missing)}")
    if len(initial_missing) == 0:
        logger.info("No need to deal missing data.")
        return df

    logger.info("Dealing with missing data.")
    logger.info("Initial missing dates:\n", pd.Series(initial_missing.date).value_counts())

    df_reindexed = df.reindex(complete_time_range)
    # mark originally missing timestamps before interpolation
    df_reindexed['missing_original'] = df_reindexed.isna().any(axis=1).astype(int).astype(str)
    df_reindexed['date'] = df_reindexed.index.date
    missing_flag = df_reindexed.drop(columns=['date', 'missing_original']).isna().any(axis=1)
    missing_count_per_day = missing_flag.groupby(df_reindexed['date']).sum()
    logger.info("Missing timestamps per day:")
    logger.info(missing_count_per_day)

    # calculate maximum allowed missing hours per day (e.g., 10% of 24h = 2.4 -> 2 hours)
    threshold_hours = max(1, int(threshold_percent * 24))
    logger.info(f"Using threshold_percent={threshold_percent:.2f}, threshold_hours={threshold_hours}")
    # drop any day with missing count above threshold_hours
    days_to_drop = missing_count_per_day[missing_count_per_day > threshold_hours].index
    logger.info(f"\nDays to drop: {len(days_to_drop)}")

    df_cleaned = df_reindexed[~df_reindexed['date'].isin(days_to_drop)].copy()
    df_cleaned.drop(columns='date', inplace=True)
    # Optionally, keep or drop 'missing_original' before returning.
    df_cleaned.drop(columns='missing_original', inplace=True)
    df_filled = df_cleaned.interpolate(method='time')
    df_filled.ffill(inplace=True)
    df_filled.bfill(inplace=True)

    expected_range = pd.date_range(
        start=df_filled.index.min(),
        end=df_filled.index.max(),
        freq='h'
    )
    missing_timestamps = expected_range.difference(df_filled.index)
    logger.info(f"Total missing timestamps: {len(missing_timestamps)}")
    logger.info("\nMissing timestamps by date:")
    missing_dates = pd.Series(missing_timestamps.date).value_counts().sort_index()
    logger.info(missing_dates)
    
    return df_filled

def preprocess_data(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """数据预处理.

    包括数据去重/处理缺失值/特征工程.
    
    入参:
        df: 待处理的数据
        city: 数据的具体城市

    返回:
        df_processed: 预处理完成的数据
    """
    # Ensure index is datetime before proceeding, if not, try to set it from 'datetime' column
    if not isinstance(df.index, pd.DatetimeIndex) and 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex or a 'datetime' column for preprocessing.")

    df_deduplicated = _deduplicate_data(df)
    df_filled = _deal_missing_data(df_deduplicated) # This function now also ensures datetime index
    
    # _add_more_features expects df_filled to have a DatetimeIndex
    df_more_features = _add_more_features(df_filled)
    
    # Add 'datetime' column back if it was part of the index and used for features, for compatibility with set_time_wise_feature
    # set_time_wise_feature will use and then drop it.
    df_more_features['datetime'] = df_more_features.index

    return df_more_features

def is_dataset_valid(df: pd.DataFrame) -> bool:
    """验证数据集的完整性和质量
    
    打印数据集的基本信息，包括总行数、每列缺失值数量和数值范围，
    用于快速评估数据集的质量和完整性。
    
    入参:
        df: 待验证的DataFrame
        
    返回:
        bool: 数据集是否有效（目前总是返回True，仅用于日志记录）
    """
    logger.info('Data validation:')
    logger.info(f'Total rows: {len(df)}')
    logger.info('Missing values per column:')
    logger.info(df.isnull().sum())
    logger.info('\nValue ranges:')
    logger.info(df.describe())
    return not df.isnull().any().any()

def set_time_wise_feature(df: pd.DataFrame) -> pd.DataFrame:
    """为数据集添加和处理基于时间的特征
    
    将时间特征转换为字符串格式，用于后续的独热编码处理。
    添加时间索引和分组标识，为深度学习模型准备特征。
    
    入参:
        df: 包含时间信息的DataFrame（需要有datetime列或DatetimeIndex）
        
    返回:
        添加了时间特征的DataFrame（年、月、日、小时作为字符串）
    """
    df_time_features = df.copy() # Work on a copy
    df_time_features['time_idx'] = np.arange(1, len(df_time_features) + 1)
    # Only one time series: assign a single group
    df_time_features['group_id'] = "0"
    
    # Extract basic time features as strings for one-hot encoding
    # 这些是给独热编码用的，sin/cos是数值型
    if 'datetime' in df_time_features.columns and isinstance(df_time_features['datetime'].iloc[0], (pd.Timestamp, np.datetime64)):
        dt_series = pd.to_datetime(df_time_features['datetime'])
    elif isinstance(df_time_features.index, pd.DatetimeIndex):
        dt_series = df_time_features.index
    else:
        raise ValueError("DataFrame needs a 'datetime' column or DatetimeIndex for set_time_wise_feature.")

    # 兼容 DatetimeIndex 和 Series
    if isinstance(dt_series, pd.DatetimeIndex):
        df_time_features['year'] = dt_series.year.astype(str)
        df_time_features['month'] = dt_series.month.astype(str)
        df_time_features['day'] = dt_series.day.astype(str)
        df_time_features['hour'] = dt_series.hour.astype(str)
    else:
        df_time_features['year'] = dt_series.dt.year.astype(str)
        df_time_features['month'] = dt_series.dt.month.astype(str)
        df_time_features['day'] = dt_series.dt.day.astype(str)
        df_time_features['hour'] = dt_series.dt.hour.astype(str)
    
    # Drop the datetime column if it exists and was used
    if 'datetime' in df_time_features.columns:
        df_time_features.drop(columns=['datetime'], inplace=True)

    return df_time_features

def plot_predictions(true_values: pd.Series, predicted_values: np.ndarray, time_index: pd.DatetimeIndex, title: str, filename: str, rmse: float, mae: float, r2: float):
    """绘制并保存预测结果对比图
    
    创建一个包含实际值和预测值的对比图表，同时在图表上显示评估指标。
    图表将被保存到指定路径，如果提供的是相对路径，则保存到src/figures目录下。
    
    入参:
        true_values: 实际观测值的Series
        predicted_values: 模型预测值的ndarray
        time_index: 时间索引，用于X轴
        title: 图表标题
        filename: 保存的文件名或路径
        rmse: 均方根误差值
        mae: 平均绝对误差值
        r2: R²决定系数
    """
    # 如果filename不是绝对路径，则保存到src/figures下
    import os
    if not os.path.isabs(filename):
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        filename = os.path.join(figures_dir, filename)
    
    plt.figure(figsize=(15, 7))
    plt.plot(time_index, true_values, label='Actual Values', color='blue', marker='.', linewidth=1)
    plt.plot(time_index, predicted_values, label='Predicted Values', color='red', linestyle='--', marker='.', linewidth=1)
    
    metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
    # 在图的右上角添加文本框显示指标
    plt.text(0.98, 0.98, metrics_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
    
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel('Wind Output')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    try:
        plt.savefig(filename)
        logger.info(f"Saved plot to {filename}")
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
    plt.close()
    
if __name__ == '__main__':
    CITY_FOR_POWER_DATA = '乌兰察布'
    CITY_FOR_WEATHER_DATA = CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA]

    weather_df = get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
    power_df = get_history_wind_power_for_city(CITY_FOR_POWER_DATA)
    merged_df = merge_weather_and_power_df(weather_df, power_df)
    
    # preprocess_data now includes the new feature engineering steps
    # Its output `preprocessed_df` will have a DatetimeIndex and a 'datetime' column.
    preprocessed_df = preprocess_data(merged_df.copy(), CITY_FOR_POWER_DATA) # Use .copy() for safety
    print("Sample of preprocessed_df with new features (before time_wise features like year, month as str):")
    print(preprocessed_df.head())
    print(preprocessed_df.info())

    # time_wise_df will have string versions of year, month, day, hour for one-hot encoding
    # and will drop the 'datetime' column.
    time_wise_df = set_time_wise_feature(preprocessed_df.copy()) 
    print("\nSample of time_wise_df for training (after string time features, datetime column dropped):")
    print(time_wise_df.head())
    print(time_wise_df.info())