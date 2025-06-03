"""
数据预处理模块
用于加载、清洗和预处理风力发电和天气数据，为Informer模型训练准备特征
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from pub_tools import db_tools, const
from sqlalchemy import text
import logging
from datetime import datetime, timedelta
import pickle

logger = logging.getLogger('informer')

# 定义所有可导出的函数，方便使用 from data_preprocessing import * 方式导入
__all__ = [
    'CITIES_FOR_WEATHER',
    'CITIES_FOR_POWER', 
    'CITY_NAME_MAPPING_DICT',
    'get_history_weather_data_for_city', 
    'get_history_wind_power_for_city',
    'get_predicted_weather_data_for_city',
    'merge_weather_and_power_df',
    'preprocess_data',
    'set_time_wise_feature',
    '_add_more_features_for_future',
    'plot_predictions',
    'create_datasets',
    'apply_feature_scaling'
]

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


def get_history_weather_data_for_city(city_name):
    """
    从数据库获取指定城市的历史天气数据
    
    参数:
        city_name: 城市名称，应该是CITIES_FOR_WEATHER中的一个值
        
    返回:
        pandas DataFrame: 包含历史天气数据，以时间为索引
    """
    logger.info(f"获取 {city_name} 的历史天气数据...")
    try:
        engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
        
        query = text("""
            SELECT 
                date_time, temp, humidity, pressure, wind_speed, wind_dir
            FROM 
                weather_history_data
            WHERE 
                city_name = :city_name
            ORDER BY 
                date_time ASC
        """)
        
        df = pd.read_sql(query, engine, params={'city_name': city_name})
        
        if df.empty:
            logger.warning(f"警告: 未找到 {city_name} 的历史天气数据")
            return pd.DataFrame()
            
        # 设置时间索引
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
        
        logger.info(f"成功获取 {city_name} 的历史天气数据，共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        logger.error(f"获取 {city_name} 的历史天气数据失败: {e}")
        return pd.DataFrame()
    finally:
        db_tools.release_db_connection(engine)


def get_history_wind_power_for_city(city_name):
    """
    从数据库获取指定城市的历史风力发电数据
    
    参数:
        city_name: 城市名称，应该是CITIES_FOR_POWER中的一个值
        
    返回:
        pandas DataFrame: 包含历史风力发电数据，以时间为索引
    """
    logger.info(f"获取 {city_name} 的历史风力发电数据...")
    try:
        engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
        
        query = text("""
            SELECT 
                date_time, wind_output
            FROM 
                wind_output_history
            WHERE 
                city_name = :city_name
            ORDER BY 
                date_time ASC
        """)
        
        df = pd.read_sql(query, engine, params={'city_name': city_name})
        
        if df.empty:
            logger.warning(f"警告: 未找到 {city_name} 的历史风力发电数据")
            return pd.DataFrame()
            
        # 设置时间索引
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
        
        logger.info(f"成功获取 {city_name} 的历史风力发电数据，共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        logger.error(f"获取 {city_name} 的历史风力发电数据失败: {e}")
        return pd.DataFrame()
    finally:
        db_tools.release_db_connection(engine)


def get_predicted_weather_data_for_city(city_name):
    """
    从数据库获取指定城市的未来天气预测数据
    
    参数:
        city_name: 城市名称，应该是CITIES_FOR_WEATHER中的一个值
        
    返回:
        pandas DataFrame: 包含未来天气预测数据，以时间为索引
    """
    logger.info(f"获取 {city_name} 的未来天气预测数据...")
    try:
        engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
        
        query = text("""
            SELECT 
                date_time, temp, humidity, pressure, wind_speed, wind_dir
            FROM 
                weather_forecast_data
            WHERE 
                city_name = :city_name
            ORDER BY 
                date_time ASC
        """)
        
        df = pd.read_sql(query, engine, params={'city_name': city_name})
        
        if df.empty:
            logger.warning(f"警告: 未找到 {city_name} 的未来天气预测数据")
            return pd.DataFrame()
            
        # 设置时间索引
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.set_index('date_time')
        
        logger.info(f"成功获取 {city_name} 的未来天气预测数据，共 {len(df)} 条记录")
        return df
    
    except Exception as e:
        logger.error(f"获取 {city_name} 的未来天气预测数据失败: {e}")
        return pd.DataFrame()
    finally:
        db_tools.release_db_connection(engine)


def merge_weather_and_power_df(weather_df, power_df):
    """
    合并天气和风力发电数据
    
    参数:
        weather_df: 天气数据DataFrame
        power_df: 风力发电数据DataFrame
        
    返回:
        pandas DataFrame: 合并后的数据
    """
    if weather_df.empty or power_df.empty:
        return pd.DataFrame()
        
    # 确保索引是datetime类型
    weather_df.index = pd.to_datetime(weather_df.index)
    power_df.index = pd.to_datetime(power_df.index)
    
    # 找出共同的时间范围
    common_start = max(weather_df.index.min(), power_df.index.min())
    common_end = min(weather_df.index.max(), power_df.index.max())
    
    if common_start >= common_end:
        logger.warning("警告: 天气和风力数据没有重叠的时间范围")
        return pd.DataFrame()
    
    # 过滤数据到共同时间范围
    weather_df = weather_df[(weather_df.index >= common_start) & (weather_df.index <= common_end)]
    power_df = power_df[(power_df.index >= common_start) & (power_df.index <= common_end)]
    
    # 合并数据集
    merged_df = weather_df.join(power_df, how='inner')
    
    # 检查是否有NaN值
    nan_count = merged_df.isna().sum().sum()
    if nan_count > 0:
        logger.info(f"合并数据集中有 {nan_count} 个NaN值，将进行插值处理")
        merged_df = merged_df.interpolate(method='time')
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    
    logger.info(f"合并后的数据集包含 {len(merged_df)} 条记录")
    return merged_df


def preprocess_data(merged_df, city_name):
    """
    对合并后的数据进行预处理和特征工程
    
    参数:
        merged_df: 合并后的数据DataFrame
        city_name: 城市名称
        
    返回:
        pandas DataFrame: 预处理后的数据
    """
    if merged_df.empty:
        return pd.DataFrame()
    
    # 复制数据以避免修改原始数据
    df = merged_df.copy()
    
    # 添加时间特征
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    
    # 定义季节
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['season'] = df['month'].map(season_map)
    
    # 将风向转换为正弦和余弦分量以处理周期性
    if 'wind_dir' in df.columns:
        df['wind_dir_rad'] = np.radians(df['wind_dir'])
        df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
        df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
        df.drop('wind_dir_rad', axis=1, inplace=True)  # 删除中间变量
    
    # 添加滞后特征 (前1, 2, 3, 6, 12, 24小时)
    if 'wind_output' in df.columns:
        # 时间序列特征 - 滞后项
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'wind_output_lag_{lag}'] = df['wind_output'].shift(lag)
    
    # 处理日间和夜间
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    
    # 添加与季节相关的特征
    df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
    
    # 删除NaN值 (通常是由于滞后特征导致的)
    df.dropna(inplace=True)
    
    logger.info(f"{city_name} 数据预处理完成，最终数据集包含 {len(df)} 条记录和 {df.shape[1]} 个特征")
    return df


def set_time_wise_feature(df):
    """
    添加时间序列相关特征
    
    参数:
        df: 输入数据DataFrame
        
    返回:
        pandas DataFrame: 添加了时间序列特征的数据
    """
    if 'datetime' not in df.columns:
        df['datetime'] = df.index
    
    # 添加用于时间序列分组的特征
    df['time_idx'] = range(len(df))
    df['group_id'] = 0  # 单一时间序列的情况
    
    return df


def _add_more_features_for_future(future_weather_df, history_df):
    """
    为未来天气数据添加额外特征
    
    参数:
        future_weather_df: 未来天气数据DataFrame
        history_df: 历史数据DataFrame，用于获取最近的风力发电值
        
    返回:
        pandas DataFrame: 添加了所需特征的未来数据
    """
    df = future_weather_df.copy()
    
    # 添加时间特征
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofweek'] = df.index.dayofweek
    
    # 定义季节
    season_map = {1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 0}
    df['season'] = df['month'].map(season_map)
    
    # 将风向转换为正弦和余弦分量
    if 'wind_dir' in df.columns:
        df['wind_dir_rad'] = np.radians(df['wind_dir'])
        df['wind_dir_sin'] = np.sin(df['wind_dir_rad'])
        df['wind_dir_cos'] = np.cos(df['wind_dir_rad'])
        df.drop('wind_dir_rad', axis=1, inplace=True)
    
    # 添加日间和夜间标志
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    
    # 添加冬季标志
    df['is_winter'] = ((df['month'] == 12) | (df['month'] <= 2)).astype(int)
    
    # 对于需要滞后特征的情况，我们使用历史数据的最后值
    if 'wind_output' in history_df.columns and not history_df.empty:
        last_values = {}
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(history_df) >= lag:
                last_values[lag] = history_df['wind_output'].iloc[-lag]
            else:
                last_values[lag] = history_df['wind_output'].iloc[-1] if len(history_df) > 0 else 0
                
        # 为第一个未来时间点设置滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'wind_output_lag_{lag}'] = np.nan
            if len(df) > 0:
                df[f'wind_output_lag_{lag}'].iloc[0] = last_values[lag]
        
        # 递归填充后续时间点的滞后特征
        for i in range(1, len(df)):
            for lag in [1, 2, 3, 6, 12, 24]:
                if i >= lag:
                    # 使用先前预测的值作为滞后特征
                    df[f'wind_output_lag_{lag}'].iloc[i] = df[f'wind_output_lag_{lag-(lag-1)}'].iloc[i-1]
                else:
                    lag_idx = lag - i
                    if lag_idx <= len(history_df):
                        df[f'wind_output_lag_{lag}'].iloc[i] = history_df['wind_output'].iloc[-lag_idx]
                    else:
                        df[f'wind_output_lag_{lag}'].iloc[i] = last_values[1]  # 使用最近的历史值
    
    return df


def plot_predictions(actual, pred, timestamps, title, save_path=None, rmse=None, mae=None, r2=None):
    """
    绘制预测结果与实际值的对比图
    
    参数:
        actual: 实际值
        pred: 预测值
        timestamps: 时间戳
        title: 图表标题
        save_path: 保存路径
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
    """
    plt.figure(figsize=(15, 7))
    plt.plot(timestamps, actual, label='实际值', alpha=0.7)
    plt.plot(timestamps, pred, label='预测值', alpha=0.7)
    
    if rmse is not None and mae is not None and r2 is not None:
        plt.title(f'{title}\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    else:
        plt.title(title)
    
    plt.xlabel('时间')
    plt.ylabel('风力发电输出 (MW)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"预测结果图表已保存至: {save_path}")
    
    return plt.gcf()


def create_datasets(df, seq_len, label_len, pred_len, target_col='wind_output', scale=True):
    """
    为Informer模型创建数据集，并对不同类型的特征使用不同的归一化方法
    
    参数:
        df: 输入数据DataFrame
        seq_len: 输入序列长度
        label_len: 标签序列长度
        pred_len: 预测序列长度
        target_col: 目标列名
        scale: 是否进行缩放
        
    返回:
        输入特征、标签和对应的缩放器字典
    """
    # 提取特征和标签列
    cols_data = df.columns.tolist()
    cols_data.remove(target_col) if target_col in cols_data else None
    cols_data.remove('time_idx') if 'time_idx' in cols_data else None
    cols_data.remove('group_id') if 'group_id' in cols_data else None
    
    # 特征和目标变量
    data_x = df[cols_data]
    data_y = df[[target_col]]
    
    if scale:
        # 根据特征类型对特征进行分组
        # 温度、湿度、压力等气象特征使用StandardScaler
        weather_cols = [col for col in cols_data if col in ['temp', 'humidity', 'pressure']]
        
        # 风速、风向等可能包含异常值的特征使用RobustScaler
        wind_cols = [col for col in cols_data if 'wind' in col.lower()]
        
        # 周期性特征如小时、日期使用MinMaxScaler
        cyclical_cols = [col for col in cols_data if col in ['hour', 'day', 'month', 'dayofweek', 'is_daytime']]
        
        # 二值特征无需缩放
        binary_cols = [col for col in cols_data if col in ['is_winter', 'season']]
        
        # 滞后特征使用与目标变量相同的缩放器
        lag_cols = [col for col in cols_data if 'lag' in col.lower()]
        
        # 剩余的特征使用StandardScaler
        remaining_cols = [col for col in cols_data if col not in weather_cols + wind_cols + cyclical_cols + binary_cols + lag_cols]
        
        # 创建各种缩放器
        weather_scaler = StandardScaler()
        wind_scaler = RobustScaler()
        cyclical_scaler = MinMaxScaler()
        target_scaler = StandardScaler()
        remaining_scaler = StandardScaler()
        
        # 创建一个空的DataFrame用于存储缩放后的特征
        scaled_data_x = pd.DataFrame(index=data_x.index)
        
        # 对各种特征进行缩放
        if weather_cols:
            weather_data = data_x[weather_cols]
            scaled_weather = weather_scaler.fit_transform(weather_data)
            scaled_data_x[weather_cols] = scaled_weather
        
        if wind_cols:
            wind_data = data_x[wind_cols]
            scaled_wind = wind_scaler.fit_transform(wind_data)
            scaled_data_x[wind_cols] = scaled_wind
        
        if cyclical_cols:
            cyclical_data = data_x[cyclical_cols]
            scaled_cyclical = cyclical_scaler.fit_transform(cyclical_data)
            scaled_data_x[cyclical_cols] = scaled_cyclical
        
        if binary_cols:
            # 二值特征不需要缩放
            scaled_data_x[binary_cols] = data_x[binary_cols]
        
        if remaining_cols:
            remaining_data = data_x[remaining_cols]
            scaled_remaining = remaining_scaler.fit_transform(remaining_data)
            scaled_data_x[remaining_cols] = scaled_remaining
        
        # 对目标变量进行缩放
        scaled_data_y = target_scaler.fit_transform(data_y)
        
        # 对滞后特征使用与目标变量相同的缩放器
        if lag_cols:
            lag_data = data_x[lag_cols]
            scaled_lag = target_scaler.transform(lag_data)
            scaled_data_x[lag_cols] = scaled_lag
        
        # 收集所有缩放器
        scalers = {
            'weather_scaler': weather_scaler,
            'wind_scaler': wind_scaler, 
            'cyclical_scaler': cyclical_scaler,
            'target_scaler': target_scaler,
            'remaining_scaler': remaining_scaler,
            'weather_cols': weather_cols,
            'wind_cols': wind_cols,
            'cyclical_cols': cyclical_cols,
            'binary_cols': binary_cols,
            'lag_cols': lag_cols,
            'remaining_cols': remaining_cols
        }
        
        # 转换为numpy数组
        data_x_scaled = scaled_data_x.values
        data_y_scaled = scaled_data_y
    else:
        data_x_scaled = data_x.values
        data_y_scaled = data_y.values
        scalers = None
    
    # 准备训练数据
    x_windows = []
    y_windows = []
    timestamps = []
    
    # 滑动窗口创建序列
    for i in range(len(df) - seq_len - pred_len + 1):
        x_window = data_x_scaled[i:i+seq_len]
        y_window = data_y_scaled[i:i+seq_len+pred_len]
        x_windows.append(x_window)
        y_windows.append(y_window)
        timestamps.append(df.index[i:i+seq_len+pred_len])
    
    # 转换为PyTorch可用的numpy数组
    x_windows = np.array(x_windows)
    y_windows = np.array(y_windows)
    
    # 重塑为 [batch, seq_len, feature]
    x_windows = x_windows.reshape(-1, seq_len, data_x.shape[1])
    y_windows = y_windows.reshape(-1, seq_len+pred_len, 1)
    
    logger.info(f"创建了 {len(x_windows)} 个训练样本，输入形状: {x_windows.shape}, 标签形状: {y_windows.shape}")
    
    return x_windows, y_windows, timestamps, scalers


def apply_feature_scaling(data, scalers):
    """
    使用已保存的缩放器对数据进行缩放
    
    参数:
        data: 输入数据DataFrame
        scalers: 缩放器字典
        
    返回:
        scaled_data: 缩放后的数据numpy数组
    """
    # 创建一个空的DataFrame用于存储缩放后的特征
    scaled_data = pd.DataFrame(index=data.index)
    
    # 对各种特征进行缩放
    if scalers['weather_cols']:
        weather_data = data[scalers['weather_cols']]
        scaled_weather = scalers['weather_scaler'].transform(weather_data)
        scaled_data[scalers['weather_cols']] = scaled_weather
    
    if scalers['wind_cols']:
        wind_data = data[scalers['wind_cols']]
        scaled_wind = scalers['wind_scaler'].transform(wind_data)
        scaled_data[scalers['wind_cols']] = scaled_wind
    
    if scalers['cyclical_cols']:
        cyclical_data = data[scalers['cyclical_cols']]
        scaled_cyclical = scalers['cyclical_scaler'].transform(cyclical_data)
        scaled_data[scalers['cyclical_cols']] = scaled_cyclical
    
    if scalers['binary_cols']:
        # 二值特征不需要缩放
        scaled_data[scalers['binary_cols']] = data[scalers['binary_cols']]
    
    if scalers['remaining_cols']:
        remaining_data = data[scalers['remaining_cols']]
        scaled_remaining = scalers['remaining_scaler'].transform(remaining_data)
        scaled_data[scalers['remaining_cols']] = scaled_remaining
    
    if scalers['lag_cols']:
        lag_data = data[scalers['lag_cols']]
        scaled_lag = scalers['target_scaler'].transform(lag_data)
        scaled_data[scalers['lag_cols']] = scaled_lag
    
    return scaled_data.values 