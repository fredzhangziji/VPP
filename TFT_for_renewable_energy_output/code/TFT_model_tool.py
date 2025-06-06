"""
此模块定义了风电出力TFT（Temporal Fusion Transformer）模型所需的所有功能函数。

主要功能包括：
- 从数据库获取历史和预测天气数据
- 获取历史风电和光伏出力数据
- 数据预处理、标准化和特征工程
- TFT模型的构建、训练和优化
- 模型预测和评估

模块可被训练和预测脚本按需调用。
"""

from pub_tools import db_tools, const
from pub_tools.pub_tools import get_system_font_path
from sqlalchemy import Table, select
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import os
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import matplotlib.font_manager as fm
import optuna

import logging
import pub_tools.logging_config
logger = logging.getLogger('tft_model')

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
    '鄂尔多斯+薛家湾': '鄂尔多斯市'
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

    # query = select(terraqt_weather.c.ts, terraqt_weather.c.ws10m).where(
    #     (terraqt_weather.c.model == 'era5_land') &
    #     (terraqt_weather.c.region_name == city) 
    # )

    # ws10m_df = db_tools.read_from_db(engine, query)
    # ws10m_df['datetime'] = pd.to_datetime(ws10m_df['ts'])
    # ws10m_df.set_index('datetime', inplace=True)
    # ws10m_df.drop(columns=['ts'], inplace=True)
    # ws10m_df.sort_index(inplace=True)

    # weather_df['ws10m'] = ws10m_df['ws10m']

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
    
    if 'wind_output' not in output_df.columns or output_df['wind_output'].isnull().all():
        logger.warning("[TFTModelTool.get_history_wind_power_for_city] Warning: No valid 'wind_output' data found for city '%s' after initial load. The column might be missing or all values are NULL.", city)
        # Return an empty DataFrame with expected index type if possible, or just empty
        return pd.DataFrame(index=pd.to_datetime([]), columns=['wind_output'])
        
    output_df['wind_output'] = output_df['wind_output'].resample('h', closed='right', label='right').mean()
    
    # Check after resampling if all data became NaN (e.g., if original data was sparse and didn't align with hourly resampling)
    if output_df['wind_output'].isnull().all():
        logger.warning("[TFTModelTool.get_history_wind_power_for_city] Warning: All 'wind_output' data became NaN after resampling for city '%s'.", city)
        output_df.dropna(inplace=True) # This will make the DataFrame empty
        return output_df # Should be an empty DataFrame now
        
    output_df.dropna(inplace=True) # Drop rows where wind_output might be NaN after resampling

    if output_df.empty:
        logger.warning("[TFTModelTool.get_history_wind_power_for_city] Warning: 'wind_output' data is empty for city '%s' after processing (resample and dropna).", city)

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
    """将天气和出力数据合并"""
    merged_df = pd.merge(weather_df, power_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.dropna()

    return merged_df

def _wind_season(m):
    return 'big' if (1 <= m <= 5 or 9 <= m <= 12) else 'small'

def _add_more_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加更多特征"""
    df['pressure_trend'] = df['sp'].diff().fillna(0)

    # absolute and relative gustiness
    df['gustiness_abs_100m'] = (df['gust'] - df['ws100m']).fillna(0)
    # df['gustiness_abs_10m'] = (df['gust'] - df['ws10m']).fillna(0)
    df['gustiness_rel_100m'] = df['gustiness_abs_100m'] / df['ws100m']
    # df['gustiness_rel_10m']  = df['gustiness_abs_10m']  / df['ws10m']
    # linear wind shear
    # df['wind_shear_linear'] = (df['ws100m'] - df['ws10m']) / 90.0
    # power‐law exponent shear
    # df['wind_shear_exponent'] = np.log(df['ws100m'] / df['ws10m']) / np.log(100.0 / 10.0)
    # df[['wind_shear_linear','wind_shear_exponent']] = \
    #     df[['wind_shear_linear','wind_shear_exponent']].fillna(0)

    # Wind season based on index month
    df['wind_season'] = df.index.month.map(_wind_season)
    if 'wind_output' in df.columns:
        df['lag_30d']  = df['wind_output'].shift(30*24)
        df['lag_7d']   = df['wind_output'].shift(7*24)
    df.dropna(inplace=True)
    return df

def _deduplicate_data(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """数据去重"""
    dup_mask = df.duplicated(subset=subset, keep=False)
    if dup_mask.any():
        logger.info('Duplicate rows found:\n%s', df[dup_mask])
        logger.info('Total %d duplicate rows found', len(df[dup_mask]))
        df = df.drop_duplicates(subset=subset, keep='first')
        logger.info('Deduplication complete; kept first occurrence of each duplicate row')
    else:
        logger.info('No duplicate rows found')

    # Drop duplicate timestamps on index
    index_dups = df.index[df.index.duplicated(keep=False)]
    if len(index_dups) > 0:
        logger.info("Duplicate timestamps found by index:")
        logger.info(pd.Series(index_dups).value_counts())
        # Keep first occurrence of each timestamp
        df = df[~df.index.duplicated(keep='first')]
        logger.info("Dropped %d duplicate timestamp entries; kept first occurrence.", 
                   len(index_dups) - len(df.index[df.index.duplicated(keep=False)]))

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
    # Initial check for empty DataFrame
    if df.empty:
        logger.warning("Input DataFrame is empty. Returning as is.")
        return df

    # Check if the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Input DataFrame index is not a DatetimeIndex (type: %s). Returning as is.", type(df.index))
        return df

    # Try to get min and max of the index
    try:
        start_time = df.index.min()
        end_time = df.index.max()
    except Exception as e:
        # Catch any exception during min/max, e.g., if index contains non-comparable types
        logger.warning("Could not get min/max of DataFrame index. Error: %s. Returning as is.", e)
        return df

    # Check if min or max times are NaT (Not a Time)
    if pd.isna(start_time) or pd.isna(end_time):
        logger.warning("DataFrame index min ('%s') or max ('%s') is NaT. Returning as is.", start_time, end_time)
        return df

    # Proceed with creating complete_time_range if all checks pass
    complete_time_range = pd.date_range(
        start=start_time,
        end=end_time,
        freq='h'
    )
    initial_missing = complete_time_range.difference(df.index)
    logger.info("Initial missing timestamps: %d", len(initial_missing))
    if len(initial_missing) == 0:
        logger.info("No need to deal missing data.")
        return df

    logger.info("Dealing with missing data.")
    logger.info("Initial missing dates:\n%s", pd.Series(initial_missing.date).value_counts())

    df_reindexed = df.reindex(complete_time_range)
    # mark originally missing timestamps before interpolation
    df_reindexed['missing_original'] = df_reindexed.isna().any(axis=1).astype(int).astype(str)
    df_reindexed['date'] = df_reindexed.index.date
    missing_flag = df_reindexed.drop(columns=['date', 'missing_original']).isna().any(axis=1)
    missing_count_per_day = missing_flag.groupby(df_reindexed['date']).sum()
    logger.info("Missing timestamps per day:\n%s", missing_count_per_day)

    # calculate maximum allowed missing hours per day (e.g., 10% of 24h = 2.4 -> 2 hours)
    threshold_hours = max(1, int(threshold_percent * 24))
    logger.info("Using threshold_percent=%.2f, threshold_hours=%d", threshold_percent, threshold_hours)
    # drop any day with missing count above threshold_hours
    days_to_drop = missing_count_per_day[missing_count_per_day > threshold_hours].index
    logger.info("Days to drop: %d", len(days_to_drop))

    df_cleaned = df_reindexed[~df_reindexed['date'].isin(days_to_drop)].copy()
    df_cleaned.drop(columns='date', inplace=True)
    # Optionally, keep or drop 'missing_original' before returning.
    # df_cleaned.drop(columns='missing_original', inplace=True)
    df_filled = df_cleaned.interpolate(method='time')

    expected_range = pd.date_range(
        start=df_filled.index.min(), # This could also fail if df_filled becomes empty
        end=df_filled.index.max(),
        freq='h'
    )
    missing_timestamps = expected_range.difference(df_filled.index)
    logger.info("Total missing timestamps: %d", len(missing_timestamps))
    logger.info("Missing timestamps by date:\n%s", 
                pd.Series(missing_timestamps.date).value_counts().sort_index())
    
    return df_filled

def _normalize_data(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """数据归一化.

    如果已有归一化模型, 则直接加载模型并对于不同类型的特征进行不同的归一化处理;
    如果没有归一化模型, 则对于不同类型的特征进行不同的归一化处理并且输出归一化模型.

    注意, 因为TFT自带目标列的归一化功能, 所以在此不对目标列进行归一化处理
    
    入参:
        df: 待归一化的数据
        city: 具体的城市

    出参:
        归一化模型

    返回:
        df_normalized: 预处理完成的数据
    """
    model_dir = '../norm_model'
    model_path = os.path.join(model_dir, city + '_scalers.pkl')

    logger.info("Columns in dataframe: %s", df.columns)
    
    df_normalized = df.copy()
    scalers = {}
    columns_to_scale = [col for col in df.columns if col != 'city_name' and col != 'wind_output' and col != 'missing_original' and col != 'wind_season']

    # If normalization model file exists, load scalers and apply to features
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            scalers = pickle.load(f)
        logger.info("Loaded existing normalization models.")
        for col in columns_to_scale:
            if col in scalers:
                scaler = scalers[col]
                df_normalized[col] = scaler.transform(df_normalized[[col]])
            else:
                scaler = StandardScaler()
                df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
                scalers[col] = scaler
    # If normalization models not found, fit new scalers and save them
    else:
        for col in columns_to_scale:
            if 'output' in col:
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            df_normalized[col] = scaler.fit_transform(df_normalized[[col]])
            scalers[col] = scaler
        
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        with open(model_path, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info("Normalization models are saved.")

    return df_normalized


def preprocess_data(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """数据预处理.

    包括数据去重/处理缺失值/归一化.
    
    入参:
        df: 待处理的数据
        city: 数据的具体城市

    出参:
        归一化模型

    返回:
        df_cleaned: 预处理完成的数据
    """
    df_deduplicated = _deduplicate_data(df)
    df_filled = _deal_missing_data(df_deduplicated)
    df_more_features = _add_more_features(df_filled)
    df_cleaned = _normalize_data(df_more_features, city)
    df_cleaned['datetime'] = df_cleaned.index

    return df_cleaned

def is_dataset_valid(df: pd.DataFrame) -> bool:
    """验证数据集的完整性"""
    logger.info('Data validation:')
    logger.info('Total rows: %d', len(df))
    logger.info('Missing values per column:')
    logger.info(df.isnull().sum())
    logger.info('\nValue ranges:')
    logger.info(df.describe())
    return not df.isnull().any().any()

def set_time_wise_feature(df: pd.DataFrame) -> pd.DataFrame:
    """为数据集进行时间特征处理以为TFT使用"""
    df['time_idx'] = np.arange(1, len(df) + 1)
    # Only one time series: assign a single group
    df['group_id'] = "0"
    # Extract month as categorical feature
    df['month'] = df['datetime'].dt.month.astype(str)
    df.drop(columns=['datetime'], inplace=True)

    return df

def build_tft_datasets(normalized_df: pd.DataFrame,
                       training_cutoff: int,
                       encoder_length: int,
                       prediction_length: int,
                       target_col: str = "wind_output"):
    """构建TFT的训练和验证数据集

    入参:
        normalized_df: 已归一化且包含时间特征、天气特征的DataFrame，
                       要求包含 'time_idx'、'group_id'等辅助列
        training_cutoff: 分割训练和验证集的时间索引阈值（例如：normalized_df["time_idx"].max() - prediction_length）
        encoder_length: 编码器长度（历史输入长度）
        prediction_length: 预测长度，即未来时间步数
        target_col: 目标列名称，默认为"wind_output"

    返回:
        training_dataset: 训练数据集
        validation_dataset: 验证数据集（用于预测时构建）
    """
    training_dataset = TimeSeriesDataSet(
        normalized_df[normalized_df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["group_id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=prediction_length,
        max_prediction_length=prediction_length,
        static_categoricals=[],
        time_varying_known_categoricals=["missing_original", "month", "wind_season"],
        # Time-varying known real features (e.g., weather variables)
        time_varying_known_reals=["time_idx", "t2m", "ws100m", "wd100m", "gust", "sp", "pressure_trend", "gustiness_abs_100m", "gustiness_rel_100m", "lag_30d", "lag_7d"],
        # Time-varying unknown real features (target)
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_encoder_length=True,
        add_target_scales=True,
        allow_missing_timesteps=True
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        normalized_df,
        predict=True,
        stop_randomization=True,
    )
    return training_dataset, validation_dataset

def create_dataloaders(training_dataset: TimeSeriesDataSet,
                       validation_dataset: TimeSeriesDataSet,
                       batch_size: int = 64):
    """根据构造好的数据集创建数据加载器

    入参:
        training_dataset: 训练数据集
        validation_dataset: 验证数据集
        batch_size: 批量大小，默认值为64

    返回:
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
    """
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=32, pin_memory=True, persistent_workers=True
    )
    # validation DataLoader: shuffle=False to keep deterministic ordering
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size, num_workers=32, shuffle=False
    )
    return train_dataloader, val_dataloader

def train_tft_model(training_dataset: TimeSeriesDataSet,
                    train_dataloader: DataLoader,
                    val_dataloader: DataLoader,
                    learning_rate: float = 0.01,
                    hidden_size: int = 32,
                    attention_head_size: int = 2,
                    dropout: float = 0.2,
                    hidden_continuous_size: int = 16,
                    max_epochs: int = 30,
                    callbacks: list = None):
    """训练TFT模型

    参数:
        training_dataset: 用于构造模型的训练数据集
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        learning_rate: 学习率
        hidden_size: 隐藏层维度
        attention_head_size: 注意力头数
        dropout: dropout比率
        hidden_continuous_size: 连续变量隐藏层维度
        max_epochs: 最大训练轮数
        callbacks: 回调函数列表

    返回:
        tft: 训练好的TemporalFusionTransformer模型
        trainer: 用于训练的Trainer对象
    """
    # Print current training hyperparameters
    logger.info("Training TFT with hyperparameters: learning_rate=%f, hidden_size=%d, "
                "attention_head_size=%d, dropout=%f, hidden_continuous_size=%d",
                learning_rate, hidden_size, attention_head_size, dropout, hidden_continuous_size)
    # # Use custom HuberLossMetric instead of QuantileLoss
    # loss = HuberLossMetric(delta=1.0)
    # output_size = 1  # HuberLossMetric outputs a single value
    loss = QuantileLoss()
    output_size = len(loss.quantiles)
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=loss,
        output_size=output_size,
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=True, mode="min"
    )
    lr_logger = LearningRateMonitor()
    # Build hyperparameter string for checkpoint filenames
    params_str = (
        f"lr{learning_rate:.4f}"
        f"-hs{hidden_size}"
        f"-ah{attention_head_size}"
        f"-do{dropout}"
        f"-hcs{hidden_continuous_size}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename=f"tft-{params_str}-" + "{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    # Use provided callbacks if available, else default to early_stop_callback
    final_callbacks = callbacks if callbacks is not None else [early_stop_callback]
    if not any(isinstance(cb, EarlyStopping) for cb in final_callbacks) and not callbacks: # Add default if no callbacks provided at all
        final_callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # Specify the number of GPUs or CPU cores
        gradient_clip_val=0.1,
        enable_progress_bar=True, # Keep progress bar for interactive training
        enable_model_summary=True
    )
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return tft, trainer


# ----------------- Optuna objective function -----------------
def objective(trial, city, merged_df):
    # Suggest hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 16, 256, step=16)
    attention_head_size = trial.suggest_int("attention_head_size", 1, 4)
    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    hidden_continuous_size = trial.suggest_int("hidden_continuous_size", 8, 32, step=8)
    max_epochs = trial.suggest_int("max_epochs", 10, 50, step=10)

    # Initialize trainer with trial suggestions
    trainer_obj = TFTModelTrainer(
        city=city,
        learning_rate=lr,
        hidden_size=hidden_size,
        head_size=attention_head_size,
        drop_out=dropout,
        hidden_continuous_size=hidden_continuous_size,
        max_epochs=max_epochs
    )

    # Preprocess and setup
    df_preprocessed = trainer_obj.preprocess_and_set_features(merged_df)
    trainer_obj.build_datasets(df_preprocessed)
    trainer_obj.create_loaders()

    # Train
    tft_model, trainer = trainer_obj.train()

    # Evaluate on validation
    val_metrics = trainer.validate(tft_model, dataloaders=trainer_obj.val_dl, verbose=False)[0]
    return val_metrics["val_loss"]

def predict_tft(tft: TemporalFusionTransformer, dataloader: DataLoader):
    """使用训练好的TFT模型进行预测

    入参:
        tft: 训练好的TemporalFusionTransformer模型
        dataloader: 用于预测的数据加载器

    返回:
        raw_predictions: 模型预测的原始结果
        x: 对应的输入数据
    """
    # unpack predictions; handle extra outputs gracefully
    preds = tft.predict(
        dataloader,
        mode="raw",
        return_x=True,
        trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "benchmark": False
        }
    )
    # preds may include additional tensors after raw_predictions and x
    raw_predictions, x, *_ = preds
    return raw_predictions, x

class TFTModelTrainer:
    """封装TFT模型相关功能，包括数据预处理、数据集构建、模型训练和预测。"""
    def __init__(self, city: str, target_col: str = "wind_output", prediction_length: int = 24*5,
                 encoder_length: int = 24*3, batch_size: int = 256, max_epochs: int = 50,
                 learning_rate: float = 0.001, hidden_size: int = 32, head_size: int = 2,
                 drop_out: float = 0.2, hidden_continuous_size: int = 16):
        self.city = city
        self.target_col = target_col
        self.prediction_length = prediction_length
        self.encoder_length = encoder_length
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.drop_out = drop_out
        self.hidden_continuous_size = hidden_continuous_size
        self.tft_model = None
        self.trainer = None
        self.df_cleaned = None
        
    def preprocess_and_set_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """对输入DataFrame进行预处理和时间特征设置"""
        df_cleaned = preprocess_data(df, self.city)
        df_cleaned = set_time_wise_feature(df_cleaned)
        # if not is_dataset_valid(df_cleaned):
        #     raise ValueError("数据预处理结果不合法，请检查数据！")
        self.df_cleaned = df_cleaned
        return df_cleaned

    def build_datasets(self, df: pd.DataFrame):
        """使用预处理后的数据构建TFT的数据集"""
        training_cutoff = df["time_idx"].max() - self.prediction_length
        self.training_dataset, self.validation_dataset = build_tft_datasets(
            df,
            training_cutoff,
            self.encoder_length,
            self.prediction_length,
            target_col=self.target_col
        )
        return self.training_dataset, self.validation_dataset

    def create_loaders(self):
        """根据构建好的数据集创建数据加载器"""
        self.train_dl, self.val_dl = create_dataloaders(self.training_dataset,
                                                         self.validation_dataset,
                                                         batch_size=self.batch_size)
        return self.train_dl, self.val_dl

    def train(self, callbacks: list = None):
        """训练TFT模型"""
        self.tft_model, self.trainer = train_tft_model(
            self.training_dataset,
            self.train_dl,
            self.val_dl,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            dropout=self.drop_out,
            hidden_size=self.hidden_size,
            attention_head_size=self.head_size,
            hidden_continuous_size=self.hidden_continuous_size,
            callbacks=callbacks
        )
        return self.tft_model, self.trainer

    def predict_validation(self):
        """在验证集上预测，反归一化，并绘制预测 vs 实际曲线，保存图像。"""
        if self.tft_model is None:
            raise ValueError("请先训练模型！")
        # Generate model predictions with deterministic settings
        raw_preds, x = predict_tft(self.tft_model, self.val_dl)
        # Extract median quantile predictions (median at index len(quantiles)//2)
        median_idx = raw_preds.prediction.shape[-1] // 2
        preds = raw_preds.prediction[..., median_idx].detach().cpu().numpy().flatten()
        # Build original validation index
        df_val = self.df_cleaned[self.df_cleaned["time_idx"] >
                                 (self.df_cleaned["time_idx"].max() - self.prediction_length)]
        # Predictions are already denormalized internally by TFT
        pred_denorm = pd.Series(preds, index=df_val.index)
        actual_denorm = df_val[self.target_col]
        # Plot validation results
        # Configure system font for plot
        font_path = get_system_font_path()
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 6), dpi=150)
        pred_denorm.plot(label='Prediction')
        actual_denorm.plot(label='Actual')
        plt.legend()
        plt.xlabel('datetime')
        plt.ylabel(self.target_col)
        # Display hyperparameters in plot title
        params_text = (f"lr={self.learning_rate}, hs={self.hidden_size}, ah={self.head_size}, "
                       f"do={self.drop_out}, hcs={self.hidden_continuous_size}")
        # Retrieve validation metrics and prepare annotation text
        val_metrics = self.trainer.validate(
            self.tft_model,
            dataloaders=self.val_dl,
            verbose=False
        )[0]
        metrics_text = (
            f"loss={val_metrics['val_loss']:.3f}, "
            f"SMAPE={val_metrics['val_SMAPE']:.3f}, "
            f"MAE={val_metrics['val_MAE']:.3f}, "
            f"RMSE={val_metrics['val_RMSE']:.3f}, "
            f"MAPE={val_metrics['val_MAPE']:.3f}, "
            f"R2={val_metrics.get('val_R2Score', float('nan')):.3f}, "
            f"MSLE={val_metrics.get('val_MSLE', float('nan')):.3f}"
        )
        plt.title(
            f'{self.city} Validation Prediction\n'
            f'{params_text}\n'
            f'{metrics_text}'
        )
        plt.tight_layout()
        plt.savefig(f"../figures/{self.city}_validation_prediction.png")
        plt.close()
        return pred_denorm, actual_denorm

    def predict_future(self, future_steps: int):
        """在未来 future_steps 步进行预测，反归一化，并绘制未来预测曲线，保存图像。"""
        if self.tft_model is None:
            raise ValueError("请先训练模型！")
        # Construct dataset for future steps
        last_idx = self.df_cleaned["time_idx"].max()
        logger.info('**********************************************')
        logger.info(f"Last time index: {last_idx}")
        logger.info('**********************************************')

        # Use the same build_tft_datasets to construct only the future dataset
        _, future_dataset = build_tft_datasets(
            self.df_cleaned,
            last_idx,
            self.encoder_length,
            future_steps,
            target_col=self.target_col
        )
        future_loader = future_dataset.to_dataloader(
            train=False, 
            batch_size=self.batch_size,
            num_workers=32,
            pin_memory=True
        )

        raw_preds, _ = predict_tft(self.tft_model, future_loader)
        median_idx = raw_preds.prediction.shape[-1] // 2
        preds = raw_preds.prediction[..., median_idx].detach().cpu().numpy().flatten()
        # Construct future time index
        last_time = self.df_cleaned.index[-1]
        future_index = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=future_steps,
            freq="h"
        )
        # Predictions have been denormalized internally by TFT
        pred_denorm = pd.Series(preds, index=future_index)
        # Plot future forecast
        # Configure system font for plot
        font_path = get_system_font_path()
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 6), dpi=150)
        pred_denorm.plot(label=f'{future_steps}-step Forecast')
        plt.legend()
        plt.xlabel('datetime')
        plt.ylabel(self.target_col)
        plt.title(f'{self.city} {future_steps}-Step Forecast')
        plt.tight_layout()
        plt.savefig(f"../figures/{self.city}_future_prediction_{future_steps}.png")
        plt.close()
        return pred_denorm

if __name__ == '__main__':
    # Load and merge data
    weather_df = get_history_weather_data_for_city(CITIES_FOR_WEATHER[0])
    output_df = get_history_wind_power_for_city(CITIES_FOR_POWER[0])
    merged_df = merge_weather_and_power_df(weather_df, output_df)

    # ----------------- Optuna hyperparameter tuning -----------------
    # Create the Optuna study
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    # Run optimization
    study.optimize(objective, n_trials=30, timeout=3600)

    # Print and save best parameters
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_value}")
    logger.info("  Params: ")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")

    # Train final model using best parameters
    best_params = study.best_params
    final_trainer = TFTModelTrainer(
        city=CITIES_FOR_POWER[0],
        learning_rate=best_params["learning_rate"],
        hidden_size=best_params["hidden_size"],
        head_size=best_params["attention_head_size"],
        drop_out=best_params["dropout"],
        hidden_continuous_size=best_params["hidden_continuous_size"],
        max_epochs=best_params["max_epochs"]
    )
    df_preprocessed = final_trainer.preprocess_and_set_features(merged_df)
    logger.info('final dataset:')
    logger.info(df_preprocessed)
    final_trainer.build_datasets(df_preprocessed)
    final_trainer.create_loaders()
    final_model, _ = final_trainer.train()
    final_trainer.tft_model = final_model

    # Predict on validation and future
    pred_val, actual_val = final_trainer.predict_validation()
    pred_future = final_trainer.predict_future(24 * 5)