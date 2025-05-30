"""
此模块定义了风电出力Baseline所有所需的子功能函数，例如
- normalize_data(merged_df): 标准化数据
- unnormalize_data(normalized_df): 反标准化数据
- predict_province_future(cities, future_features_dict, start_date, n_days): 实际预测未来n天的风电出力
等等

按需被调用。
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json
import matplotlib.pyplot as plt 
import pandas as pd
import optuna
from pub_tools import db_tools, const, pub_tools
from sqlalchemy import Table, select, insert
from datetime import datetime, timedelta, timezone
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 计算斜率
def _compute_daily_metrics(df: pd.DataFrame):
    features = ['t2m', 'ws100m', 'wd100m_sin', 'wd100m_cos', 'sp']
    weather_ts = df[features].values  # shape (T, len(features))
    wind_series = df['wind_output'].values
    return weather_ts, wind_series

def _compute_similarity(day1, day2, params, daily_metrics):
    # 获取天气时间序列
    weather1 = daily_metrics[day1]['weather_ts']
    weather2 = daily_metrics[day2]['weather_ts']
    # 截取最小长度，确保数组形状一致
    min_length = min(weather1.shape[0], weather2.shape[0])
    weather1 = weather1[:min_length, :]
    weather2 = weather2[:min_length, :]
    weather_dist = np.sqrt(np.sum((weather1 - weather2)**2, axis=0))
    sim_feature = - np.dot(params['w_feature'], weather_dist)
    
    season_diff = np.sqrt(
        (daily_metrics[day1]['season_sin'] - daily_metrics[day2]['season_sin'])**2 +
        (daily_metrics[day1]['season_cos'] - daily_metrics[day2]['season_cos'])**2
    )
    sim_season = - params['w_season'] * season_diff
    
    total_similarity = sim_feature + sim_season
    return total_similarity

def _objective(trial, daily_dates, daily_metrics):
    w_feature = [
        trial.suggest_float('w_feature_0', 0, 1),
        trial.suggest_float('w_feature_1', 0, 1),
        trial.suggest_float('w_feature_2', 0, 1),
        trial.suggest_float('w_feature_3', 0, 1),
        trial.suggest_float('w_feature_4', 0, 1)
    ]

    w_season = trial.suggest_float('w_season', 0, 1)

    params = {
        'w_feature': np.array(w_feature),
        'w_season': w_season
    }
    
    errors = []
    for val_day in daily_dates:
        similarities = {}
        for candidate_day in daily_dates:
            if candidate_day == val_day:
                continue
            sim = _compute_similarity(val_day, candidate_day, params, daily_metrics)
            similarities[candidate_day] = sim
        if len(similarities) == 0:
            continue
        best_day = max(similarities, key=similarities.get)
        forecast = daily_metrics[best_day]['wind_series']
        actual = daily_metrics[val_day]['wind_series']
        if len(forecast) != len(actual):
            continue
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        errors.append(rmse)
    if len(errors) == 0:
        return 0.0
    return np.mean(errors)

def _get_similar_day(target_day, params, daily_dates, daily_metrics):
    similarities = {}
    for candidate_day in daily_dates:
        if candidate_day == target_day:
            continue
        sim = _compute_similarity(target_day, candidate_day, params, daily_metrics)
        similarities[candidate_day] = sim
    best_day = max(similarities, key=similarities.get)
    return best_day

def normalize_data(merged_df):
    model_file = '../output/wind_data_normalization_models.json'
    
    # 如果已存在归一化模型，则加载参数，并按模型归一化
    if os.path.exists(model_file):
        with open(model_file, 'r') as f:
            norm_models = json.load(f)
        print("加载已有归一化模型...")
        # 温度 t2m （z-score 归一化）
        mean_t2m = norm_models['t2m']['mean']
        std_t2m = norm_models['t2m']['std']
        merged_df['t2m'] = merged_df['t2m'].apply(lambda x: (x - mean_t2m) / std_t2m)
        
        # 风速 ws100m （z-score 归一化）
        mean_ws100m = norm_models['ws100m']['mean']
        std_ws100m = norm_models['ws100m']['std']
        merged_df['ws100m'] = merged_df['ws100m'].apply(lambda x: (x - mean_ws100m) / std_ws100m)
        
        # 气压 sp （z-score 归一化）
        mean_sp = norm_models['sp']['mean']
        std_sp = norm_models['sp']['std']
        merged_df['sp'] = merged_df['sp'].apply(lambda x: (x - mean_sp) / std_sp)
        
        # 风向 wd100m: 采用正余弦转换
        merged_df['wd100m_sin'] = np.sin(merged_df['wd100m'] * np.pi / 180)
        merged_df['wd100m_cos'] = np.cos(merged_df['wd100m'] * np.pi / 180)
        merged_df.drop(columns=['wd100m'], inplace=True)
        
        # 风力发电输出 wind_output: Min-Max 归一化
        min_wind = norm_models['wind_output']['min']
        max_wind = norm_models['wind_output']['max']
        merged_df['wind_output'] = merged_df['wind_output'].apply(lambda x: (x - min_wind) / (max_wind - min_wind))
        
    # 如果模型文件不存在，则计算归一化参数，并应用归一化，同时保存参数到文件
    else:
        norm_models = {}

        # 温度 t2m: Z-score 标准化（原地修改）
        mean_t2m = merged_df['t2m'].mean()
        std_t2m = merged_df['t2m'].std()
        merged_df['t2m'] = merged_df['t2m'].apply(lambda x: (x - mean_t2m) / std_t2m)
        norm_models['t2m'] = {'method': 'z-score', 'mean': float(mean_t2m), 'std': float(std_t2m)}

        # 风速 ws100m: Z-score 标准化（原地修改）
        mean_ws100m = merged_df['ws100m'].mean()
        std_ws100m = merged_df['ws100m'].std()
        merged_df['ws100m'] = merged_df['ws100m'].apply(lambda x: (x - mean_ws100m) / std_ws100m)
        norm_models['ws100m'] = {'method': 'z-score', 'mean': float(mean_ws100m), 'std': float(std_ws100m)}

        # 气压 sp: Z-score 标准化（原地修改）
        mean_sp = merged_df['sp'].mean()
        std_sp = merged_df['sp'].std()
        merged_df['sp'] = merged_df['sp'].apply(lambda x: (x - mean_sp) / std_sp)
        norm_models['sp'] = {'method': 'z-score', 'mean': float(mean_sp), 'std': float(std_sp)}

        # 风向 wd100m: 采用正余弦转换
        # 扩展为两个新列，再删除原始 wd100m（后续在反归一化时恢复原始角度）
        merged_df['wd100m_sin'] = np.sin(merged_df['wd100m'] * np.pi/180)
        merged_df['wd100m_cos'] = np.cos(merged_df['wd100m'] * np.pi/180)
        merged_df.drop(columns=['wd100m'], inplace=True)
        norm_models['wd100m'] = {'method': 'sin-cos'}

        # 风力发电输出 wind_output: Min-Max 归一化（原地修改）
        min_wind = merged_df['wind_output'].min()
        max_wind = merged_df['wind_output'].max()
        merged_df['wind_output'] = merged_df['wind_output'].apply(lambda x: (x - min_wind) / (max_wind - min_wind))
        norm_models['wind_output'] = {'method': 'min-max', 'min': float(min_wind), 'max': float(max_wind)}

        # 保存归一化模型参数到 json 文件
        os.makedirs(os.path.dirname(model_file), exist_ok=True)
        with open(model_file, 'w') as f:
            json.dump(norm_models, f, indent=4)
        print("归一化模型已保存到", model_file)

    print("归一化预览（原地修改）：")
    print(merged_df.head(5))
    
    return merged_df

def unnormalize_data(normalized_df):
    # 从文件中加载归一化模型参数
    with open('../output/wind_data_normalization_models.json', 'r') as f:
        norm_models = json.load(f)

    # 反归一化风力发电输出 wind_output（Min-Max）
    min_wind = norm_models['wind_output']['min']
    max_wind = norm_models['wind_output']['max']
    normalized_df['wind_output'] = normalized_df['wind_output'].apply(
        lambda x: x * (max_wind - min_wind) + min_wind
    )

    return normalized_df
    
def save_weights(best_params, city):
    # 将 numpy 数组转换为 list（如果存在 numpy 数组）
    best_params_to_save = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in best_params.items()}

    with open('../output/wind_output_baseline_best_params_for_' + city + '.json', 'w') as f:
        json.dump(best_params_to_save, f, indent=4)

    print("best_params has been saved.")

def load_weights(city):
    with open('../output/wind_output_baseline_best_params_for_' + city + '.json', 'r') as f:
        loaded_params = json.load(f)

    best_params = {
        'w_feature': np.array(loaded_params['w_feature']),
        'w_season': loaded_params['w_season']
    }

    print(f"loaded best_params for {city}: {best_params}")
    return best_params

# 测试集：预测单城市
def predict_for_test_return(city: str, day_index: int):
    """
    对单个城市的指定测试日做预测，返回：
      actual_unnorm: 实际结果 (1D numpy 数组)
      forecast: 预测结果 (1D numpy 数组)
      metrics: 字典，包含 RMSE, MAE, MAPE
    """
    print(f'cur city: {city}')
    normalized_df = get_history_weather_data_for_city(city)
    print(normalized_df.head(1))
    daily_dates, daily_metrics = get_daily_data(normalized_df)
    print(f'daily_dates len: {len(daily_dates)}')
    # 加载该城市训练好的权重参数
    best_params = load_weights(city)
    target_day = daily_dates[day_index]
    similar_day = _get_similar_day(target_day, best_params, daily_dates, daily_metrics)
    print(f"[{city}] Target day: {target_day}; Most similar day: {similar_day}")
    
    # 使用最相似日的风电序列作为预测（归一化状态下）
    forecast_norm = daily_metrics[similar_day]['wind_series']
    forecast_df = pd.DataFrame({'wind_output': forecast_norm})
    forecast_df = unnormalize_data(forecast_df)
    forecast = forecast_df['wind_output'].values
    
    actual = daily_metrics[target_day]['wind_series']
    actual_df = pd.DataFrame({'wind_output': actual})
    actual_df = unnormalize_data(actual_df)
    actual_unnorm = actual_df['wind_output'].values
    
    # 确保长度一致
    min_len = min(len(actual_unnorm), len(forecast))
    actual_unnorm = actual_unnorm[:min_len]
    forecast = forecast[:min_len]
    
    rmse = np.sqrt(mean_squared_error(actual_unnorm, forecast))
    mae = mean_absolute_error(actual_unnorm, forecast)
    actual_safe = np.where(actual_unnorm==0, 1e-6, actual_unnorm)
    mape = np.mean(np.abs((actual_unnorm - forecast) / actual_safe)) * 100
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    # 设置字体
    try:
        font_list = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Zen Hei' in font_list:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        else:
            raise Exception("WenQuanYi Zen Hei font not found")
    except Exception as e:
        print("无法加载 WenQuanYi Zen Hei 字体，请检查该字体是否安装，错误信息：", e)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制单城市预测曲线并保存
    plt.figure(figsize=(10,6))
    plt.plot(actual_unnorm, label='Actual', marker='o')
    plt.plot(forecast, label='Forecast', marker='x', linestyle='--')
    plt.legend()
    plt.title(f"{city} {target_day} Forecast Results")
    plt.xlabel("Hour")
    plt.ylabel("Wind Power Output")
    plt.gca().text(0.05, 0.95, 
               f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAPE: {mape:.2f}%", 
               transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
    
    fig_folder = "../figure"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # 保存图像到 figure 文件夹中，并根据城市和日期命名图片文件
    save_path = os.path.join(fig_folder, f"{city.replace(' ', '_')}_{target_day}_test_Forecast.png")
    plt.savefig(save_path)
    plt.close()
    
    return actual_unnorm, forecast, metrics, target_day

# 测试集：每个城市预测
def test_single_city(cities, day_index= -1):
    # day_index 可以指定测试日期在 daily_dates 中的位置，如 -1 表示最后一天
    results = {}
    for city in cities:
        print(f"\n开始对城市 {city} 进行测试...")
        actual, forecast, metrics, target_day = predict_for_test_return(city, day_index)
        results[city] = {"actual": actual, "forecast": forecast, "metrics": metrics}
    return results, target_day

# 测试集：全省总和
def predict_province_day(cities, day_index=-1):
    province_actual = None
    province_forecast = None
    city_results, target_day = test_single_city(cities, day_index)
    for city in cities:
        r = city_results[city]
        # 若不同城市预测长度可能不一致，则取最短长度
        if province_actual is None:
            province_actual = r["actual"]
            province_forecast = r["forecast"]
        else:
            min_len = min(len(province_actual), len(r["actual"]))
            province_actual = province_actual[:min_len] + r["actual"][:min_len]
            province_forecast = province_forecast[:min_len] + r["forecast"][:min_len]
    # 计算全省指标
    rmse = np.sqrt(mean_squared_error(province_actual, province_forecast))
    mae = mean_absolute_error(province_actual, province_forecast)
    actual_safe = np.where(province_actual==0, 1e-6, province_actual)
    mape = np.mean(np.abs((province_actual - province_forecast) / actual_safe)) * 100
    print(f"\nProvince Forecast Metrics: RMSE: {rmse:.3f}, MAE: {mae:.3f}, MAPE: {mape:.2f}%")

    # 设置字体
    try:
        font_list = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Zen Hei' in font_list:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        else:
            raise Exception("WenQuanYi Zen Hei font not found")
    except Exception as e:
        print("无法加载 WenQuanYi Zen Hei 字体，请检查该字体是否安装，错误信息：", e)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制全省预测曲线
    plt.figure(figsize=(10,6))
    plt.plot(province_actual, label='Province Actual', marker='o')
    plt.plot(province_forecast, label='Province Forecast', marker='x', linestyle='--')
    plt.legend()
    plt.title("Province Forecast Results")
    plt.xlabel("Hour")
    plt.ylabel("Wind Power Output")
    plt.gca().text(0.05, 0.95, 
               f"RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAPE: {mape:.2f}%", 
               transform=plt.gca().transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle="round", fc="w"))
    
    fig_folder = "../figure"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # 保存图像到 figure 文件夹中，并根据城市和日期命名图片文件
    save_path = os.path.join(fig_folder, f"Province_{target_day}_test_Forecast.png")
    plt.savefig(save_path)
    plt.close()
    
    return province_actual, province_forecast, {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# 对未来日指标与历史日指标计算相似性
def compute_similarity_future(future_metric, historical_metric, params):
    """
    future_metric: dict，包含 'weather_ts'、'season_sin'、'season_cos'；
    historical_metric: dict，包含 'weather_ts'、'slope'、'wind_series'，'season_sin'、'season_cos'
    params: 权重参数字典, now contains 'w_feature', 'w_season'
    返回：总相似性分数
    """
    # 计算天气信息相似性（假设两者均为 (24, n_features) 的数组）
    weather1 = future_metric['weather_ts']
    weather2 = historical_metric['weather_ts']
    min_length = min(weather1.shape[0], weather2.shape[0])
    weather1 = weather1[:min_length, :]
    weather2 = weather2[:min_length, :]
    weather_dist = np.sqrt(np.sum((weather1 - weather2)**2, axis=0))
    sim_feature = - np.dot(params['w_feature'], weather_dist)
    
    # Add season similarity for consistency with _compute_similarity
    season_diff = np.sqrt(
        (future_metric['season_sin'] - historical_metric['season_sin'])**2 +
        (future_metric['season_cos'] - historical_metric['season_cos'])**2
    )
    sim_season = - params['w_season'] * season_diff
    
    total_similarity = sim_feature + sim_season
    return total_similarity

# 计算相似日
def get_similar_day_future(future_metric, params, daily_dates, daily_metrics):
    """
    对于给定的未来一天指标 future_metric（字典），遍历所有历史日，计算相似性并返回最相似的历史日期
    """
    similarities = {}
    for candidate_day in daily_dates:
        sim = compute_similarity_future(future_metric, daily_metrics[candidate_day], params)
        similarities[candidate_day] = sim
    best_day = max(similarities, key=similarities.get)
    return best_day

# 实际预测未来n天
def predict_province_future(cities, future_features_dict, start_date, n_days):
    """
    对于未来n天，利用每个城市未来特征数据，通过历史相似性方法预测风电出力，然后累加得到全省预测。
    参数:
      cities: 城市列表
      future_features_dict: dict，键为城市名称，值为该城市未来n天特征数据，格式为DataFrame，
         要求包含训练时使用的天气特征（例如 't2m','ws100m','wd100m','sp'），并且是原始尺度、小时级、已插值的数据。
      start_date: 预测开始日期 (YYYY-MM-DD字符串)
      n_days: 预测未来天数，每天24个数据点
    返回:
      province_future_forecast: 全省未来预测序列（1D numpy数组）
    """
    province_future_forecast = None
    fcst_time_overall = datetime.now(timezone.utc).replace(tzinfo=None)  # 去掉时区信息

    # Load normalization parameters once
    norm_model_file = '../output/wind_data_normalization_models.json'
    if not os.path.exists(norm_model_file):
        print(f"CRITICAL: Normalization model file not found at {norm_model_file}. Cannot proceed with prediction.")
        return None
    with open(norm_model_file, 'r') as f:
        norm_models = json.load(f)
    print("Successfully loaded normalization models for prediction.")

    for city in cities:
        print(f"\nPredicting future {n_days} days of wind power output for [{city}]...")
        
        raw_city_future_features = future_features_dict.get(city)
        if raw_city_future_features is None or raw_city_future_features.empty:
            print(f"[{city}] No future_features data provided or data is empty in the input dict. Skipping city.")
            continue
        
        # --- Start: Normalization of future weather data for the current city ---
        city_future_features_normalized = raw_city_future_features.copy()

        # Temperature t2m (z-score normalization)
        if 't2m' in norm_models and 't2m' in city_future_features_normalized.columns:
            mean_t2m = norm_models['t2m']['mean']
            std_t2m = norm_models['t2m']['std']
            if std_t2m == 0: std_t2m = 1 # Avoid division by zero, though unlikely for std from training data
            city_future_features_normalized['t2m'] = (city_future_features_normalized['t2m'] - mean_t2m) / std_t2m
        else:
            print(f"[{city}] Warning: 't2m' normalization parameters or column not found. Skipping t2m normalization.")

        # Wind speed ws100m (z-score normalization)
        if 'ws100m' in norm_models and 'ws100m' in city_future_features_normalized.columns:
            mean_ws100m = norm_models['ws100m']['mean']
            std_ws100m = norm_models['ws100m']['std']
            if std_ws100m == 0: std_ws100m = 1
            city_future_features_normalized['ws100m'] = (city_future_features_normalized['ws100m'] - mean_ws100m) / std_ws100m
        else:
            print(f"[{city}] Warning: 'ws100m' normalization parameters or column not found. Skipping ws100m normalization.")

        # Pressure sp (z-score normalization)
        if 'sp' in norm_models and 'sp' in city_future_features_normalized.columns:
            mean_sp = norm_models['sp']['mean']
            std_sp = norm_models['sp']['std']
            if std_sp == 0: std_sp = 1
            city_future_features_normalized['sp'] = (city_future_features_normalized['sp'] - mean_sp) / std_sp
        else:
            print(f"[{city}] Warning: 'sp' normalization parameters or column not found. Skipping sp normalization.")

        # Wind direction wd100m: convert to sin/cos components
        # The model expects 'wd100m_sin' and 'wd100m_cos'
        if 'wd100m' in city_future_features_normalized.columns:
            print(f"[{city}] Converting wd100m to sin/cos components for future data.")
            city_future_features_normalized['wd100m_sin'] = np.sin(city_future_features_normalized['wd100m'] * np.pi / 180)
            city_future_features_normalized['wd100m_cos'] = np.cos(city_future_features_normalized['wd100m'] * np.pi / 180)
            # We can drop original 'wd100m' after conversion if desired, but it is not strictly necessary
            # if not used by subsequent code for this DataFrame copy.
        elif 'wd100m_sin' in city_future_features_normalized.columns and 'wd100m_cos' in city_future_features_normalized.columns:
            print(f"[{city}] Using provided wd100m_sin and wd100m_cos components for future data.")
        else:
            print(f"[{city}] CRITICAL: Wind direction 'wd100m' (or its sin/cos components) not found in future data for normalization. Skipping city.")
            continue
        # --- End: Normalization of future weather data ---

        # 1a. Get historical data & train/load weights (original step 1 & 2)
        normalized_df_history = get_history_weather_data_for_city(city) # This df is already normalized by normalize_data()
        if normalized_df_history.empty:
            print(f"[{city}] No historical data found. Skipping city.")
            continue
        daily_dates, daily_metrics = get_daily_data(normalized_df_history) 
        if not daily_dates or not daily_metrics:
            print(f"[{city}] Historical daily metrics are empty. Skipping city.")
            continue
        best_params = load_weights(city)
        
        # 3. Construct future_daily_metrics from the now normalized `city_future_features_normalized`
        future_daily_metrics = {}
        current_pred_date_obj_base = pd.to_datetime(start_date) 

        city_hourly_forecasts_for_db = []

        # Ensure n_days aligns with the actual shape of the processed future data if it was truncated by data prep
        # However, predict_province_future is called with specific n_days, 
        # and data prep should provide data for these n_days.
        # The loop `for day_offset in range(n_days)` will iterate based on the requested n_days.
        # Slicing `city_future_features_normalized.iloc[day_offset*24:(day_offset+1)*24]` will handle data availability.

        for day_offset in range(n_days):
            day_data = city_future_features_normalized.iloc[day_offset*24:(day_offset+1)*24]
            
            if len(day_data) < 24:
                print(f"[{city}] Future day {day_offset + 1} (date: {current_pred_date_obj_base + pd.Timedelta(days=day_offset)}) normalized data is incomplete ({len(day_data)} points). Skipping this day for prediction.")
                continue

            # Wind direction components should already be in day_data from normalization step
            # Check if essential columns for weather_ts are present and not NaN after normalization
            required_ts_cols = ['t2m', 'ws100m', 'wd100m_sin', 'wd100m_cos', 'sp']
            missing_cols_in_day_data = [col for col in required_ts_cols if col not in day_data.columns]
            if missing_cols_in_day_data:
                print(f"[{city}] Future day {day_offset + 1}: Essential columns {missing_cols_in_day_data} missing in day_data for weather_ts. Skipping day.")
                continue
            
            if day_data[required_ts_cols].isnull().any().any():
                print(f"[{city}] Future day {day_offset + 1}: NaNs found in required columns for weather_ts construction after normalization. Skipping this day.")
                # print(day_data[required_ts_cols].isnull().sum())
                continue
                
            weather_ts = day_data[required_ts_cols].values
            
            actual_future_date_for_metric = current_pred_date_obj_base + pd.Timedelta(days=day_offset)
            month = actual_future_date_for_metric.month
            season_sin = np.sin(2 * np.pi * (month - 1) / 12)
            season_cos = np.cos(2 * np.pi * (month - 1) / 12)
            
            future_daily_metrics[day_offset] = {
                'weather_ts': weather_ts,
                'wind_series': np.zeros(24), 
                'season_sin': season_sin, 
                'season_cos': season_cos  
            }
        
        # 4. 对每个未来日找到最相似的历史日，并取该日的风电序列作为预测
        city_future_forecasts = []
        if not future_daily_metrics: # Check if any daily metrics were successfully constructed
            print(f"[{city}] No valid future daily metrics could be constructed. Skipping forecast generation for this city.")
            # No need to proceed to DB write if city_hourly_forecasts_for_db is empty
        else:
            for day_idx in sorted(future_daily_metrics.keys()):
                future_metric = future_daily_metrics[day_idx]
                similar_day = get_similar_day_future(future_metric, best_params, daily_dates, daily_metrics)
                print(f"[{city}] Future day {day_idx+1}, most similar historical day: {similar_day}")
                forecast_norm = daily_metrics[similar_day]['wind_series']
                forecast_df = pd.DataFrame({'wind_output': forecast_norm})
                forecast_df = unnormalize_data(forecast_df)
                forecast = forecast_df['wind_output'].values 
                city_future_forecasts.append(forecast)

                actual_forecast_date = current_pred_date_obj_base + pd.Timedelta(days=day_idx)

                for hour_offset in range(len(forecast)):
                    date_time_val = pd.Timestamp(actual_forecast_date) + pd.Timedelta(hours=hour_offset + 1)
                    
                    db_record = {
                        'id': pub_tools.generate_snowflake_id(),  # Added Snowflake ID
                        'city_name': city,
                        'date_time': date_time_val.to_pydatetime(),
                        'model': 'WindOutputBaselineSimilarDay',
                        'wind_output': float(forecast[hour_offset]),
                        'fcst_time': fcst_time_overall,
                        'similar_date': similar_day 
                    }
                    city_hourly_forecasts_for_db.append(db_record)

        if not city_future_forecasts: # Check if any forecasts were actually made for the city
            print(f"[{city}] No forecasts generated (city_future_forecasts is empty). Skipping DB write for this city.")
            # continue to next city is implicitly handled by loop structure if province_future_forecast is not built upon
        else: # Only proceed if there are forecasts
            city_forecast_full = np.concatenate(city_future_forecasts)
            if province_future_forecast is None:
                province_future_forecast = city_forecast_full
            else:
                min_len = min(len(province_future_forecast), len(city_forecast_full))
                province_future_forecast = province_future_forecast[:min_len] + city_forecast_full[:min_len]

        if city_hourly_forecasts_for_db:
            engine_db, metadata_db = None, None  
            try:
                engine_db, metadata_db = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
                # 将记录转换为DataFrame并使用upsert_to_db函数
                records_df = pd.DataFrame(city_hourly_forecasts_for_db)
                db_tools.upsert_to_db(engine_db, records_df, 'wind_output_forecast', update_column='wind_output')
                print(f"Successfully upserted {len(city_hourly_forecasts_for_db)} hourly forecast records to DB for city {city}")
            except Exception as e:
                print(f"Error writing forecast to DB for city {city}: {e}")
            finally:
                if engine_db:
                    db_tools.release_db_connection(engine_db)
                    
    # Set font for plotting (if any plot is generated after this loop)
    if province_future_forecast is None:
        print("\nNo province future forecast could be generated as no city forecasts were successful.")
        return None # Or handle as appropriate
        
    try:
        # Set font for plotting (if any plot is generated after this loop)
        font_list = [f.name for f in fm.fontManager.ttflist]
        if 'WenQuanYi Zen Hei' in font_list:
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        else:
            raise Exception("WenQuanYi Zen Hei font not found")
    except Exception as e:
        print("无法加载 WenQuanYi Zen Hei 字体，请检查该字体是否安装，错误信息：", e)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False

    # 绘制全省未来预测曲线
    plt.figure(figsize=(12,6))
    plt.plot(province_future_forecast, label='Province Future Forecast', marker='o')
    plt.title(f"Province Future {n_days} Days Forecast")
    plt.xlabel("Hour")
    plt.ylabel("Wind Power Output")
    plt.legend()
    
    fig_folder = "../figure"
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    # 保存图像到 figure 文件夹中，并根据城市和日期命名图片文件
    save_path = os.path.join(fig_folder, f"Province_{start_date}_real_Forecast.png")
    plt.savefig(save_path)
    plt.close()
    return province_future_forecast

def get_history_weather_data_for_city(city):
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
            'id', 'region_code', 'region_name', 'model', 'ws10m', 'wd10m', 'gust', 'irra', 'tp', 'lng', 'lat', 'time_fcst'
            ],
        inplace=True
    )

    weather_df.sort_index(inplace=True)

    neimeng_wind_output = Table('neimeng_wind_power', metadata, autoload_with=engine)
    query = select(neimeng_wind_output).where(
        neimeng_wind_output.c.type == '3'
    )

    output_df = db_tools.read_from_db(engine, query)

    db_tools.release_db_connection(engine)

    output_df.drop(columns=['type', 'city_name', 'id'], inplace=True)
    output_df['date_time'] = pd.to_datetime(output_df['date_time'])
    output_df.set_index('date_time', inplace=True)
    output_df.rename(columns={'date_time': 'datetime', 'value': 'wind_output'}, inplace=True)
    output_df = output_df.resample('H', closed='right', label='right').mean()

    output_df.sort_index(inplace=True)

    merged_df = pd.merge(weather_df, output_df, left_index=True, right_index=True, how='inner')
    merged_df = merged_df.dropna()
    # 强制将索引转换为 DatetimeIndex
    merged_df.index = pd.to_datetime(merged_df.index)
    merged_df['date'] = merged_df.index.date
    merged_df = merged_df.groupby('date').filter(lambda group: len(group) == 24)
    normalized_df = normalize_data(merged_df)

    return normalized_df

def get_predict_weather_data_for_city(city: str, start_date: str, end_date: str):
    engine, metadata = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
    terraqt_weather = Table('terraqt_weather', metadata, autoload_with=engine)
    
    start_dt_obj = datetime.strptime(start_date, '%Y-%m-%d')
    # end_date is inclusive for the query, but for pd.date_range, we might need to adjust based on n_days
    # For n_days calculation, if end_date is '2023-01-05' and start_date is '2023-01-01', it means 5 days.
    end_dt_obj_inclusive = datetime.strptime(end_date, '%Y-%m-%d')
    # Query up to the end of the end_date
    query_end_dt = end_dt_obj_inclusive + timedelta(days=1) 

    query = select(terraqt_weather).where(
        (terraqt_weather.c.model == 'gfs_surface') &
        (terraqt_weather.c.region_name == city) &
        (terraqt_weather.c.ts >= start_dt_obj) &
        (terraqt_weather.c.ts < query_end_dt) # Use < for up to the start of the next day
    )

    weather_df = db_tools.read_from_db(engine, query)
    db_tools.release_db_connection(engine)

    if weather_df.empty:
        print(f"[{city}] No future weather data found in DB for {start_date} to {end_date}. Returning empty DataFrame.")
        return pd.DataFrame() # Return an empty DataFrame if no data

    weather_df['ts'] = pd.to_datetime(weather_df['ts'])
    weather_df.rename(columns={'ts': 'datetime'}, inplace=True)
    weather_df.set_index('datetime', inplace=True)
    
    # Define all potentially useful weather columns from gfs_surface that model might need
    # Keep only the columns that are used by the model or for deriving model features
    # Current features used: 't2m', 'ws100m', 'wd100m_sin', 'wd100m_cos', 'sp'
    # So we need 't2m', 'ws100m', 'sp', and 'wd100m' (if sin/cos are derived later) or 'wd100m_sin'/'cos' if provided
    cols_to_keep = ['t2m', 'ws100m', 'sp', 'wd100m'] # Assume wd100m is primary for wind direction
    # Add other columns if they are directly provided and used, e.g., if GFS provides sin/cos directly
    # For now, let's assume we always get wd100m and derive sin/cos later if needed.
    
    actual_cols_present = [col for col in cols_to_keep if col in weather_df.columns]
    missing_essential_cols = [col for col in cols_to_keep if col not in weather_df.columns]
    if missing_essential_cols:
        print(f"[{city}] CRITICAL: Essential columns {missing_essential_cols} not found in fetched GFS data. Returning empty DataFrame.")
        return pd.DataFrame()
        
    weather_df = weather_df[actual_cols_present]
    weather_df.sort_index(inplace=True)

    # Calculate n_days for creating the target hourly index
    # n_days = (end_dt_obj_inclusive - start_dt_obj).days + 1 # Correct n_days for full coverage
    # Ensure start_dt_obj is at the beginning of the day for date_range
    target_index_start_dt = pd.Timestamp(start_dt_obj.strftime('%Y-%m-%d %H:%M:%S')).normalize()
    # Calculate the number of 24-hour periods required. +1 because end_date is inclusive.
    n_periods = ((end_dt_obj_inclusive.date() - start_dt_obj.date()).days + 1) * 24
    target_hourly_index = pd.date_range(start=target_index_start_dt, periods=n_periods, freq='H')

    # Reindex to ensure hourly frequency and fill gaps with NaN for interpolation
    # Important: ensure weather_df index is timezone-naive or matches target_hourly_index timezone if any
    if weather_df.index.tz is not None:
        print(f"[{city}] Warning: Fetched weather data has timezone {weather_df.index.tz}. Converting to timezone-naive for processing.")
        weather_df.index = weather_df.index.tz_localize(None)
    
    aligned_weather_df = weather_df.reindex(target_hourly_index)

    # Interpolate numerical features
    # We expect 't2m', 'ws100m', 'sp', 'wd100m' to be numeric here
    numeric_cols_to_interpolate = ['t2m', 'ws100m', 'sp', 'wd100m']
    for col in numeric_cols_to_interpolate:
        if col in aligned_weather_df.columns:
            if aligned_weather_df[col].isnull().any():
                print(f"[{city}] Interpolating column {col} for future weather using 'time' method.")
                aligned_weather_df[col] = aligned_weather_df[col].interpolate(method='time')
                aligned_weather_df[col] = aligned_weather_df[col].ffill().bfill() # Fill any remaining NaNs at boundaries
            if aligned_weather_df[col].isnull().all(): # Check if column is all NaN after interpolation
                print(f"[{city}] CRITICAL: Column '{col}' is all NaNs after interpolation. Returning empty DataFrame.")
                return pd.DataFrame()
        else:
            # This case should have been caught by missing_essential_cols earlier, but as a safeguard:
            print(f"[{city}] CRITICAL: Essential column '{col}' for interpolation not found. Returning empty DataFrame.")
            return pd.DataFrame()
            
    # At this point, aligned_weather_df contains raw (unnormalized), hourly, interpolated weather data.
    # No normalization here. No merging with output_df.
    print(f"[{city}] Successfully prepared future weather data from {start_date} to {end_date}. Shape: {aligned_weather_df.shape}")
    return aligned_weather_df

def get_daily_data(normalized_df):
    daily_metrics = {}
    for date, group in normalized_df.groupby('date'):
        weather_ts, wind_series = _compute_daily_metrics(group)

        month = pd.to_datetime(str(date)).month
        season_sin = np.sin(2 * np.pi * (month - 1) / 12)
        season_cos = np.cos(2 * np.pi * (month - 1) / 12)

        daily_metrics[date] = {
            'weather_ts': weather_ts,
            'wind_series': wind_series,
            'season_sin': season_sin,
            'season_cos': season_cos
        }

    daily_dates = sorted(daily_metrics.keys())
    print("参与优化的日期数量:", len(daily_dates))

    return daily_dates, daily_metrics

def search_best_weights(normalized_df, n_trials):
    daily_dates, daily_metrics = get_daily_data(normalized_df)

    # 优化过程：通过 lambda 包装 _objective 传入 daily_dates 和 daily_metrics
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: _objective(trial, daily_dates, daily_metrics), n_trials)
    print("Best trial:")
    print(study.best_trial.values)
    print(study.best_trial.params)

    best_params = {
        'w_feature': np.array([
            study.best_trial.params['w_feature_0'],
            study.best_trial.params['w_feature_1'],
            study.best_trial.params['w_feature_2'],
            study.best_trial.params['w_feature_3'],
            study.best_trial.params['w_feature_4']
        ]),
        'w_season': study.best_trial.params['w_season']
    }

    return best_params

def train(cities: list, n_trials: int):
    for city in cities:
        normalized_df = get_history_weather_data_for_city(city)
        best_params = search_best_weights(normalized_df, n_trials)
        save_weights(best_params, city)

if __name__ == '__main__':
    pass