"""
此模块实现了基于XGBoost的风力发电预测应用功能。
从数据库加载训练好的模型和天气数据，生成未来五天的风力发电预测，
同时支持多城市预测、数据可视化和预测结果存入数据库等功能。
"""

import os
# 设置 MLflow 环境变量，避免权限问题
os.environ['MLFLOW_TRACKING_DIR'] = os.path.join(os.path.dirname(__file__), 'mlruns')
# 或者完全禁用 MLflow
# os.environ['MLFLOW_DISABLE'] = 'true'

import mlflow
import pandas as pd
import xgboost as xgb
import numpy as np
from data_preprocessing import (
    CITY_NAME_MAPPING_DICT, CITIES_FOR_POWER, get_predicted_weather_data_for_city, 
    get_history_weather_data_for_city, get_history_wind_power_for_city,
    merge_weather_and_power_df, preprocess_data, set_time_wise_feature, _add_more_features_for_future,
    ensure_consistent_encoding
)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pub_tools import get_system_font_path, generate_snowflake_id, db_tools, const
from datetime import datetime, timezone
from sqlalchemy import text

import logging
import pub_tools.logging_config
logger = logging.getLogger('xgboost')

# 设置matplotlib中文字体
font_path = get_system_font_path()
if font_path is not None:
    try:
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"已设置matplotlib字体: {font_path}")
    except Exception as e:
        logger.warning(f"设置字体时出错: {e}, 继续使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用一个通用的备选字体
else:
    logger.warning("未找到可用的中文字体，继续使用默认字体。")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用一个通用的备选字体

MODEL_DIR = 'models'
MLFLOW_EXPERIMENT = "XGBoost Wind Power Prediction"
MODEL_NAME = "XGBoost_wind_power"  # 模型名称，用于数据库记录

# 安全设置 MLflow
USE_MLFLOW = False
try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    USE_MLFLOW = True
    logger.info(f"MLflow 实验已设置为: {MLFLOW_EXPERIMENT}")
except Exception as e:
    logger.warning(f"MLflow 设置失败: {e}, 将不使用 MLflow 记录")
    USE_MLFLOW = False

def load_model(model_path):
    """加载XGBoost预测模型
    
    从指定路径加载已训练的XGBoost模型，处理可能的错误情况。
    
    入参:
        model_path: 模型文件的路径
        
    返回:
        成功时返回加载的XGBoost模型，失败时返回None
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return None

def predict_for_city(city_power):
    """为指定城市预测未来5天的风力发电输出
    
    加载指定城市的训练模型，获取历史数据构建特征，结合未来天气预报数据，
    预测未来5天的风力发电输出。生成预测结果图表，并将结果保存到CSV文件和数据库。
    
    入参:
        city_power: 城市名称，用于确定使用哪个城市的模型和数据
        
    返回:
        db_result_df: DataFrame，包含日期时间和预测的风力发电量，用于数据库存储
    """
    global USE_MLFLOW
    
    CITY_FOR_POWER_DATA = city_power
    CITY_FOR_WEATHER_DATA = CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA]
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, f'xgb_best_model_{CITY_FOR_POWER_DATA}.json')
    FEATURE_PATH = os.path.join(MODEL_DIR, f'trained_features_{CITY_FOR_POWER_DATA}.txt')
    
    logger.info(f"\n开始为城市 {CITY_FOR_POWER_DATA} 预测风力发电输出...")
    
    # 检查模型和特征文件是否存在
    if not os.path.exists(BEST_MODEL_PATH):
        logger.warning(f"警告: 找不到城市 {CITY_FOR_POWER_DATA} 的模型文件, 跳过预测")
        return
    if not os.path.exists(FEATURE_PATH):
        logger.warning(f"警告: 找不到城市 {CITY_FOR_POWER_DATA} 的特征文件, 跳过预测")
        return
    
    try:
        # 使用 MLflow 记录预测过程，但仅当 USE_MLFLOW 为 True 时
        if USE_MLFLOW:
            try:
                mlflow.start_run(run_name=f"predict_future_{CITY_FOR_POWER_DATA}")
            except Exception as e:
                logger.warning(f"启动 MLflow run 失败: {e}, 将不使用 MLflow")
                USE_MLFLOW = False
                
        # 1. 获取历史数据，生成历史特征（用于滞后项）
        weather_df = get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
        power_df = get_history_wind_power_for_city(CITY_FOR_POWER_DATA)
        
        if weather_df.empty or power_df.empty:
            logger.warning(f"警告: {CITY_FOR_POWER_DATA} 缺少历史数据, 跳过预测")
            return
            
        merged_df = merge_weather_and_power_df(weather_df, power_df)
        if merged_df.empty:
            logger.warning(f"警告: {CITY_FOR_POWER_DATA} 合并数据为空, 跳过预测")
            return
            
        preprocessed_df = preprocess_data(merged_df, CITY_FOR_POWER_DATA)
        
        # 2. 获取未来天气数据
        future_weather_raw_df = get_predicted_weather_data_for_city(CITY_FOR_WEATHER_DATA)
        if future_weather_raw_df.empty:
            logger.warning(f"警告: {CITY_FOR_WEATHER_DATA} 未找到未来天气数据, 跳过预测")
            return
            
        # 3. 对未来天气数据补齐索引、插值
        if not future_weather_raw_df.index.is_unique:
            future_weather_raw_df = future_weather_raw_df[~future_weather_raw_df.index.duplicated(keep='first')]
            
        last_historical_datetime = preprocessed_df.index.max()
        future_start_dt = last_historical_datetime + pd.Timedelta(hours=1)
        future_end_dt = future_start_dt + pd.Timedelta(days=5) - pd.Timedelta(hours=1)
        future_hourly_index = pd.date_range(start=future_start_dt, end=future_end_dt, freq='h')
        
        future_weather_df = future_weather_raw_df.reindex(future_hourly_index)
        if future_weather_df.empty or future_weather_df.isnull().all().all():
            logger.warning(f"警告: {CITY_FOR_WEATHER_DATA} 未来时间段内无天气数据, 跳过预测")
            return
            
        future_weather_df.interpolate(method='time', inplace=True)
        future_weather_df.fillna(method='ffill', inplace=True)
        future_weather_df.fillna(method='bfill', inplace=True)
        
        # 4. 构造未来特征 (_add_more_features_for_future 不包含滞后项)
        future_features_df = _add_more_features_for_future(future_weather_df, preprocessed_df)
        future_features_df['datetime'] = future_features_df.index
        
        # 5. 时间特征处理
        time_wise_future_df = set_time_wise_feature(future_features_df.copy())
        
        # 应用一致的特征编码
        time_wise_future_df = ensure_consistent_encoding(time_wise_future_df, CITY_FOR_POWER_DATA)
        
        # 6. 加载特征名
        with open(FEATURE_PATH, 'r') as f:
            trained_features = [line.strip() for line in f.readlines() if line.strip()]
        
        # 确保future_df有所有需要的特征列
        missing_features = [f for f in trained_features if f not in time_wise_future_df.columns]
        if missing_features:
            logger.warning(f"警告: 未来数据缺少某些特征: {missing_features}")
            # 添加缺失的特征列，填充0
            for feature in missing_features:
                time_wise_future_df[feature] = 0
        
        # 7. 加载模型
        model = load_model(BEST_MODEL_PATH)
        if model is None:
            logger.warning(f"警告: 无法加载 {CITY_FOR_POWER_DATA} 的模型, 跳过预测")
            return
            
        # 8. 预测
        X_pred = time_wise_future_df[trained_features]
        predictions = model.predict(X_pred)
        
        # 9. 输出结果
        result_df = pd.DataFrame(index=future_hourly_index)
        result_df[f'predicted_wind_output_{CITY_FOR_POWER_DATA}'] = predictions
        
        # 创建可用于数据库写入的格式
        db_result_df = pd.DataFrame({
            'date_time': future_hourly_index,
            'wind_output': predictions
        })
        
        # 保存预测结果到CSV
        output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'future_predictions_{CITY_FOR_POWER_DATA}.csv')
        db_result_df.to_csv(output_path, index=False)
        
        # 安全地记录 MLflow 工件
        if USE_MLFLOW:
            try:
                mlflow.log_artifact(output_path)
            except Exception as e:
                logger.warning(f"MLflow 记录工件失败: {e}")
                USE_MLFLOW = False
        
        # 保存预测结果图
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 7))
        plt.plot(result_df.index, result_df[f'predicted_wind_output_{CITY_FOR_POWER_DATA}'], 
                 label=f'Predicted Power Output', color='orange', marker='.')
        plt.title(f'{CITY_FOR_POWER_DATA} - Future Wind Power Output Prediction (Next 5 Days)')
        plt.xlabel('Datetime')
        plt.ylabel('Predicted Power Output')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(figures_dir, f"{CITY_FOR_POWER_DATA}_future_5day_predictions.png")
        plt.savefig(save_path)
        
        # 安全地记录 MLflow 图表
        if USE_MLFLOW:
            try:
                mlflow.log_artifact(save_path)
            except Exception as e:
                logger.warning(f"MLflow 记录图表失败: {e}")
                USE_MLFLOW = False
                
        plt.close()
        
        # 10. 将预测结果写入数据库
        try:
            engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
            now = datetime.now(timezone.utc).replace(tzinfo=None)  # 去掉时区信息
            
            # 准备写入数据库的记录
            records = []
            for dt, pred in zip(future_hourly_index, predictions):
                sf_id = generate_snowflake_id()
                records.append({
                    "id": sf_id,
                    "city_name": CITY_FOR_POWER_DATA,
                    "date_time": dt,
                    "model": MODEL_NAME,
                    "wind_output": float(pred),  # 确保数值类型正确
                    "fcst_time": now,
                    "similar_date": None
                })
            
            # 将记录转换为DataFrame并使用upsert_to_db函数
            records_df = pd.DataFrame(records)
            db_tools.upsert_to_db(engine, records_df, 'wind_output_forecast', update_column='wind_output')
            logger.info(f"成功写入 {len(records)} 条预测记录到数据库，城市: {CITY_FOR_POWER_DATA}")
        except Exception as e:
            logger.error(f"写入数据库失败，城市: {CITY_FOR_POWER_DATA}, 错误: {e}")
        finally:
            db_tools.release_db_connection(engine)
        
        logger.info(f"{CITY_FOR_POWER_DATA} 未来5天预测完成, 结果已保存到 {output_path} 和 {save_path}")
        
        # 安全地结束 MLflow run
        if USE_MLFLOW:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"结束 MLflow run 失败: {e}")
                
        return db_result_df
        
    except Exception as e:
        logger.error(f"预测城市 {CITY_FOR_POWER_DATA} 时发生错误: {e}")
        # 确保 MLflow run 被正确结束
        if USE_MLFLOW:
            try:
                mlflow.end_run()
            except:
                pass
        return None

if __name__ == '__main__':
    logger.info(f"将为以下城市进行预测: {CITIES_FOR_POWER}")
    all_predictions = {}
    
    for city in CITIES_FOR_POWER:
        try:
            pred_df = predict_for_city(city)
            if pred_df is not None:
                all_predictions[city] = pred_df
        except Exception as e:
            logger.error(f"预测城市 {city} 时发生错误: {e}")
    
    # 可选：合并所有城市的预测结果到一个文件
    if all_predictions:
        combined_df = pd.concat([df for df in all_predictions.values()], axis=1)
        combined_path = os.path.join(os.path.dirname(__file__), 'outputs', 'all_cities_predictions.csv')
        combined_df.to_csv(combined_path)
        logger.info(f"所有城市的预测结果已合并保存到 {combined_path}")
    
    logger.info("所有城市预测完成!")