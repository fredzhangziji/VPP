"""
Informer模型预测脚本
用于使用训练好的Informer模型预测未来风力发电输出，包含数据加载、预处理、预测和可视化过程
并将预测结果保存到数据库
"""

import os
import mlflow
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
from sqlalchemy import text

from model import Informer
from utils import setup_chinese_font, evaluate_model
from data_preprocessing import (
    CITY_NAME_MAPPING_DICT, CITIES_FOR_POWER, get_history_weather_data_for_city, 
    get_history_wind_power_for_city, get_predicted_weather_data_for_city,
    merge_weather_and_power_df, preprocess_data, set_time_wise_feature, _add_more_features_for_future,
    apply_feature_scaling
)
from pub_tools import get_system_font_path, generate_snowflake_id, db_tools, const

import logging
import pub_tools.logging_config
logger = logging.getLogger('informer')

# 设置matplotlib中文字体
setup_chinese_font()

# 配置路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# MLflow设置
MLFLOW_EXPERIMENT = "Informer Wind Power Prediction"
MODEL_NAME = "Informer_wind_power"  # 模型名称，用于数据库记录

USE_MLFLOW = False
try:
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    USE_MLFLOW = True
except Exception as e:
    logger.warning(f"警告: MLflow设置失败: {e}")
    logger.info("继续运行但不使用MLflow记录实验。")

# 预测参数
class Args:
    def __init__(self):
        self.model = 'informer'
        self.seq_len = 96  # 输入序列长度 (4天 * 24小时)
        self.label_len = 48  # 解码器标签序列长度 (2天 * 24小时)
        self.pred_len = 120  # 预测序列长度 (5天 * 24小时)
        
        # 预测时需要与训练时一致的参数
        self.enc_in = 20  # 编码器输入维度(会根据实际特征数量更新)
        self.dec_in = 1   # 解码器输入维度
        self.c_out = 1    # 输出维度
        self.factor = 5   # probsparse注意力因子
        self.d_model = 512  # 模型维度
        self.n_heads = 8    # 多头注意力头数
        self.e_layers = 2   # 编码器层数
        self.d_layers = 1   # 解码器层数
        self.d_ff = 2048    # 前馈网络维度
        self.dropout = 0.05     # dropout率
        self.attn = 'prob'      # 注意力机制类型
        self.embed = 'timeF'    # 时间特征编码
        self.activation = 'gelu'  # 激活函数
        self.distil = True      # 是否使用蒸馏
        self.output_attention = False  # 是否输出注意力权重
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_for_city(city_power, args):
    """
    为指定城市预测未来5天的风力发电输出
    
    参数:
        city_power: 城市名称
        args: 预测参数
    
    返回:
        pandas DataFrame: 包含日期时间和预测的风力发电量，用于数据库存储
    """
    CITY_FOR_POWER_DATA = city_power
    CITY_FOR_WEATHER_DATA = CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA]
    MODEL_PATH = os.path.join(MODEL_DIR, f'informer_model_{CITY_FOR_POWER_DATA}.pth')
    SCALER_PATH = os.path.join(MODEL_DIR, f'scalers_{CITY_FOR_POWER_DATA}.pkl')
    
    logger.info(f"\n开始为城市 {CITY_FOR_POWER_DATA} 预测风力发电输出...")
    
    # 检查模型和缩放器文件是否存在
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"警告: 找不到城市 {CITY_FOR_POWER_DATA} 的模型文件, 跳过预测")
        return None
    if not os.path.exists(SCALER_PATH):
        logger.warning(f"警告: 找不到城市 {CITY_FOR_POWER_DATA} 的缩放器文件, 跳过预测")
        return None
    
    try:
        with mlflow.start_run(run_name=f"informer_predict_{CITY_FOR_POWER_DATA}") if USE_MLFLOW else nullcontext():
            # 1. 获取历史数据，用于构建输入序列
            weather_df = get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
            power_df = get_history_wind_power_for_city(CITY_FOR_POWER_DATA)
            
            if weather_df.empty or power_df.empty:
                logger.warning(f"警告: {CITY_FOR_POWER_DATA} 缺少历史数据, 跳过预测")
                return None
                
            merged_df = merge_weather_and_power_df(weather_df, power_df)
            if merged_df.empty:
                logger.warning(f"警告: {CITY_FOR_POWER_DATA} 合并数据为空, 跳过预测")
                return None
                
            preprocessed_df = preprocess_data(merged_df, CITY_FOR_POWER_DATA)
            
            # 确认历史数据量足够
            if len(preprocessed_df) < args.seq_len:
                logger.warning(f"警告: {CITY_FOR_POWER_DATA} 历史数据量不足，需要至少 {args.seq_len} 条记录，但只有 {len(preprocessed_df)} 条")
                return None
            
            # 使用最后seq_len个时间点的数据
            history_data = preprocessed_df.iloc[-args.seq_len:]
            
            # 2. 获取未来天气数据
            future_weather_raw_df = get_predicted_weather_data_for_city(CITY_FOR_WEATHER_DATA)
            if future_weather_raw_df.empty:
                logger.warning(f"警告: {CITY_FOR_WEATHER_DATA} 未找到未来天气数据, 跳过预测")
                return None
                
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
                return None
                
            future_weather_df.interpolate(method='time', inplace=True)
            future_weather_df.fillna(method='ffill', inplace=True)
            future_weather_df.fillna(method='bfill', inplace=True)
            
            # 4. 构造未来特征
            future_features_df = _add_more_features_for_future(future_weather_df, preprocessed_df)
            
            # 5. 加载缩放器和模型
            with open(SCALER_PATH, 'rb') as f:
                scalers = pickle.load(f)
            
            # 创建未来时间序列的特征DataFrame
            future_features_df = future_features_df.reset_index().rename(columns={'index': 'datetime'})
            future_features_df = set_time_wise_feature(future_features_df.copy())
            
            # 获取特征列（排除不需要的列）
            feature_cols = list(future_features_df.columns)
            for col in ['time_idx', 'group_id', 'datetime']:
                if col in feature_cols:
                    feature_cols.remove(col)
            
            # 准备输入数据
            history_x = history_data[feature_cols].values
            
            # 缩放数据 - 使用改进后的特征缩放方法
            history_x_scaled = apply_feature_scaling(history_data[feature_cols], scalers)
            future_x_scaled = apply_feature_scaling(future_features_df[feature_cols], scalers)
            
            # 制作历史和未来的输入序列
            history_x_tensor = torch.FloatTensor(history_x_scaled).unsqueeze(0)  # [1, seq_len, feature_dim]
            
            # 为解码器准备输入（标签长度部分，用0填充）
            dec_inp = torch.zeros((1, args.label_len, 1)).float()
            
            # 6. 加载模型
            args.enc_in = history_x_tensor.shape[2]  # 更新编码器输入维度
            
            model = Informer(
                enc_in=args.enc_in,
                dec_in=args.dec_in,
                c_out=args.c_out,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                factor=args.factor,
                d_model=args.d_model,
                n_heads=args.n_heads,
                e_layers=args.e_layers,
                d_layers=args.d_layers,
                d_ff=args.d_ff,
                dropout=args.dropout,
                attn=args.attn,
                embed=args.embed,
                activation=args.activation,
                output_attention=args.output_attention,
                distil=args.distil,
                device=args.device
            ).to(args.device)
            
            model.load_state_dict(torch.load(MODEL_PATH, map_location=args.device))
            model.eval()
            
            # 7. 进行预测
            with torch.no_grad():
                history_x_tensor = history_x_tensor.to(args.device)
                dec_inp = dec_inp.to(args.device)
                
                if args.output_attention:
                    outputs, _ = model(history_x_tensor, dec_inp)
                else:
                    outputs = model(history_x_tensor, dec_inp)  # [1, pred_len, 1]
                
                # 转换为numpy数组并反缩放
                predictions = outputs.cpu().numpy().reshape(-1, 1)
                predictions = scalers['target_scaler'].inverse_transform(predictions)
                
            # 8. 处理预测结果
            result_df = pd.DataFrame(index=future_hourly_index)
            result_df[f'predicted_wind_output_{CITY_FOR_POWER_DATA}'] = predictions.flatten()
            
            # 9. 生成用于数据库的格式
            db_result_df = pd.DataFrame({
                'date_time': future_hourly_index,
                'wind_output': predictions.flatten()
            })
            
            # 10. 保存预测结果到CSV
            output_path = os.path.join(OUTPUT_DIR, f'future_predictions_{CITY_FOR_POWER_DATA}.csv')
            db_result_df.to_csv(output_path, index=False)
            
            if USE_MLFLOW:
                mlflow.log_artifact(output_path)
            
            # 11. 生成预测结果图表
            plt.figure(figsize=(15, 7))
            plt.plot(result_df.index, result_df[f'predicted_wind_output_{CITY_FOR_POWER_DATA}'], 
                     label='预测风力发电输出', color='orange', marker='.')
            plt.title(f'{CITY_FOR_POWER_DATA} - 未来五天风力发电预测')
            plt.xlabel('时间')
            plt.ylabel('预测风力发电输出 (MW)')
            plt.legend()
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = os.path.join(FIGURE_DIR, f"{CITY_FOR_POWER_DATA}_future_5day_predictions.png")
            plt.savefig(save_path)
            plt.close()
            
            if USE_MLFLOW:
                mlflow.log_artifact(save_path)
            
            # 12. 将预测结果写入数据库
            try:
                engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
                now = datetime.now(timezone.utc).replace(tzinfo=None)  # 去掉时区信息
                
                # 准备写入数据库的记录
                records = []
                for dt, pred in zip(future_hourly_index, predictions.flatten()):
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
            return db_result_df
    
    except Exception as e:
        logger.error(f"预测城市 {CITY_FOR_POWER_DATA} 时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


class nullcontext:
    """替代Python 3.7以下版本缺少的nullcontext"""
    def __enter__(self): return None
    def __exit__(self, exc_type, exc_val, exc_tb): pass


if __name__ == '__main__':
    args = Args()
    logger.info(f"将为以下城市进行预测: {CITIES_FOR_POWER}")
    all_predictions = {}
    
    for city in CITIES_FOR_POWER:
        try:
            pred_df = predict_for_city(city, args)
            if pred_df is not None:
                all_predictions[city] = pred_df
        except Exception as e:
            logger.error(f"预测城市 {city} 时发生错误: {e}")
    
    # 可选：合并所有城市的预测结果到一个文件
    if all_predictions:
        combined_df = pd.DataFrame()
        for city, df in all_predictions.items():
            df_copy = df.copy()
            df_copy.rename(columns={'wind_output': f'wind_output_{city}'}, inplace=True)
            if combined_df.empty:
                combined_df = df_copy
            else:
                combined_df = pd.merge(combined_df, df_copy, on='date_time', how='outer')
        
        combined_path = os.path.join(OUTPUT_DIR, 'all_cities_predictions.csv')
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"所有城市的预测结果已合并保存到 {combined_path}")
    
    logger.info("所有城市预测完成!") 