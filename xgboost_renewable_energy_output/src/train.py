"""
此模块实现了基于XGBoost的风力发电预测模型训练流程。
包括数据加载、预处理、特征工程、超参数优化、模型训练和评估等全流程。
支持多城市训练、MLflow实验跟踪，以及模型性能可视化功能。
"""

import os
import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import (
    CITY_NAME_MAPPING_DICT, CITIES_FOR_POWER, get_history_weather_data_for_city, get_history_wind_power_for_city,
    merge_weather_and_power_df, preprocess_data, set_time_wise_feature, plot_predictions
)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pub_tools import get_system_font_path

import logging
import pub_tools.logging_config
logger = logging.getLogger('xgboost')

# 设置matplotlib中文字体
font_path = get_system_font_path()
if font_path is not None:
    plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    logger.info(f"已设置matplotlib字体: {font_path}")
else:
    logger.warning("未找到可用的中文字体，继续使用默认字体。")

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# MLflow设置，将变量声明放在最前面
USE_MLFLOW = False  # 默认为False，只有在成功连接MLflow后才设为True

try:
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    print(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    # 设置连接超时
    import socket
    socket.setdefaulttimeout(5)  # 5秒超时
    
    # 尝试创建实验
    mlflow.set_experiment("XGBoost Wind Power Prediction")
    print("MLflow实验已设置为 'XGBoost Wind Power Prediction'")
    mlflow.xgboost.autolog(exclusive=True, log_input_examples=True)
    print("MLflow自动日志记录已启用。")
    USE_MLFLOW = True  # 设置成功时为True
except Exception as e:
    print(f"警告: MLflow设置失败: {e}")
    print("继续运行但不使用MLflow记录实验。")
    USE_MLFLOW = False

def train_model_for_city(city_power):
    """为指定城市训练风电预测模型
    
    完整的模型训练流程，包含数据获取、预处理、特征工程、训练测试集划分、
    超参数优化和最终模型评估。训练完成后保存模型和特征列表，生成评估图表。
    
    入参:
        city_power: 城市名称，用于确定获取哪个城市的风电和天气数据
    """
    global USE_MLFLOW  # 在函数开头声明全局变量
    
    CITY_FOR_POWER_DATA = city_power
    CITY_FOR_WEATHER_DATA = CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA]
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, f'xgb_best_model_{CITY_FOR_POWER_DATA}.json')
    
    logger.info(f"\n开始训练城市: {CITY_FOR_POWER_DATA} 的风力发电预测模型...")
    
    # 1. 数据准备
    weather_df = get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
    power_df = get_history_wind_power_for_city(CITY_FOR_POWER_DATA)
    
    # 检查是否有数据
    if weather_df.empty or power_df.empty:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 缺少天气或电力数据, 跳过训练")
        return
    
    merged_df = merge_weather_and_power_df(weather_df, power_df)
    if merged_df.empty:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 合并数据为空, 跳过训练")
        return
        
    preprocessed_df = preprocess_data(merged_df, CITY_FOR_POWER_DATA)
    time_wise_df = set_time_wise_feature(preprocessed_df.copy())

    # 2. 特征与标签
    if 'wind_output' not in time_wise_df.columns:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 数据中没有 'wind_output' 列, 跳过训练")
        return
        
    y = time_wise_df['wind_output']
    X = time_wise_df.drop(columns=['wind_output', 'time_idx', 'group_id'], errors='ignore')

    categorical_features = ['wind_season', 'year', 'month', 'day', 'hour']
    actual_categorical = [col for col in categorical_features if col in X.columns]
    X_encoded = pd.get_dummies(X, columns=actual_categorical, drop_first=True)

    # 3. 划分数据集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=False
    )

    logger.info(f"{CITY_FOR_POWER_DATA} 数据集分割完成: 训练集 {X_train.shape[0]}行, 验证集 {X_val.shape[0]}行, 测试集 {X_test.shape[0]}行")

    # 4. Optuna 超参搜索
    def objective(trial):
        global USE_MLFLOW  # 在嵌套函数中再次声明全局变量
        
        params = {
            'objective': 'reg:squarederror',
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 有条件使用MLflow
        if USE_MLFLOW:
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    mlflow.log_metric("rmse_train_optuna", rmse)
                return rmse
            except Exception as e:
                logger.error(f"MLflow记录失败，继续不使用MLflow: {e}")
                USE_MLFLOW = False
        
        # 如果MLflow禁用或失败，直接进行训练和评估
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)

    best_params = study.best_params
    logger.info(f"{CITY_FOR_POWER_DATA} 最佳参数: {best_params}")

    # 5. 用最佳参数训练模型
    if USE_MLFLOW:
        try:
            with mlflow.start_run(run_name=f"xgb_optuna_best_{CITY_FOR_POWER_DATA}"):
                _train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                                  best_params, BEST_MODEL_PATH, X_encoded, CITY_FOR_POWER_DATA)
        except Exception as e:
            logger.error(f"MLflow记录失败，继续不使用MLflow: {e}")
            USE_MLFLOW = False
            _train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                              best_params, BEST_MODEL_PATH, X_encoded, CITY_FOR_POWER_DATA, use_mlflow=False)
    else:
        _train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, 
                          best_params, BEST_MODEL_PATH, X_encoded, CITY_FOR_POWER_DATA, use_mlflow=False)

    logger.info(f"{CITY_FOR_POWER_DATA} 的模型训练与评估全部完成，最佳模型和结果已保存。")

def _train_final_model(X_train, y_train, X_val, y_val, X_test, y_test, best_params, 
                      model_path, X_encoded, city_name, use_mlflow=True):
    """使用最佳超参数训练最终模型并进行评估
    
    使用最佳超参数训练XGBoost模型，保存模型和特征列表，
    在训练集、验证集和测试集上评估模型性能，并生成评估图表。
    支持可选的MLflow跟踪功能。
    
    入参:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征
        y_val: 验证集标签
        X_test: 测试集特征
        y_test: 测试集标签
        best_params: 通过Optuna优化得到的最佳超参数
        model_path: 模型保存路径
        X_encoded: 完整编码后的特征DataFrame，用于提取特征名
        city_name: 城市名称，用于文件命名
        use_mlflow: 是否使用MLflow记录训练过程和结果，默认为True
    """
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # 保存模型
    best_model.save_model(model_path)
    if use_mlflow:
        mlflow.log_artifact(model_path, artifact_path="model")
    
    # 保存特征名
    feature_path = os.path.join(MODEL_DIR, f"trained_features_{city_name}.txt")
    with open(feature_path, "w") as f:
        f.write("\n".join(X_encoded.columns))
    
    if use_mlflow:
        mlflow.log_artifact(feature_path)
    
    # 创建图表保存目录
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 训练集评估
    y_pred_train = best_model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    if use_mlflow:
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("r2_train", r2_train)
    
    train_plot_path = os.path.join(figures_dir, f"train_predictions_{city_name}.png")
    plot_predictions(y_train, y_pred_train, y_train.index, f"{city_name} - Train: Actual vs Predicted", 
                     train_plot_path, rmse_train, mae_train, r2_train)
    
    if use_mlflow:
        try:
            mlflow.log_artifact(train_plot_path)
        except Exception as e:
            logger.error(f"无法记录训练集图片到MLflow: {e}")
    
    # 验证集评估
    y_pred_val = best_model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)
    
    if use_mlflow:
        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("mae_val", mae_val)
        mlflow.log_metric("r2_val", r2_val)
    
    val_plot_path = os.path.join(figures_dir, f"val_predictions_{city_name}.png")
    plot_predictions(y_val, y_pred_val, y_val.index, f"{city_name} - Validation: Actual vs Predicted", 
                     val_plot_path, rmse_val, mae_val, r2_val)
    
    if use_mlflow:
        try:
            mlflow.log_artifact(val_plot_path)
        except Exception as e:
            logger.error(f"无法记录验证集图片到MLflow: {e}")
    
    # 测试集评估
    y_pred_test = best_model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    if use_mlflow:
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("r2_test", r2_test)
    
    test_plot_path = os.path.join(figures_dir, f"test_predictions_{city_name}.png")
    plot_predictions(y_test, y_pred_test, y_test.index, f"{city_name} - Test: Actual vs Predicted", 
                     test_plot_path, rmse_test, mae_test, r2_test)
    
    if use_mlflow:
        try:
            mlflow.log_artifact(test_plot_path)
        except Exception as e:
            logger.error(f"无法记录测试集图片到MLflow: {e}")

if __name__ == '__main__':
    logger.info(f"将为以下城市训练模型: {CITIES_FOR_POWER}")
    for city in CITIES_FOR_POWER:
        try:
            train_model_for_city(city)
        except Exception as e:
            logger.error(f"训练城市 {city} 的模型时发生错误: {e}")
    
    logger.info("所有城市模型训练完成!")
