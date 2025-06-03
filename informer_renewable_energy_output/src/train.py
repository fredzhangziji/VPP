"""
Informer模型训练脚本
用于训练Informer风力发电预测模型，包含数据获取、预处理、训练、评估过程和超参数优化
"""

import os
import mlflow
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.samplers import TPESampler
import pickle
import json

from model import Informer
from utils import setup_chinese_font, evaluate_model, plot_predictions, adjust_learning_rate, EarlyStopping
from data_preprocessing import (
    CITY_NAME_MAPPING_DICT, CITIES_FOR_POWER, get_history_weather_data_for_city, 
    get_history_wind_power_for_city, merge_weather_and_power_df, preprocess_data, 
    set_time_wise_feature, create_datasets
)

import logging
import pub_tools.logging_config
from pub_tools import get_system_font_path
logger = logging.getLogger('informer')

# 设置matplotlib中文字体
setup_chinese_font()

# 配置路径
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# MLflow设置
USE_MLFLOW = False
MLFLOW_EXPERIMENT = "Informer Wind Power Prediction"

try:
    mlflow_tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"MLflow Tracking URI: {mlflow_tracking_uri}")
    
    # 设置连接超时
    import socket
    socket.setdefaulttimeout(5)  # 5秒超时
    
    # 尝试创建实验
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info(f"MLflow实验已设置为 '{MLFLOW_EXPERIMENT}'")
    USE_MLFLOW = True  # 设置成功时为True
except Exception as e:
    logger.warning(f"警告: MLflow设置失败: {e}")
    logger.info("继续运行但不使用MLflow记录实验。")
    USE_MLFLOW = False

# 模型训练参数
class Args:
    def __init__(self):
        self.model = 'informer'
        self.data = 'custom'
        self.root_path = './'
        self.checkpoints = './models/'
        
        # 模型参数
        self.seq_len = 96  # 输入序列长度 (4天 * 24小时)
        self.label_len = 48  # 解码器标签序列长度 (2天 * 24小时)
        self.pred_len = 120  # 预测序列长度 (5天 * 24小时)
        
        self.enc_in = 20  # 编码器输入维度
        self.dec_in = 1   # 解码器输入维度
        self.c_out = 1    # 输出维度
        self.factor = 5   # probsparse注意力因子
        self.d_model = 512  # 模型维度
        self.n_heads = 8    # 多头注意力头数
        self.e_layers = 2   # 编码器层数
        self.d_layers = 1   # 解码器层数
        self.d_ff = 2048    # 前馈网络维度
        
        # 训练参数
        self.dropout = 0.05     # dropout率
        self.attn = 'prob'      # 注意力机制类型
        self.embed = 'timeF'    # 时间特征编码
        self.activation = 'gelu'  # 激活函数
        self.distil = True      # 是否使用蒸馏
        self.output_attention = False  # 是否输出注意力权重
        
        self.batch_size = 32  # 批次大小
        self.learning_rate = 0.0001  # 学习率
        self.epochs = 100  # 训练轮数
        self.patience = 10  # 早停耐心值
        self.lradj = 'type1'  # 学习率调整策略
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def to_dict(self):
        """将参数转换为字典，用于MLflow日志记录"""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('__') and not callable(value)}


def train_model_for_city(city_power, args, return_val_metrics=False):
    """
    为指定城市训练风电预测模型
    
    参数:
        city_power: 城市名称
        args: 模型参数
        return_val_metrics: 是否返回验证集指标（用于超参数优化）
        
    返回:
        如果return_val_metrics为True，返回验证集损失和评估指标，否则无返回值
    """
    global USE_MLFLOW
    
    CITY_FOR_POWER_DATA = city_power
    CITY_FOR_WEATHER_DATA = CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA]
    MODEL_PATH = os.path.join(MODEL_DIR, f'informer_model_{CITY_FOR_POWER_DATA}.pth')
    SCALER_PATH = os.path.join(MODEL_DIR, f'scalers_{CITY_FOR_POWER_DATA}.pkl')
    
    logger.info(f"\n开始训练城市: {CITY_FOR_POWER_DATA} 的风力发电预测模型...")
    
    # 1. 数据准备
    weather_df = get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
    power_df = get_history_wind_power_for_city(CITY_FOR_POWER_DATA)
    
    if weather_df.empty or power_df.empty:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 缺少天气或电力数据, 跳过训练")
        return float('inf') if return_val_metrics else None
    
    merged_df = merge_weather_and_power_df(weather_df, power_df)
    if merged_df.empty:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 合并数据为空, 跳过训练")
        return float('inf') if return_val_metrics else None
        
    preprocessed_df = preprocess_data(merged_df, CITY_FOR_POWER_DATA)
    time_wise_df = set_time_wise_feature(preprocessed_df.copy())
    
    # 确认训练数据量足够
    if len(time_wise_df) < args.seq_len + args.pred_len:
        logger.warning(f"警告: {CITY_FOR_POWER_DATA} 数据量不足，需要至少 {args.seq_len + args.pred_len} 条记录，但只有 {len(time_wise_df)} 条")
        return float('inf') if return_val_metrics else None
    
    # 2. 创建训练和测试数据集
    x_data, y_data, timestamps, scalers = create_datasets(
        time_wise_df, 
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        target_col='wind_output',
        scale=True
    )
    
    # 更新编码器输入维度
    args.enc_in = x_data.shape[2]
    
    # 保存缩放器以便后续预测使用
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    
    # 划分训练集和测试集
    train_indices, test_indices = train_test_split(
        np.arange(len(x_data)), 
        test_size=0.2,
        shuffle=False
    )
    
    # 划分验证集
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.25,  # 占训练集的25%
        shuffle=False
    )
    
    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_data[train_indices])
    y_train = torch.FloatTensor(y_data[train_indices])
    x_val = torch.FloatTensor(x_data[val_indices])
    y_val = torch.FloatTensor(y_data[val_indices])
    x_test = torch.FloatTensor(x_data[test_indices])
    y_test = torch.FloatTensor(y_data[test_indices])
    
    # 创建数据集和加载器
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"{CITY_FOR_POWER_DATA} 数据集分割完成: 训练集 {len(train_indices)}个样本, "
                f"验证集 {len(val_indices)}个样本, 测试集 {len(test_indices)}个样本")
    
    # 3. 初始化模型
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
    
    # 4. 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 5. 早停机制
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # 6. 训练模型
    train_losses = []
    val_losses = []
    
    try:
        with mlflow.start_run(run_name=f"informer_train_{CITY_FOR_POWER_DATA}") if USE_MLFLOW else nullcontext():
            # 记录超参数
            if USE_MLFLOW:
                mlflow.log_params(args.to_dict())
                
            # 开始训练循环
            best_val_loss = float('inf')
            for epoch in range(1, args.epochs + 1):
                # 学习率调整
                optimizer = adjust_learning_rate(optimizer, epoch, args)
                
                model.train()
                train_loss = 0
                for i, (batch_x, batch_y) in enumerate(train_loader):
                    batch_x = batch_x.to(args.device)
                    batch_y = batch_y.to(args.device)
                    
                    # 获取对应索引的y用于解码器输入
                    dec_inp = batch_y[:, :args.label_len, :]
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    if args.output_attention:
                        outputs, _ = model(batch_x, dec_inp)
                    else:
                        outputs = model(batch_x, dec_inp)
                        
                    # 计算损失
                    # 只关注预测部分的输出
                    loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                    
                    # 反向传播
                    loss.backward()
                    
                    # 更新参数
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 验证
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for i, (batch_x, batch_y) in enumerate(val_loader):
                        batch_x = batch_x.to(args.device)
                        batch_y = batch_y.to(args.device)
                        
                        dec_inp = batch_y[:, :args.label_len, :]
                        
                        if args.output_attention:
                            outputs, _ = model(batch_x, dec_inp)
                        else:
                            outputs = model(batch_x, dec_inp)
                            
                        loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                        val_loss += loss.item()
                
                # 计算平均损失
                train_loss = train_loss / len(train_loader)
                val_loss = val_loss / len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # 记录最佳验证损失
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # 记录日志
                logger.info(f"Epoch {epoch}/{args.epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
                
                if USE_MLFLOW:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                # 早停检查
                early_stopping(val_loss, model, path=MODEL_PATH)
                if early_stopping.early_stop:
                    logger.info("早停触发，停止训练!")
                    break
            
            # 7. 绘制训练过程损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.title(f'{CITY_FOR_POWER_DATA} - 训练与验证损失')
            plt.legend()
            plt.grid(True)
            
            figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            loss_plot_path = os.path.join(figures_dir, f'{CITY_FOR_POWER_DATA}_training_loss.png')
            plt.savefig(loss_plot_path)
            
            if USE_MLFLOW:
                mlflow.log_artifact(loss_plot_path)
                
            # 8. 加载最佳模型进行测试评估
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            
            # 收集所有预测结果
            all_y_true = []
            all_y_pred = []
            test_timestamps = []
            
            with torch.no_grad():
                for i, (batch_x, batch_y) in enumerate(test_loader):
                    batch_x = batch_x.to(args.device)
                    batch_y = batch_y.to(args.device)
                    
                    dec_inp = batch_y[:, :args.label_len, :]
                    
                    if args.output_attention:
                        outputs, _ = model(batch_x, dec_inp)
                    else:
                        outputs = model(batch_x, dec_inp)
                    
                    # 收集真实值和预测值
                    pred = outputs.detach().cpu().numpy()
                    true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
                    
                    all_y_pred.append(pred)
                    all_y_true.append(true)
                    
                    # 收集对应的时间戳
                    for idx in range(len(batch_x)):
                        batch_idx = test_indices[i * args.batch_size + idx] if i * args.batch_size + idx < len(test_indices) else -1
                        if batch_idx >= 0:
                            test_timestamps.append(timestamps[batch_idx][-args.pred_len:])
            
            # 合并批次结果
            all_y_true = np.vstack([y.reshape(-1, args.pred_len, args.c_out) for y in all_y_true])
            all_y_pred = np.vstack([y.reshape(-1, args.pred_len, args.c_out) for y in all_y_pred])
            
            # 反标准化预测结果
            all_y_true_rescaled = scalers['target_scaler'].inverse_transform(all_y_true.reshape(-1, args.c_out)).reshape(-1, args.pred_len, args.c_out)
            all_y_pred_rescaled = scalers['target_scaler'].inverse_transform(all_y_pred.reshape(-1, args.c_out)).reshape(-1, args.pred_len, args.c_out)
            
            # 9. 评估测试集性能
            test_true = all_y_true_rescaled.reshape(-1)
            test_pred = all_y_pred_rescaled.reshape(-1)
            
            rmse = np.sqrt(mean_squared_error(test_true, test_pred))
            mae = mean_absolute_error(test_true, test_pred)
            r2 = r2_score(test_true, test_pred)
            
            logger.info(f"{CITY_FOR_POWER_DATA} 测试集评估结果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
            
            if USE_MLFLOW:
                mlflow.log_metric("test_rmse", rmse)
                mlflow.log_metric("test_mae", mae)
                mlflow.log_metric("test_r2", r2)
                
            # 10. 绘制测试集预测结果
            # 选择第一个样本的结果进行可视化
            sample_idx = 0
            sample_true = all_y_true_rescaled[sample_idx, :, 0]
            sample_pred = all_y_pred_rescaled[sample_idx, :, 0]
            sample_time = test_timestamps[sample_idx] if sample_idx < len(test_timestamps) else None
            
            if sample_time is not None:
                fig = plot_predictions(
                    sample_true, sample_pred, sample_time,
                    title=f'{CITY_FOR_POWER_DATA} - 测试集预测结果示例',
                    save_path=os.path.join(figures_dir, f'{CITY_FOR_POWER_DATA}_test_prediction.png'),
                    rmse=rmse, mae=mae, r2=r2
                )
                
                if USE_MLFLOW:
                    mlflow.log_artifact(os.path.join(figures_dir, f'{CITY_FOR_POWER_DATA}_test_prediction.png'))
            
            logger.info(f"{CITY_FOR_POWER_DATA} 的模型训练与评估全部完成，最佳模型已保存到 {MODEL_PATH}")
            
            # 如果是超参数优化，返回需要优化的指标
            if return_val_metrics:
                return best_val_loss
            
    except Exception as e:
        logger.error(f"训练 {CITY_FOR_POWER_DATA} 的模型时发生错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        if return_val_metrics:
            return float('inf')


def objective(trial, city_power):
    """
    Optuna优化目标函数，为给定的城市优化超参数
    
    参数:
        trial: Optuna trial对象
        city_power: 城市名称
        
    返回:
        验证集损失值
    """
    # 创建一个新的参数集
    args = Args()
    
    # 定义要搜索的超参数空间
    args.d_model = trial.suggest_categorical('d_model', [128, 256, 512, 1024])
    args.n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])
    args.e_layers = trial.suggest_int('e_layers', 1, 3)
    args.d_layers = trial.suggest_int('d_layers', 1, 3)
    args.d_ff = trial.suggest_categorical('d_ff', [1024, 2048, 4096])
    args.factor = trial.suggest_int('factor', 3, 7)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.3)
    args.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # 使用要优化的超参数训练模型
    val_loss = train_model_for_city(city_power, args, return_val_metrics=True)
    
    return val_loss


def hyperparameter_search_for_city(city_power, n_trials=50):
    """
    为指定城市进行超参数搜索
    
    参数:
        city_power: 城市名称
        n_trials: Optuna试验次数
    """
    logger.info(f"开始为城市 {city_power} 进行超参数搜索...")
    
    # 创建Optuna学习记录器以将试验同步到MLflow
    if USE_MLFLOW:
        # 启动MLflow顶级运行以存储优化过程
        with mlflow.start_run(run_name=f"informer_hyperparameter_search_{city_power}"):
            mlflow.log_param("city", city_power)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("optimization_method", "Optuna TPE Sampler")
            
            # 创建Optuna学习
            study_name = f"informer_study_{city_power}"
            storage_name = "sqlite:///optuna_studies.db"
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                load_if_exists=True,
                direction='minimize',
                sampler=TPESampler(seed=42)
            )
            
            # 自定义回调函数记录到MLflow
            def mlflow_callback(study, trial):
                trial_value = trial.value if trial.value is not None else float("nan")
                
                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    mlflow.log_params(trial.params)
                    mlflow.log_metric("val_loss", trial_value)
            
            # 运行优化
            study.optimize(
                lambda trial: objective(trial, city_power),
                n_trials=n_trials,
                callbacks=[mlflow_callback]
            )
            
            # 记录最佳超参数
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_val_loss", study.best_value)
            
            # 生成优化历史图并记录
            try:
                figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
                os.makedirs(figures_dir, exist_ok=True)
                
                # 优化历史图
                history_fig = optuna.visualization.plot_optimization_history(study)
                history_path = os.path.join(figures_dir, f"{city_power}_optuna_history.png")
                history_fig.write_image(history_path)
                mlflow.log_artifact(history_path)
                
                # 参数重要性图
                param_imp_fig = optuna.visualization.plot_param_importances(study)
                param_imp_path = os.path.join(figures_dir, f"{city_power}_param_importance.png")
                param_imp_fig.write_image(param_imp_path)
                mlflow.log_artifact(param_imp_path)
                
                # 并行坐标图
                parallel_fig = optuna.visualization.plot_parallel_coordinate(study)
                parallel_path = os.path.join(figures_dir, f"{city_power}_parallel_coordinate.png")
                parallel_fig.write_image(parallel_path)
                mlflow.log_artifact(parallel_path)
                
            except Exception as e:
                logger.warning(f"生成Optuna可视化图表时出错: {e}")
                
            # 保存最佳超参数
            best_params_path = os.path.join(MODEL_DIR, f"best_params_{city_power}.json")
            with open(best_params_path, 'w') as f:
                json.dump(study.best_params, f, indent=4)
            mlflow.log_artifact(best_params_path)
            
            # 使用最佳超参数训练最终模型
            logger.info(f"使用最佳超参数为 {city_power} 训练最终模型")
            best_args = Args()
            for param_name, param_value in study.best_params.items():
                setattr(best_args, param_name, param_value)
            
            # 使用最佳参数训练最终模型
            train_model_for_city(city_power, best_args)
            
    else:
        # 如果MLflow不可用，仍然运行Optuna但不记录到MLflow
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(
            lambda trial: objective(trial, city_power),
            n_trials=n_trials
        )
        
        logger.info(f"最佳超参数: {study.best_params}")
        logger.info(f"最佳验证损失: {study.best_value}")
        
        # 保存最佳超参数
        best_params_path = os.path.join(MODEL_DIR, f"best_params_{city_power}.json")
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
            
        # 使用最佳超参数训练最终模型
        logger.info(f"使用最佳超参数为 {city_power} 训练最终模型")
        best_args = Args()
        for param_name, param_value in study.best_params.items():
            setattr(best_args, param_name, param_value)
        
        # 使用最佳参数训练最终模型
        train_model_for_city(city_power, best_args)


class nullcontext:
    """替代Python 3.7以下版本缺少的nullcontext"""
    def __enter__(self): return None
    def __exit__(self, exc_type, exc_val, exc_tb): pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Informer模型训练脚本，支持超参数优化')
    parser.add_argument('--optuna', action='store_true', help='启用Optuna超参数优化')
    parser.add_argument('--trials', type=int, default=50, help='Optuna优化尝试次数')
    args_cmd = parser.parse_args()
    
    if args_cmd.optuna:
        logger.info(f"启用Optuna超参数优化，将为以下城市搜索最佳超参数: {CITIES_FOR_POWER}")
        for city in CITIES_FOR_POWER:
            try:
                hyperparameter_search_for_city(city, n_trials=args_cmd.trials)
            except Exception as e:
                logger.error(f"为城市 {city} 进行超参数优化时发生错误: {e}")
                continue
    else:
        default_args = Args()
        logger.info(f"将使用默认超参数为以下城市训练模型: {CITIES_FOR_POWER}")
        for city in CITIES_FOR_POWER:
            try:
                train_model_for_city(city, default_args)
            except Exception as e:
                logger.error(f"训练城市 {city} 时发生错误: {e}")
                continue
    
    logger.info("所有城市的模型训练完成!") 