"""
工具函数模块
包含Informer模型所需的各种辅助函数，例如数据处理、可视化、评估指标等
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pub_tools import get_system_font_path
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import logging
import pub_tools.logging_config
logger = logging.getLogger('informer')


def setup_chinese_font():
    """设置matplotlib中文字体"""
    font_path = get_system_font_path()
    if font_path is not None:
        plt.rcParams['font.sans-serif'] = [fm.FontProperties(fname=font_path).get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"已设置matplotlib字体: {font_path}")
        return True
    else:
        logger.warning("未找到可用的中文字体，继续使用默认字体。")
        return False


def plot_predictions(y_true, y_pred, time_index, title, save_path=None, rmse=None, mae=None, r2=None):
    """
    绘制预测结果与真实值的对比图
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        time_index: 时间索引
        title: 图表标题
        save_path: 保存路径
        rmse: 均方根误差
        mae: 平均绝对误差
        r2: 决定系数
    """
    plt.figure(figsize=(15, 7))
    plt.plot(time_index, y_true, label='真实值', marker='.', alpha=0.7)
    plt.plot(time_index, y_pred, label='预测值', marker='.', alpha=0.7)
    
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


def evaluate_model(y_true, y_pred):
    """
    评估模型性能
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        
    返回:
        包含各评估指标的字典
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def forecast_accuracy(y_pred, y_true):
    """
    计算预测的准确性指标
    
    参数:
        y_pred: 预测值 numpy array
        y_true: 实际值 numpy array
        
    返回:
        包含各评估指标的字典
    """
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        'mape': mape, 
        'rmse': rmse,
        'mae': mae
    }


def adjust_learning_rate(optimizer, epoch, args):
    """
    根据epoch调整学习率
    
    参数:
        optimizer: PyTorch优化器
        epoch: 当前epoch数
        args: 参数配置
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        # 默认调整策略
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    if epoch in lr_adjust:
        logger.info(f'调整学习率为: {lr_adjust[epoch]}')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adjust[epoch]
    
    return optimizer


class EarlyStopping:
    """
    早停机制，避免过拟合
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
    def __call__(self, val_loss, model, path):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path):
        """保存模型检查点"""
        if self.verbose:
            logger.info(f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f})，保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss 