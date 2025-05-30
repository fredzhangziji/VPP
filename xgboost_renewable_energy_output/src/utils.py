# 工具函数模块
# 你可以在这里添加一些通用的辅助函数，例如：
# - 日志记录配置
# - 特殊的绘图函数
# - 参数解析等

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_feature_importance(model, feature_names, savefig=False, filename='feature_importance.png'):
    """绘制 XGBoost 模型的特征重要性"""
    if not hasattr(model, 'feature_importances_'):
        print("模型没有 feature_importances_ 属性。请确保模型已训练且是合适的类型。")
        return
    
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(16, 12))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('XGBoost feature importance')
    plt.tight_layout()
    
    if savefig:
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, filename)
        plt.savefig(save_path)
        print(f"特征重要性图已保存到 {save_path}")

def plot_predictions_vs_actual(y_true, y_pred, title='预测值 vs 真实值', savefig=False, filename='predictions_vs_actual.png'):
    """绘制预测值与真实值的对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='真实值', marker='.', linestyle='-')
    plt.plot(y_pred, label='预测值', marker='.', linestyle='--')
    plt.title(title)
    plt.xlabel('样本索引')
    plt.ylabel('风力发电输出')
    plt.legend()
    plt.grid(True)
    
    if savefig:
        figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        save_path = os.path.join(figures_dir, filename)
        plt.savefig(save_path)
        print(f"预测对比图已保存到 {save_path}")