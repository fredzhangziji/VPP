"""
此模块用于读取每个城市的XGBoost风电预测模型，并生成特征重要性图表。
通过可视化各特征对模型预测的影响程度，帮助分析不同城市风电预测中的关键因素。
"""

import os
import xgboost as xgb
from utils import plot_feature_importance
from data_preprocessing import CITIES_FOR_POWER, CITY_NAME_MAPPING_DICT

import logging
import pub_tools.logging_config
logger = logging.getLogger('xgboost')

def load_model_and_features(city):
    """加载指定城市的模型和特征列表
    
    入参:
        city: 城市名称
        
    返回:
        model: 加载的XGBoost模型
        features: 模型使用的特征名列表
    """
    model_dir = 'models'
    model_path = os.path.join(model_dir, f'xgb_best_model_{city}.json')
    feature_path = os.path.join(model_dir, f'trained_features_{city}.txt')
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        logger.warning(f"找不到城市 {city} 的模型文件: {model_path}")
        return None, None
    
    if not os.path.exists(feature_path):
        logger.warning(f"找不到城市 {city} 的特征文件: {feature_path}")
        return None, None
    
    # 加载模型
    try:
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        logger.info(f"成功加载城市 {city} 的模型")
    except Exception as e:
        logger.error(f"加载城市 {city} 的模型时出错: {e}")
        return None, None
    
    # 加载特征名
    try:
        with open(feature_path, 'r') as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"成功加载城市 {city} 的特征列表，共 {len(features)} 个特征")
    except Exception as e:
        logger.error(f"加载城市 {city} 的特征列表时出错: {e}")
        return model, None
    
    return model, features

def generate_feature_importance_plots():
    """为所有城市生成特征重要性图表"""
    logger.info("开始生成各城市的特征重要性图表")
    
    # 确保figures目录存在
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # 遍历所有城市
    for city in CITIES_FOR_POWER:
        logger.info(f"处理城市: {city}")
        
        # 加载模型和特征
        model, features = load_model_and_features(city)
        
        if model is not None and features is not None:
            # 生成特征重要性图表
            filename = f"{city}_feature_importance_final.png"
            try:
                plot_feature_importance(model, features, savefig=True, filename=filename)
                logger.info(f"成功为城市 {city} 生成特征重要性图表")
            except Exception as e:
                logger.error(f"为城市 {city} 生成特征重要性图表时出错: {e}")
        else:
            logger.warning(f"由于无法加载模型或特征，跳过城市 {city} 的特征重要性图表生成")
    
    logger.info("所有城市的特征重要性图表生成完成")

if __name__ == '__main__':
    generate_feature_importance_plots() 