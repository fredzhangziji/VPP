import tool
import numpy as np
import pandas as pd

cities = [
    '呼和浩特',
    '阿拉善盟']

n_days = 4
future_features_dict = {}
hours = n_days * 24

# FIXME: 修改为获取每个城市的天气数据
for city in cities:
    future_features = pd.DataFrame({
        't2m': np.random.uniform(-10, 35, hours),           # 温度
        'ws10m': np.random.uniform(0, 10, hours),           # 10米处风速
        'wd10m': np.random.uniform(0, 360, hours),          # 风向（度）
        'sp': np.random.uniform(900, 1050, hours)           # 气压，示例范围
    })
    # 正余弦转换风向，生成新特征
    future_features['wd10m_sin'] = np.sin(np.deg2rad(future_features['wd10m']))
    future_features['wd10m_cos'] = np.cos(np.deg2rad(future_features['wd10m']))
    # 选择并排列与训练一致的特征顺序
    future_features = future_features[['t2m', 'ws10m', 'wd10m_sin', 'wd10m_cos', 'sp']]
    future_features_dict[city] = future_features

province_future_forecast = tool.predict_province_future(cities, future_features_dict, n_days)