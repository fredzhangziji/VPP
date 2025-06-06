"""
风电出力Baseline对于真实情况的预测入口脚本
"""

import model_tool
import numpy as np
import pandas as pd
from pub_tools import const
from datetime import date, timedelta

if __name__ == '__main__':
    n_days = 5
    future_features_dict = {}
    hours = n_days * 24

    start_date = date.today()
    end_date = start_date + timedelta(days=5)
    start_date = str(start_date)
    end_date = str(end_date)

    cities_for_prediction = [city for city in const.REGIONS_FOR_WEATHER if city != '乌海市']

    for city in cities_for_prediction:
        df_city_normalized = model_tool.get_predict_weather_data_for_city(city=city, start_date=start_date, end_date=end_date)
        future_features_dict[city] = df_city_normalized

    province_future_forecast = model_tool.predict_province_future(cities_for_prediction, future_features_dict, start_date, n_days)