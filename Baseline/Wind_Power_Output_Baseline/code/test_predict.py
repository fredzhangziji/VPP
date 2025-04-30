"""
测试某一天的预测表现和实际表现的程序入口。
"""

from Baseline.Wind_Power_Output_Baseline.code import model_tool
from pub_tools import const

if __name__ == '__main__':
    cities = const.REGIONS_FOR_WEATHER

    province_actual, province_forecast, province_metrics = model_tool.predict_province_day(cities, day_index=-80)