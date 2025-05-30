"""
此脚本用于进行风电出力Baseline的训练。
"""

from Baseline.Wind_Power_Output_Baseline.code import model_tool
from pub_tools import const

if __name__ == '__main__':
    cities = const.REGIONS_FOR_WEATHER
    cities = [city for city in cities if city != '乌海市']

    model_tool.train(cities, 50)