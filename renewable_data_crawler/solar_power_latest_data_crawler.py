"""
内蒙每日光伏出力的数据爬取程序入口。
"""

import crawler_tools
from pub_tools import const
from datetime import date
import pandas as pd
import time

import logging
import pub_tools.logging_config
logger = logging.getLogger(__name__)

if __name__ == '__main__': 
    logger.info("开始每日光伏数据爬取程序...")
    start_time = time.time()

    today_date = pd.to_datetime(date.today(), format="%Y-%m-%d")
    start_date = (today_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (today_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    crawler_tools.fetch_multi_day_solar_power_data_for_each_city(const.NEIMENG_RENEWABLE_ENERGY_URL,
                                                                 start_date,
                                                                 end_date)
    
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"每日光伏数据爬取开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logger.info(f"每日光伏数据爬取结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logger.info(f"总共花费时间: {elapsed:.2f} 秒")