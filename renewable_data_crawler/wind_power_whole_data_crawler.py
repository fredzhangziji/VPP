"""
内蒙全量风电出力的数据爬取程序入口。
"""

from renewable_data_crawler import crawler_tools
from pub_tools import const
from datetime import date
import argparse
import time

import logging
import pub_tools.logging_config
logger = logging.getLogger(__name__)

if __name__ == '__main__': 
    logger.info("开始全量光伏数据爬取程序...")
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Wind Power Data Crawler")
    parser.add_argument(
        "--start_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="起始日期，格式为 YYYY-MM-DD，默认为今天的日期"
    )

    args = parser.parse_args()
    start_date = args.start_date

    crawler_tools.fetch_multi_day_wind_power_data_for_each_city(const.NEIMENG_RENEWABLE_ENERGY_URL,
                                                          start_date=start_date)
    
    end_time = time.time()
    elapsed = end_time - start_time
    logger.info(f"全量风力数据爬取开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    logger.info(f"全量风力数据爬取结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    logger.info(f"总共花费时间: {elapsed:.2f} 秒")