#!/usr/bin/env python3
"""
测试非市场核电实时总出力爬虫
"""

import os
import sys
import asyncio
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from utils.config import DB_CONFIG, TARGET_TABLE
from pub_tools.db_tools import get_db_connection, release_db_connection
from crawlers.non_market_nuclear_output_crawler import (
    NonMarketNuclearOutputCrawler,
)

# 目标表名和字段名
TABLE_NAME = 'power_market_data'
FIELD_NAME = 'non_market_nuclear_output'

# 设置日志记录器
logger = setup_logger('test_non_market_nuclear_output')

def check_db_config():
    """
    检查数据库配置
    """
    logger.info("检查数据库配置")
    safe_config = DB_CONFIG.copy()
    if 'password' in safe_config:
        safe_config['password'] = '***隐藏***'
    logger.info(f"数据库配置: {safe_config}")
    logger.info(f"目标表名: {TABLE_NAME}")
    try:
        engine, metadata = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        connection = engine.connect()
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if TABLE_NAME in tables:
            logger.info(f"表 {TABLE_NAME} 存在")
            columns = inspector.get_columns(TABLE_NAME)
            column_names = [col['name'] for col in columns]
            logger.info(f"表 {TABLE_NAME} 的列: {column_names}")
            required_columns = ['date_time', FIELD_NAME]
            missing_columns = [col for col in required_columns if col not in column_names]
            if not missing_columns:
                logger.info("表结构正确，包含必要的列")
            else:
                logger.warning(f"表结构缺少以下列: {missing_columns}")
                return False
        else:
            logger.error(f"表 {TABLE_NAME} 不存在")
            return False
        connection.close()
        release_db_connection(engine)
        return True
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        traceback.print_exc()
        return False

async def test_single_date():
    """
    测试爬取单个日期的数据
    """
    logger.info("测试爬取单个日期的数据")
    test_date = "2025-06-02"
    try:
        crawler = NonMarketNuclearOutputCrawler(target_table=TABLE_NAME)
        df = crawler.fetch_data(test_date)
        if not df.empty:
            logger.info(f"成功爬取 {test_date} 的数据，共 {len(df)} 条记录")
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            if FIELD_NAME in df.columns:
                logger.info(f"{FIELD_NAME} 范围: {df[FIELD_NAME].min()} 至 {df[FIELD_NAME].max()}")
            update_columns = [col for col in df.columns if col != 'date_time']
            success = crawler.save_to_db(df, update_columns=update_columns)
            if success:
                logger.info(f"成功将数据保存到表 {TABLE_NAME}")
            else:
                logger.error("保存数据失败")
        else:
            logger.error(f"爬取 {test_date} 的数据失败")
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        traceback.print_exc()

async def test_date_range():
    """
    测试爬取日期范围的数据
    """
    logger.info("测试爬取日期范围的数据")
    start_date_str = "2025-06-02"
    end_date_str = "2025-06-03"
    try:
        crawler = NonMarketNuclearOutputCrawler(target_table=TABLE_NAME)
        df = crawler.fetch_data(start_date_str, end_date_str)
        if not df.empty:
            logger.info(f"成功爬取 {start_date_str} 至 {end_date_str} 的数据，共 {len(df)} 条记录")
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            min_date = df['date_time'].min().strftime('%Y-%m-%d %H:%M:%S')
            max_date = df['date_time'].max().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"日期范围: {min_date} 至 {max_date}")
            if FIELD_NAME in df.columns:
                logger.info(f"{FIELD_NAME} 范围: {df[FIELD_NAME].min()} 至 {df[FIELD_NAME].max()}")
            dates = pd.to_datetime(df['date_time']).dt.date
            date_counts = df.groupby(dates).size()
            logger.info(f"按日期统计的数据点数量:")
            for date, count in date_counts.items():
                logger.info(f"  {date}: {count} 条记录")
            update_columns = [col for col in df.columns if col != 'date_time']
            success = crawler.save_to_db(df, update_columns=update_columns)
            if success:
                logger.info(f"成功将数据保存到表 {TABLE_NAME}")
            else:
                logger.error("保存数据失败")
        else:
            logger.error(f"爬取 {start_date_str} 至 {end_date_str} 的数据失败")
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        traceback.print_exc()

async def main():
    """
    主函数
    """
    logger.info("开始测试非市场核电实时总出力爬虫")
    if not check_db_config():
        logger.error("数据库配置检查失败，请检查配置后重试")
        return
    await test_single_date()
    await test_date_range()
    logger.info("测试完成")

if __name__ == '__main__':
    asyncio.run(main()) 