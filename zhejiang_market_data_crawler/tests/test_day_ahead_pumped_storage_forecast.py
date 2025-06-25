#!/usr/bin/env python3
"""
测试日抽蓄总出力预测爬虫
"""

import asyncio
import pandas as pd
import traceback
from utils.logger import setup_logger
from utils.config import DB_CONFIG
from pub_tools.db_tools import get_db_connection, release_db_connection
from crawlers.day_ahead_pumped_storage_forecast_crawler import DayAheadPumpedStorageForecastCrawler
from sqlalchemy import inspect

# 目标表名和字段名
TABLE_NAME = 'power_market_data'
FIELD_NAME = 'day_ahead_pumped_storage_forecast'

# 设置日志记录器
logger = setup_logger('test_day_ahead_pumped_storage_forecast')

def check_db_config():
    """
    检查数据库配置和表结构
    """
    logger.info("检查数据库配置")
    safe_config = DB_CONFIG.copy()
    if 'password' in safe_config:
        safe_config['password'] = '***隐藏***'
    logger.info(f"数据库配置: {safe_config}")
    logger.info(f"目标表名: {TABLE_NAME}")
    try:
        engine, _ = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        connection = engine.connect()
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

async def main():
    try:
        logger.info("===== 测试日抽蓄总出力预测爬虫 =====")
        # 检查数据库结构
        assert check_db_config(), "数据库结构检查未通过"
        # 测试单日数据
        test_date = '2025-06-02'
        crawler = DayAheadPumpedStorageForecastCrawler(target_table=TABLE_NAME)
        df = crawler.fetch_data(start_date=test_date, end_date=test_date)
        assert isinstance(df, pd.DataFrame)
        logger.info(f"单日数据采集成功，记录数: {len(df)}")
        # 测试区间数据
        start_date = '2025-06-02'
        end_date = '2025-06-03'
        df_range = crawler.fetch_data(start_date=start_date, end_date=end_date)
        assert isinstance(df_range, pd.DataFrame)
        logger.info(f"区间数据采集成功，记录数: {len(df_range)}")
        # 测试入库
        if not df_range.empty:
            result = crawler.save_to_db(df_range)
            assert result
            logger.info("数据入库成功")
        else:
            logger.warning("区间数据为空，跳过入库测试")
        logger.info("===== 日抽蓄总出力预测爬虫测试通过 =====")
    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 