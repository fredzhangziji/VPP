#!/usr/bin/env python3
"""
测试非市场风电实时总出力爬虫
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
from crawlers.non_market_wind_output_crawler import (
    NonMarketWindOutputCrawler,
)

# 目标表名和字段名
TABLE_NAME = 'power_market_data'
FIELD_NAME = 'non_market_wind_output'

# 设置日志记录器
logger = setup_logger('test_non_market_wind_output')

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
        # 获取数据库连接
        conn = get_db_connection(DB_CONFIG)[0]
        if conn:
            logger.info("数据库连接成功")
            release_db_connection(conn)
            return True
        else:
            logger.error("数据库连接失败")
            return False
    except Exception as e:
        logger.error(f"数据库连接异常: {e}")
        return False


async def test_single_date():
    """
    测试爬取单个日期的数据
    """
    logger.info("测试爬取单个日期的数据")
    test_date = "2025-06-02"
    try:
        crawler = NonMarketWindOutputCrawler(target_table=TABLE_NAME)
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
    start_date = "2025-06-02"
    end_date = "2025-06-03"
    try:
        crawler = NonMarketWindOutputCrawler(target_table=TABLE_NAME)
        df = crawler.fetch_data(start_date, end_date)
        if not df.empty:
            logger.info(f"成功爬取 {start_date} 至 {end_date} 的数据，共 {len(df)} 条记录")
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            if FIELD_NAME in df.columns:
                logger.info(f"{FIELD_NAME} 范围: {df[FIELD_NAME].min()} 至 {df[FIELD_NAME].max()}")
            
            # 检查日期分布
            date_counts = df['date_time'].dt.date.value_counts().sort_index()
            logger.info(f"日期分布:\n{date_counts}")
            
            # 保存数据
            update_columns = [col for col in df.columns if col != 'date_time']
            success = crawler.save_to_db(df, update_columns=update_columns)
            if success:
                logger.info(f"成功将数据保存到表 {TABLE_NAME}")
            else:
                logger.error("保存数据失败")
        else:
            logger.error(f"爬取 {start_date} 至 {end_date} 的数据失败")
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        traceback.print_exc()


async def test_data_persistence():
    """
    测试数据持久化
    """
    logger.info("测试数据持久化")
    test_date = "2025-06-02"
    try:
        # 1. 获取数据
        crawler = NonMarketWindOutputCrawler(target_table=TABLE_NAME)
        df = crawler.fetch_data(test_date)
        if df.empty:
            logger.error(f"爬取 {test_date} 的数据失败，无法测试数据持久化")
            return
        
        # 2. 保存数据到数据库
        update_columns = [col for col in df.columns if col != 'date_time']
        success = crawler.save_to_db(df, update_columns=update_columns)
        if not success:
            logger.error("保存数据失败，无法测试数据持久化")
            return
        
        # 3. 从数据库读取数据进行验证
        from utils.db_helper import get_db_engine
        engine = get_db_engine()
        
        # 获取时间范围
        min_time = df['date_time'].min()
        max_time = df['date_time'].max()
        
        # 查询数据库
        query = f"""
        SELECT date_time, {FIELD_NAME}
        FROM {TABLE_NAME}
        WHERE date_time BETWEEN '{min_time}' AND '{max_time}'
        ORDER BY date_time
        """
        
        db_df = pd.read_sql(query, engine)
        engine.dispose()
        
        if db_df.empty:
            logger.error("从数据库读取数据失败")
            return
        
        # 4. 验证数据
        logger.info(f"从数据库读取到 {len(db_df)} 条记录")
        
        # 转换date_time列为datetime类型，以便比较
        df['date_time'] = pd.to_datetime(df['date_time'])
        db_df['date_time'] = pd.to_datetime(db_df['date_time'])
        
        # 按照date_time排序
        df = df.sort_values('date_time').reset_index(drop=True)
        db_df = db_df.sort_values('date_time').reset_index(drop=True)
        
        # 比较数据条数
        if len(df) == len(db_df):
            logger.info("数据条数匹配")
        else:
            logger.warning(f"数据条数不匹配: 爬取数据 {len(df)} 条，数据库数据 {len(db_df)} 条")
        
        # 比较日期范围
        if df['date_time'].min() == db_df['date_time'].min() and df['date_time'].max() == db_df['date_time'].max():
            logger.info("日期范围匹配")
        else:
            logger.warning(f"日期范围不匹配: 爬取数据 {df['date_time'].min()} 至 {df['date_time'].max()}，数据库数据 {db_df['date_time'].min()} 至 {db_df['date_time'].max()}")
        
        # 比较字段值
        if FIELD_NAME in df.columns and FIELD_NAME in db_df.columns:
            # 检查是否所有的日期时间都匹配
            date_time_matches = all(df['date_time'].eq(db_df['date_time']))
            if date_time_matches:
                logger.info("所有日期时间匹配")
            else:
                logger.warning("日期时间不完全匹配")
            
            # 检查非市场风电实时总出力值是否匹配
            # 由于浮点数比较可能有精度问题，使用近似比较
            value_matches = all(abs(df[FIELD_NAME] - db_df[FIELD_NAME]) < 0.01)
            if value_matches:
                logger.info("所有非市场风电实时总出力值匹配")
            else:
                logger.warning("非市场风电实时总出力值不完全匹配")
                
                # 找出不匹配的记录
                mismatch_idx = (abs(df[FIELD_NAME] - db_df[FIELD_NAME]) >= 0.01)
                mismatch_df = pd.DataFrame({
                    'date_time': df.loc[mismatch_idx, 'date_time'],
                    f'original_{FIELD_NAME}': df.loc[mismatch_idx, FIELD_NAME],
                    f'db_{FIELD_NAME}': db_df.loc[mismatch_idx, FIELD_NAME],
                    'diff': abs(df.loc[mismatch_idx, FIELD_NAME] - db_df.loc[mismatch_idx, FIELD_NAME])
                })
                logger.warning(f"不匹配的记录:\n{mismatch_df.head(5)}")
            
            # 总体验证结果
            if date_time_matches and value_matches:
                logger.info("数据持久化测试通过")
            else:
                logger.warning("数据持久化测试未完全通过")
        else:
            logger.error(f"字段缺失: 爬取数据列 {df.columns.tolist()}，数据库数据列 {db_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"测试数据持久化过程中发生异常: {e}")
        traceback.print_exc()

async def main():
    """
    主函数
    """
    logger.info("开始测试非市场风电实时总出力爬虫")
    if not check_db_config():
        logger.error("数据库配置检查失败，请检查配置后重试")
        return
    await test_single_date()
    await test_date_range()
    await test_data_persistence()
    logger.info("测试完成")

if __name__ == '__main__':
    asyncio.run(main())