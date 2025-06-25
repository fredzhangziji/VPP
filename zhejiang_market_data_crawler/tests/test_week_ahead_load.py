#!/usr/bin/env python3
"""
测试周前负荷预测爬虫
"""

import asyncio
import pandas as pd
import traceback
from sqlalchemy import inspect, Column, DateTime, Float, MetaData, Table
from utils.logger import setup_logger
from utils.config import DB_CONFIG
from pub_tools.db_tools import get_db_connection, release_db_connection
from crawlers.week_ahead_load_crawler import WeekAheadLoadCrawler

# 目标表名和字段名
TABLE_NAME = 'power_market_data'
FIELD_NAME = 'week_ahead_load_forecast'

# 是否使用模拟数据（在API授权失败时可以使用）
USE_MOCK_DATA = False

# 设置日志记录器
logger = setup_logger('test_week_ahead_load')

def check_db_config():
    """检查数据库配置是否正确"""
    logger.info("检查数据库配置")
    logger.info(f"数据库配置: {DB_CONFIG}")
    logger.info(f"目标表名: {TABLE_NAME}")
    
    try:
        # 尝试连接数据库
        engine, _ = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        
        # 检查表是否存在
        connection = engine.connect()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if TABLE_NAME in tables:
            logger.info(f"表 {TABLE_NAME} 存在")
            
            # 检查表结构
            columns = inspector.get_columns(TABLE_NAME)
            logger.info(f"表 {TABLE_NAME} 的列: {[col['name'] for col in columns]}")
            
            # 检查是否有date_time和week_ahead_load_forecast列
            column_names = [col['name'] for col in columns]
            if 'date_time' in column_names and FIELD_NAME in column_names:
                logger.info("表结构正确，包含必要的列")
            else:
                logger.error(f"表结构缺少必要的列，需要date_time和{FIELD_NAME}列")
        else:
            logger.error(f"表 {TABLE_NAME} 不存在，需要创建")
            
            # 尝试创建表
            metadata = MetaData()
            table_columns = [
                Column('date_time', DateTime, primary_key=True)
            ]
            table_columns.append(Column(FIELD_NAME, Float))
            
            Table(
                TABLE_NAME, 
                metadata,
                *table_columns
            )
            
            # 将表写入数据库
            metadata.create_all(engine)
            logger.info(f"已创建表 {TABLE_NAME}")
        
        # 关闭连接
        connection.close()
        release_db_connection(engine)
        
        return True
    except Exception as e:
        logger.error(f"数据库连接或检查失败: {e}")
        traceback.print_exc()
        return False

async def test_single_date():
    """测试爬取单个日期的数据"""
    logger.info("测试爬取单个日期的数据")
    
    # 使用固定的日期：2025-06-02
    test_date = "2025-06-02"
    
    try:
        # 创建爬虫实例
        crawler = WeekAheadLoadCrawler(target_table=TABLE_NAME, field_name=FIELD_NAME)
        
        # 获取数据
        df = crawler.fetch_data(test_date)
        
        if not df.empty:
            logger.info(f"成功爬取 {test_date} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            logger.info(f"负荷预测值范围: {df[FIELD_NAME].min()} 至 {df[FIELD_NAME].max()}")
            
            # 保存到数据库
            success = crawler.save_to_db(df, update_columns=[FIELD_NAME])
            
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
    """测试爬取日期范围的数据"""
    logger.info("测试爬取日期范围的数据")
    
    # 使用固定的日期范围：2025-06-02 到 2025-06-03
    start_date_str = "2025-06-02"
    end_date_str = "2025-06-03"
    
    try:
        # 创建爬虫实例
        crawler = WeekAheadLoadCrawler(target_table=TABLE_NAME, field_name=FIELD_NAME)
        
        # 获取数据
        df = crawler.fetch_data(start_date_str, end_date_str)
        
        if not df.empty:
            logger.info(f"成功爬取 {start_date_str} 至 {end_date_str} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            
            # 检查日期和值的范围
            min_date = df['date_time'].min().strftime('%Y-%m-%d %H:%M:%S')
            max_date = df['date_time'].max().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"日期范围: {min_date} 至 {max_date}")
            logger.info(f"负荷预测值范围: {df[FIELD_NAME].min()} 至 {df[FIELD_NAME].max()}")
            
            # 获取数据点数量按日期统计
            dates = pd.to_datetime(df['date_time']).dt.date
            date_counts = df.groupby(dates).size()
            logger.info(f"按日期统计的数据点数量:")
            for date, count in date_counts.items():
                logger.info(f"  {date}: {count} 条记录")
            
            # 保存到数据库
            success = crawler.save_to_db(df, update_columns=[FIELD_NAME])
            
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
    """主函数"""
    logger.info("开始测试周前负荷预测爬虫")
    
    # 首先检查数据库配置
    if not check_db_config():
        logger.error("数据库配置检查失败，请检查配置后重试")
        return
    
    # 运行多天数据测试
    logger.info("运行多天数据测试")
    await test_date_range()
    
    logger.info("测试完成")

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main()) 