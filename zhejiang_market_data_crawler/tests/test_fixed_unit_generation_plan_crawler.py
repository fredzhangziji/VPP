#!/usr/bin/env python3
"""
测试固定出力机组发电计划爬虫
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
from crawlers.fixed_unit_generation_plan_crawler import (
    FixedUnitGenerationPlanCrawler, 
    crawl_fixed_unit_generation_plan_for_date,
    run_historical_crawl,
    run_daily_crawl
)

# 目标表名
TABLE_NAME = 'fixed_unit_generation_plan'
UNITS_TABLE_NAME = 'fixed_generation_units'
TOTAL_TABLE_NAME = 'total_fixed_generation_plan'

# 设置日志记录器
logger = setup_logger('test_fixed_unit_generation_plan')

def check_db_config():
    """检查数据库配置"""
    logger.info("检查数据库配置")
    
    # 隐藏密码
    safe_config = DB_CONFIG.copy()
    if 'password' in safe_config:
        safe_config['password'] = '***隐藏***'
    
    logger.info(f"数据库配置: {safe_config}")
    logger.info(f"目标表名: {TABLE_NAME}, {UNITS_TABLE_NAME}, {TOTAL_TABLE_NAME}")
    
    try:
        # 尝试连接数据库
        engine, metadata = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        
        # 检查表是否存在
        connection = engine.connect()
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        all_tables_exist = True
        
        # 检查机组表
        if UNITS_TABLE_NAME in tables:
            logger.info(f"表 {UNITS_TABLE_NAME} 存在")
            
            # 检查表结构
            columns = inspector.get_columns(UNITS_TABLE_NAME)
            column_names = [col['name'] for col in columns]
            logger.info(f"表 {UNITS_TABLE_NAME} 的列: {column_names}")
            
            # 检查必要的列是否存在
            required_columns = ['unit_id', 'unit_name']
            
            missing_columns = []
            for col in required_columns:
                if col not in column_names:
                    missing_columns.append(col)
            
            if not missing_columns:
                logger.info(f"表 {UNITS_TABLE_NAME} 结构正确，包含必要的列")
            else:
                logger.warning(f"表 {UNITS_TABLE_NAME} 结构缺少以下列: {missing_columns}")
                all_tables_exist = False
        else:
            logger.error(f"表 {UNITS_TABLE_NAME} 不存在")
            all_tables_exist = False
        
        # 检查计划表
        if TABLE_NAME in tables:
            logger.info(f"表 {TABLE_NAME} 存在")
            
            # 检查表结构
            columns = inspector.get_columns(TABLE_NAME)
            column_names = [col['name'] for col in columns]
            logger.info(f"表 {TABLE_NAME} 的列: {column_names}")
            
            # 检查必要的列是否存在
            required_columns = ['unit_id', 'date_time', 'generation_plan']
            
            missing_columns = []
            for col in required_columns:
                if col not in column_names:
                    missing_columns.append(col)
            
            if not missing_columns:
                logger.info(f"表 {TABLE_NAME} 结构正确，包含必要的列")
            else:
                logger.warning(f"表 {TABLE_NAME} 结构缺少以下列: {missing_columns}")
                all_tables_exist = False
        else:
            logger.error(f"表 {TABLE_NAME} 不存在")
            all_tables_exist = False
        
        # 检查总计划表
        if TOTAL_TABLE_NAME in tables:
            logger.info(f"表 {TOTAL_TABLE_NAME} 存在")
            
            # 检查表结构
            columns = inspector.get_columns(TOTAL_TABLE_NAME)
            column_names = [col['name'] for col in columns]
            logger.info(f"表 {TOTAL_TABLE_NAME} 的列: {column_names}")
            
            # 检查必要的列是否存在
            required_columns = ['date_time', 'total_generation_plan']
            
            missing_columns = []
            for col in required_columns:
                if col not in column_names:
                    missing_columns.append(col)
            
            if not missing_columns:
                logger.info(f"表 {TOTAL_TABLE_NAME} 结构正确，包含必要的列")
            else:
                logger.warning(f"表 {TOTAL_TABLE_NAME} 结构缺少以下列: {missing_columns}")
                all_tables_exist = False
        else:
            logger.error(f"表 {TOTAL_TABLE_NAME} 不存在")
            all_tables_exist = False
        
        # 断开连接
        connection.close()
        release_db_connection(engine)
        return all_tables_exist
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        traceback.print_exc()
        return False

async def test_single_date():
    """测试爬取单个日期的数据"""
    logger.info("测试爬取单个日期的数据")
    
    # 使用固定的日期：2025-06-02
    # 为避免测试中使用当前日期，我们在测试中使用固定日期
    test_date = "2025-06-02"
    
    try:
        # 创建爬虫实例
        crawler = FixedUnitGenerationPlanCrawler(target_table=TABLE_NAME)
        
        # 获取数据
        df = crawler.fetch_data(test_date)
        
        if not df.empty:
            logger.info(f"成功爬取 {test_date} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            
            # 计算发电计划字段的最小值和最大值
            if 'generation_plan' in df.columns:
                logger.info(f"generation_plan 范围: {df['generation_plan'].min()} 至 {df['generation_plan'].max()}")
            
            # 获取不同机组的数量
            if 'unit_id' in df.columns:
                unit_count = df['unit_id'].nunique()
                logger.info(f"共有 {unit_count} 个不同的机组")
            
            return True
        else:
            logger.error(f"爬取 {test_date} 的数据失败")
            return False
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        traceback.print_exc()
        return False

async def test_date_range():
    """测试爬取日期范围的数据"""
    logger.info("测试爬取日期范围的数据")
    
    # 使用固定的日期范围：2025-06-02 到 2025-06-03
    start_date_str = "2025-06-02"
    end_date_str = "2025-06-03"
    
    try:
        # 创建爬虫实例
        crawler = FixedUnitGenerationPlanCrawler(target_table=TABLE_NAME)
        
        # 获取数据
        df = crawler.fetch_data(start_date_str, end_date_str)
        
        if not df.empty:
            logger.info(f"成功爬取 {start_date_str} 至 {end_date_str} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            
            # 检查日期范围
            # 使用date列代替date_time列
            min_date = pd.to_datetime(df['date'].min()).strftime('%Y-%m-%d')
            max_date = pd.to_datetime(df['date'].max()).strftime('%Y-%m-%d')
            logger.info(f"日期范围: {min_date} 至 {max_date}")
            
            # 计算发电计划字段的最小值和最大值，仅当该字段存在时
            if 'plan_points_count' in df.columns:
                logger.info(f"plan_points_count 范围: {df['plan_points_count'].min()} 至 {df['plan_points_count'].max()}")
            
            # 获取数据点数量按日期统计
            date_counts = df.groupby('date').size()
            logger.info(f"按日期统计的数据点数量:")
            for date, count in date_counts.items():
                logger.info(f"  {date}: {count} 条记录")
            
            # 获取不同机组的数量
            if 'units_count' in df.columns:
                units_count_sum = df['units_count'].sum()
                logger.info(f"共有 {units_count_sum} 个机组记录")
            
            return True
        else:
            logger.error(f"爬取 {start_date_str} 至 {end_date_str} 的数据失败")
            return False
    except Exception as e:
        logger.error(f"测试过程中发生异常: {e}")
        traceback.print_exc()
        return False

async def main():
    """主函数"""
    logger.info("开始测试固定出力机组发电计划爬虫")
    
    # 首先检查数据库配置
    if not check_db_config():
        logger.warning("数据库配置检查失败，但仍将继续测试爬虫功能")
    
    # 运行单日数据测试
    single_date_success = await test_single_date()
    
    # 运行多天数据测试
    date_range_success = await test_date_range()
    
    if single_date_success and date_range_success:
        logger.info("所有测试都通过了")
    else:
        logger.warning("部分测试失败")
    
    logger.info("测试完成")

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main()) 