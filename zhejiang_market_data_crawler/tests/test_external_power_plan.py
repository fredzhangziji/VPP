#!/usr/bin/env python3
"""
测试外来电受电计划爬虫
"""

import asyncio
import pandas as pd
import traceback
from utils.logger import setup_logger
from utils.config import DB_CONFIG
from pub_tools.db_tools import get_db_connection, release_db_connection
from crawlers.external_power_plan_crawler import ExternalPowerPlanCrawler
from sqlalchemy import inspect

# 目标表名和字段名
TABLE_NAME = 'power_market_data'

# 设置日志记录器
logger = setup_logger('test_external_power_plan')

def check_db_config():
    """检查数据库配置"""
    logger.info("检查数据库配置")
    
    # 隐藏密码
    safe_config = DB_CONFIG.copy()
    if 'password' in safe_config:
        safe_config['password'] = '***隐藏***'
    
    logger.info(f"数据库配置: {safe_config}")
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
            column_names = [col['name'] for col in columns]
            logger.info(f"表 {TABLE_NAME} 的列: {column_names}")
            
            # 检查必要的列是否存在
            required_columns = ['date_time', 'external_power_plan']
            
            missing_columns = []
            for col in required_columns:
                if col not in column_names:
                    missing_columns.append(col)
            
            if not missing_columns:
                logger.info("表结构正确，包含必要的列")
            else:
                logger.warning(f"表结构缺少以下列: {missing_columns}")
                return False
        else:
            logger.error(f"表 {TABLE_NAME} 不存在")
            return False
        
        # 断开连接
        connection.close()
        release_db_connection(engine)
        return True
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        traceback.print_exc()
        return False

async def test_single_date():
    """测试爬取单个日期的数据"""
    logger.info("测试爬取单个日期的数据")
    
    # 使用固定的日期：2025-06-02
    test_date = "2025-06-02"
    
    try:
        # 创建爬虫实例
        crawler = ExternalPowerPlanCrawler(target_table=TABLE_NAME)
        
        # 获取数据
        df = crawler.fetch_data(test_date)
        
        if not df.empty:
            logger.info(f"成功爬取 {test_date} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            
            # 计算外来电受电计划字段的最小值和最大值
            if 'external_power_plan' in df.columns:
                logger.info(f"external_power_plan 范围: {df['external_power_plan'].min()} 至 {df['external_power_plan'].max()}")
            
            # 保存到数据库
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
    """测试爬取日期范围的数据"""
    logger.info("测试爬取日期范围的数据")
    
    # 使用固定的日期范围：2025-06-02 到 2025-06-03
    start_date_str = "2025-06-02"
    end_date_str = "2025-06-03"
    
    try:
        # 创建爬虫实例
        crawler = ExternalPowerPlanCrawler(target_table=TABLE_NAME)
        
        # 获取数据
        df = crawler.fetch_data(start_date_str, end_date_str)
        
        if not df.empty:
            logger.info(f"成功爬取 {start_date_str} 至 {end_date_str} 的数据，共 {len(df)} 条记录")
            
            # 打印数据概览
            logger.info(f"数据概览:")
            logger.info(f"列名: {df.columns.tolist()}")
            logger.info(f"数据示例:\n{df.head(3)}")
            
            # 检查日期范围
            min_date = df['date_time'].min().strftime('%Y-%m-%d %H:%M:%S')
            max_date = df['date_time'].max().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"日期范围: {min_date} 至 {max_date}")
            
            # 计算外来电受电计划字段的最小值和最大值
            if 'external_power_plan' in df.columns:
                logger.info(f"external_power_plan 范围: {df['external_power_plan'].min()} 至 {df['external_power_plan'].max()}")
            
            # 获取数据点数量按日期统计
            dates = pd.to_datetime(df['date_time']).dt.date
            date_counts = df.groupby(dates).size()
            logger.info(f"按日期统计的数据点数量:")
            for date, count in date_counts.items():
                logger.info(f"  {date}: {count} 条记录")
            
            # 保存到数据库
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
    """主函数"""
    logger.info("开始测试外来电受电计划爬虫")
    
    # 首先检查数据库配置
    if not check_db_config():
        logger.error("数据库配置检查失败，请检查配置后重试")
        return
    
    # 运行单日数据测试
    await test_single_date()
    
    # 运行多天数据测试
    await test_date_range()
    
    logger.info("测试完成")

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main()) 