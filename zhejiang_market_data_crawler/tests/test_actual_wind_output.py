#!/usr/bin/env python3
"""
测试风电实时总出力爬虫
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入爬虫
from crawlers.actual_wind_output_crawler import (
    ActualWindOutputCrawler, 
    crawl_actual_wind_output_for_date,
    run_historical_crawl,
    run_daily_crawl
)
from utils.logger import setup_logger
from utils.config import TARGET_TABLE

# 设置日志
logger = setup_logger('test_actual_wind_output')

async def test_single_date():
    """测试爬取单个日期的数据"""
    logger.info("测试爬取单个日期的数据")
    
    # 使用固定的日期：2025-06-02
    test_date = "2025-06-02"
    
    # 创建爬虫实例
    crawler = ActualWindOutputCrawler(target_table=TARGET_TABLE)
    
    # 获取数据
    df = crawler.fetch_data(test_date)
    
    if not df.empty:
        logger.info(f"成功爬取 {test_date} 的数据，共 {len(df)} 条记录")
        
        # 打印数据概览
        logger.info(f"数据概览:")
        logger.info(f"列名: {df.columns.tolist()}")
        logger.info(f"数据示例:\n{df.head(3)}")
        
        # 计算风电实时总出力字段的最小值和最大值
        if 'actual_wind_output' in df.columns:
            logger.info(f"actual_wind_output 范围: {df['actual_wind_output'].min()} 至 {df['actual_wind_output'].max()}")
        
        # 保存到数据库
        update_columns = [col for col in df.columns if col != 'date_time']
        success = crawler.save_to_db(df, update_columns=update_columns)
        
        if success:
            logger.info(f"成功将数据保存到表 {TARGET_TABLE}")
        else:
            logger.error("保存数据失败")
    else:
        logger.error(f"爬取 {test_date} 的数据失败")

async def test_date_range():
    """测试爬取日期范围的数据"""
    logger.info("测试爬取日期范围的数据")
    
    # 使用固定的日期范围：2025-06-02 到 2025-06-03
    start_date_str = "2025-06-02"
    end_date_str = "2025-06-03"
    
    # 创建爬虫实例
    crawler = ActualWindOutputCrawler(target_table=TARGET_TABLE)
    
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
        
        # 计算风电实时总出力字段的最小值和最大值
        if 'actual_wind_output' in df.columns:
            logger.info(f"actual_wind_output 范围: {df['actual_wind_output'].min()} 至 {df['actual_wind_output'].max()}")
        
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
            logger.info(f"成功将数据保存到表 {TARGET_TABLE}")
        else:
            logger.error("保存数据失败")
    else:
        logger.error(f"爬取 {start_date_str} 至 {end_date_str} 的数据失败")

async def main():
    """主函数"""
    logger.info("开始测试风电实时总出力爬虫")
    
    # 检查数据库配置
    from utils.config import DB_CONFIG
    
    # 打印数据库配置（隐藏密码）
    db_config_to_print = DB_CONFIG.copy()
    if 'password' in db_config_to_print:
        db_config_to_print['password'] = '***隐藏***'
    logger.info(f"数据库配置: {db_config_to_print}")
    logger.info(f"目标表名: {TARGET_TABLE}")
    
    # 检查数据库连接
    from utils.db_helper import get_db_connection
    from sqlalchemy import text
    
    try:
        engine, _ = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        
        # 检查表是否存在
        with engine.connect() as conn:
            result = conn.execute(text(f"SHOW TABLES LIKE '{TARGET_TABLE}'"))
            table_exists = result.fetchone() is not None
            
            if table_exists:
                logger.info(f"表 {TARGET_TABLE} 存在")
                
                # 检查表结构
                result = conn.execute(text(f"SHOW COLUMNS FROM {TARGET_TABLE}"))
                columns = [column[0] for column in result.fetchall()]
                logger.info(f"表 {TARGET_TABLE} 的列: {columns}")
                
                # 检查是否包含必要的列
                required_columns = ['date_time', 'actual_wind_output']
                if all(col in columns for col in required_columns):
                    logger.info("表结构正确，包含必要的列")
                else:
                    logger.error("表结构不正确，缺少必要的列")
                    return
            else:
                logger.error(f"表 {TARGET_TABLE} 不存在")
                return
        
        # 关闭连接
        engine.dispose()
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return
    
    # 运行单日数据测试
    await test_single_date()
    
    # 运行多天数据测试
    await test_date_range()
    
    logger.info("测试完成")

if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main()) 