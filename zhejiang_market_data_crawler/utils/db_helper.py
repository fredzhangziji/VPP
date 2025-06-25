"""
数据库辅助函数，封装数据库操作
"""

import pandas as pd
from datetime import datetime
from sqlalchemy import text
from utils.logger import setup_logger
from utils.config import DB_CONFIG, TARGET_TABLE
from pub_tools.db_tools import get_db_connection, release_db_connection, upsert_multiple_columns_to_db

# 设置日志记录器
logger = setup_logger('db_helper')

class DBHelper:
    """数据库辅助类，提供数据库操作方法"""
    
    def __init__(self, db_config=None, target_table=None):
        """
        初始化数据库辅助类
        
        Args:
            db_config: 数据库配置，默认使用config.py中的配置
            target_table: 目标表名，默认使用config.py中的配置
        """
        self.db_config = db_config or DB_CONFIG
        self.target_table = target_table or TARGET_TABLE
        self.engine = None
        self.metadata = None
    
    def connect(self):
        """连接数据库"""
        if self.engine is None:
            try:
                # 直接使用pub_tools.db_tools中的函数
                self.engine, self.metadata = get_db_connection(self.db_config)
                logger.info("数据库连接成功")
            except Exception as e:
                logger.error(f"数据库连接失败: {e}")
                raise
    
    def disconnect(self):
        """断开数据库连接"""
        if self.engine is not None:
            # 直接使用pub_tools.db_tools中的函数
            release_db_connection(self.engine)
            self.engine = None
            self.metadata = None
            logger.info("数据库连接已断开")

# 单例模式，提供全局访问点
db_helper = DBHelper()

def save_to_db(df, table_name=None, update_columns=None, timezone='Asia/Shanghai'):
    """
    将数据保存到数据库
    
    Args:
        df: 要保存的数据（DataFrame）
        table_name: 表名
        update_columns: 要更新的列
        timezone: 时区
        
    Returns:
        success: 是否成功
    """
    if df is None or df.empty:
        logger.warning("没有数据可保存")
        return False
    
    # 使用指定表名或默认表名
    table_name = table_name or TARGET_TABLE
    
    # 给定时间戳标记
    now = datetime.now()
    df['create_time'] = now
    
    # 添加update_time字段
    if 'update_time' in df.columns:
        df['update_time'] = now
    
    # 获取数据库连接
    try:
        # 使用pub_tools.db_tools中的函数获取数据库连接
        engine, _ = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        
        # 如果未指定update_columns，则更新除date_time外的所有列
        if update_columns is None:
            update_columns = [col for col in df.columns if col != 'date_time']
        
        # 使用pub_tools.db_tools中的upsert_multiple_columns_to_db函数，并添加重试机制
        # 确保date_time列是datetime类型
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
            if hasattr(df['date_time'].dt, 'tz_localize'):
                df['date_time'] = df['date_time'].dt.tz_localize(None)
        
        # 重试机制
        max_retries = 3
        try:
            for attempt in range(max_retries):
                try:
                    # 执行upsert操作
                    upsert_multiple_columns_to_db(engine, df, table_name, update_columns)
                    
                    logger.info(f"成功插入或更新 {len(df)} 条记录到表 {table_name}")
                    return True
                except Exception as e:
                    wait_time = 2 ** attempt  # 指数退避策略：1秒，2秒，4秒
                    logger.warning(f"尝试 {attempt+1}/{max_retries} 失败: {e}. 等待 {wait_time} 秒后重试...")
                    
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(wait_time)
                    else:
                        # 所有重试都失败
                        logger.error(f"插入数据失败，已重试 {max_retries} 次: {e}")
                        return False
        finally:
            # 使用pub_tools.db_tools释放数据库连接
            release_db_connection(engine)
            logger.info("数据库连接已断开")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return False

def get_db_engine():
    """获取数据库引擎"""
    # 使用pub_tools.db_tools中的get_db_connection函数获取数据库连接
    engine, _ = get_db_connection(DB_CONFIG)
    return engine

def check_table_structure(table_name, required_columns):
    """
    检查表结构是否包含所需的列
    
    Args:
        table_name: 表名
        required_columns: 所需的列列表
        
    Returns:
        bool: 是否包含所有所需的列
    """
    try:
        engine = get_db_engine()
        
        # 获取表的列
        with engine.connect() as conn:
            query = f"SHOW COLUMNS FROM {table_name}"
            result = conn.execute(text(query))
            columns = [row[0] for row in result]
        
        # 检查所需的列是否都存在
        for col in required_columns:
            if col not in columns:
                logger.warning(f"表 {table_name} 缺少列: {col}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"检查表结构失败: {e}")
        return False
    finally:
        if engine:
            engine.dispose()

def get_test_db_connection():
    """
    获取测试数据库连接
    
    Returns:
        connection: 数据库连接
    """
    try:
        # 使用pub_tools.db_tools中的函数获取数据库连接
        engine, _ = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        return engine
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return None

def close_db_connection(engine):
    """
    关闭数据库连接
    
    Args:
        engine: 数据库连接
    """
    if engine:
        try:
            release_db_connection(engine)
            logger.info("数据库连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}") 