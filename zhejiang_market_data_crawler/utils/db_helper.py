"""
数据库辅助函数，封装数据库操作
"""

import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker
import logging
from utils.logger import setup_logger
from utils.config import DB_CONFIG, TARGET_TABLE
from pub_tools.db_tools import get_db_connection, release_db_connection, upsert_multiple_columns_to_db, read_from_db, write_to_db
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text
from utils.config import DB_CONFIG, TARGET_TABLE
from utils.logger import setup_logger
from datetime import datetime, timedelta
import random
import sys
from sqlalchemy.exc import SQLAlchemyError

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
    
    def upsert_data(self, df, table_name=None, update_columns=None, batch_size=50):
        """
        更新或插入数据
        
        Args:
            df: 包含数据的DataFrame
            table_name: 目标表名，如果为None则使用初始化时的表名
            update_columns: 当记录已存在时要更新的列，默认为None（更新所有列）
            batch_size: 每批处理的记录数，默认为50
        """
        if df.empty:
            logger.warning("没有数据需要插入")
            return 0
        
        # 如果table_name为None，则使用初始化时的表名
        table_name = table_name or self.target_table
        
        try:
            self.connect()
            
            # 确保date_time列是datetime类型
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
                if hasattr(df['date_time'].dt, 'tz_localize'):
                    df['date_time'] = df['date_time'].dt.tz_localize(None)
            
            # 添加create_time列，但不添加update_time列
            df['create_time'] = datetime.now()
            
            # 使用pub_tools.db_tools中的upsert_multiple_columns_to_db函数
            try:
                # 如果未指定update_columns，则更新除date_time外的所有列
                if update_columns is None:
                    update_columns = [col for col in df.columns if col != 'date_time']
                
                # 执行upsert操作
                upsert_multiple_columns_to_db(self.engine, df, table_name, update_columns)
                
                success_count = len(df)
                logger.info(f"成功插入或更新 {success_count} 条记录到表 {table_name}")
                return success_count
            except Exception as e:
                logger.error(f"使用upsert_multiple_columns_to_db失败: {e}")
                
                # 回退到原有pandas实现
                logger.info("尝试使用pandas to_sql进行插入")
                try:
                    # 使用pandas的to_sql方法进行插入
                    df.to_sql(
                        name=table_name,
                        con=self.engine,
                        if_exists='append',  # 如果表存在则追加数据
                        index=False,
                        chunksize=batch_size
                    )
                    
                    success_count = len(df)
                    logger.info(f"成功插入 {success_count} 条记录到表 {table_name}")
                    return success_count
                except Exception as pandas_err:
                    logger.error(f"使用pandas to_sql也失败了: {pandas_err}")
                    
                    # 尝试使用SQLAlchemy会话进行更新操作
                    try:
                        from sqlalchemy.orm import Session
                        
                        logger.info("尝试使用会话方式进行更新操作")
                        success_count = 0
                        
                        with Session(self.engine) as session:
                            for _, row in df.iterrows():
                                try:
                                    # 检查记录是否存在
                                    query = f"SELECT 1 FROM {table_name} WHERE date_time = :date_time"
                                    result = session.execute(text(query), {"date_time": row['date_time']})
                                    exists = result.scalar() is not None
                                    
                                    if exists:
                                        # 构建更新语句
                                        update_cols = [col for col in df.columns if col != 'date_time' and col != 'create_time']
                                        if not update_cols:
                                            continue
                                        
                                        set_clauses = []
                                        params = {"date_time": row['date_time']}
                                        
                                        for col in update_cols:
                                            if pd.notna(row[col]):
                                                set_clauses.append(f"{col} = :{col}")
                                                params[col] = row[col]
                                        
                                        if set_clauses:
                                            update_stmt = text(f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE date_time = :date_time")
                                            session.execute(update_stmt, params)
                                            success_count += 1
                                    else:
                                        # 构建插入语句（不包括自动生成的字段）
                                        cols = [col for col in df.columns if col != 'update_time']
                                        values = ", ".join([f":{col}" for col in cols])
                                        cols_str = ", ".join(cols)
                                        
                                        params = {col: row[col] for col in cols if pd.notna(row[col])}
                                        
                                        insert_stmt = text(f"INSERT INTO {table_name} ({cols_str}) VALUES ({values})")
                                        session.execute(insert_stmt, params)
                                        success_count += 1
                                except Exception as ex:
                                    logger.error(f"记录处理失败: {ex}")
                                    logger.debug(f"失败记录: {row.to_dict()}")
                            
                            session.commit()
                        
                        logger.info(f"成功插入或更新 {success_count} 条记录到表 {table_name}")
                        return success_count
                    except Exception as ex:
                        logger.error(f"尝试使用会话方式进行更新也失败了: {ex}")
                        return 0
        finally:
            self.disconnect()
    
    def __enter__(self):
        """支持with语句"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持with语句"""
        self.disconnect()


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
        engine, metadata = get_db_connection(DB_CONFIG)
        logger.info("数据库连接成功")
        
        # 如果未指定update_columns，则更新除date_time外的所有列
        if update_columns is None:
            update_columns = [col for col in df.columns if col != 'date_time']
        
        # 使用pub_tools.db_tools中的upsert_multiple_columns_to_db函数
        try:
            # 确保date_time列是datetime类型
            if 'date_time' in df.columns:
                df['date_time'] = pd.to_datetime(df['date_time'])
                if hasattr(df['date_time'].dt, 'tz_localize'):
                    df['date_time'] = df['date_time'].dt.tz_localize(None)
            
            # 执行upsert操作
            upsert_multiple_columns_to_db(engine, df, table_name, update_columns)
            
            logger.info(f"成功插入或更新 {len(df)} 条记录到表 {table_name}")
            return True
        except Exception as e:
            logger.error(f"使用upsert_multiple_columns_to_db失败: {e}")
            
            # 回退到自定义实现
            logger.info("尝试使用自定义实现进行upsert操作")
            try:
                # 构建唯一键
                unique_key = 'date_time'
                
                # 使用sqlalchemy连接执行批量插入
                conn = engine.connect()
                try:
                    # 生成插入语句
                    insert_stmt = f"INSERT INTO {table_name} ("
                    insert_stmt += ", ".join(df.columns)
                    insert_stmt += ") VALUES ("
                    insert_stmt += ", ".join([":"+col for col in df.columns])
                    insert_stmt += ")"
                    
                    # 生成ON DUPLICATE KEY UPDATE语句
                    update_stmt = " ON DUPLICATE KEY UPDATE "
                    update_parts = []
                    for col in update_columns:
                        update_parts.append(f"{col} = VALUES({col})")
                    
                    # 始终更新create_time和update_time
                    if 'update_time' in df.columns and 'update_time' not in update_columns:
                        update_parts.append("update_time = VALUES(update_time)")
                    
                    update_stmt += ", ".join(update_parts)
                    
                    # 完整SQL语句
                    sql = insert_stmt + update_stmt
                    
                    # 将DataFrame转换为字典列表
                    records = df.to_dict('records')
                    
                    # 执行批量插入
                    result = conn.execute(text(sql), records)
                    conn.commit()
                    
                    logger.info(f"成功插入或更新 {len(records)} 条记录到表 {table_name}")
                    return True
                finally:
                    conn.close()
            except Exception as backup_err:
                logger.error(f"自定义实现也失败了: {backup_err}")
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
        engine, metadata = get_db_connection(DB_CONFIG)
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