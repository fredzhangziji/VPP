#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LEMMA: 电力市场分析智能体

基于Qwen-Agent框架开发的电力市场分析智能体，可以分析电价偏差事件，连接到远程Ollama服务和Qwen3:32b LLM模型。
具有以下核心工具：
1. 价格偏差分析工具
2. 竞价空间分析工具 
3. 电力生成偏差分析工具
4. 区域容量信息工具
"""

import json
import sys
import time
import logging
import re
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.agents import Assistant
from qwen_agent.llm.base import Message
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine

# 终端颜色设置
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    BLUE = '\033[34m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    CYAN = '\033[36m'

# 降低httpx库的日志级别
logging.getLogger("httpx").setLevel(logging.WARNING)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lemma_agent.log"),  # 文件日志
        logging.StreamHandler()  # 终端日志
    ]
)
logger = logging.getLogger("LEMMA")
# 只将ERROR及以上级别的日志输出到终端
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setLevel(logging.ERROR)

# 数据库配置 - 模块级别，便于所有工具共享
DB_CONFIG_VPP_SERVICE = {
    'host': '10.5.0.10',
    'user': 'root',
    'password': 'kunyu2023rds',
    'database': 'vpp_service',
    'port': 3306
}

@register_tool('get_price_deviation_report')
class PriceDeviationTool(BaseTool):
    """用于获取电价偏差报告的工具。"""
    
    description = "获取特定日期的电价偏差报告，包括预测价格和实际价格之间的偏差情况。"
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": False
    }]
    
    def parse_date(self, date_str: str) -> tuple:
        """
        解析日期字符串，返回用于查询的开始和结束时间
        
        Args:
            date_str: 日期字符串，可以是YYYY-MM-DD格式或描述性日期
            
        Returns:
            tuple: (start_time, end_time) 用于SQL查询的时间范围
        """
        
        # 处理"昨天"、"今天"等描述性日期
        if date_str == "昨天":
            target_date = datetime.now() - timedelta(days=1)
        elif date_str == "今天":
            target_date = datetime.now()
        elif date_str == "前天":
            target_date = datetime.now() - timedelta(days=2)
        # 处理"6月24日"格式
        elif re.match(r'(\d+)月(\d+)日', date_str):
            match = re.match(r'(\d+)月(\d+)日', date_str)
            month, day = int(match.group(1)), int(match.group(2))
            current_year = datetime.now().year
            target_date = datetime(current_year, month, day)
        # 处理YYYY-MM-DD格式
        else:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # 如果无法解析，默认使用昨天
                logger.warning(f"无法解析日期'{date_str}'，使用昨天作为默认值")
                target_date = datetime.now() - timedelta(days=1)
        
        # 生成该日的起止时间
        start_time = target_date.replace(hour=0, minute=0, second=0)
        end_time = target_date.replace(hour=23, minute=59, second=59)
        
        return start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def call(self, params: str, **kwargs) -> str:
        """执行电价偏差报告获取操作。""" 
        logger.debug(f"PriceDeviationTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 获取电价偏差报告...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date_str = parsed_params.get("date", "")
            region = parsed_params.get("region", "呼包东")  # 默认分析呼包东
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # 解析日期参数
        try:
            start_time, end_time = self.parse_date(date_str)
        except Exception as e:
            logger.error(f"日期解析错误: {e}")
            return json.dumps({"error": f"日期解析错误: {e}"}, ensure_ascii=False)
        
        # 确定价格字段
        if region == "呼包西":
            price_field = "west_price"
            price_region_filter = "呼包西"
        else:  # 默认为呼包东
            price_field = "east_price"
            price_region_filter = "呼包东"
            region = "呼包东"  # 确保region值一致
        
        # 使用SQLAlchemy创建数据库连接
        engine = None
        try:
            db_url = f"mysql+pymysql://{DB_CONFIG_VPP_SERVICE['user']}:{DB_CONFIG_VPP_SERVICE['password']}@{DB_CONFIG_VPP_SERVICE['host']}:{DB_CONFIG_VPP_SERVICE['port']}/{DB_CONFIG_VPP_SERVICE['database']}"
            engine = create_engine(db_url)
            
            # 查询1: 实际电价数据
            actual_price_query = f"""
            SELECT date_time, {price_field} as actual_price
            FROM neimeng_market_actual
            WHERE date_time BETWEEN %s AND %s
            ORDER BY date_time
            """
            
            df_actual = pd.read_sql_query(
                actual_price_query, 
                engine, 
                params=(start_time, end_time)
            )
            
            # 查询2: 预测电价数据 (SE模型) - 只获取每个时间点最新的预测
            forecast_price_query = """
            SELECT s.date_time, s.price as forecast_price
            FROM spotprice_forecast s
            INNER JOIN (
                SELECT date_time, price_region, MAX(fcst_time) as latest_fcst_time
                FROM spotprice_forecast
                WHERE date_time BETWEEN %s AND %s
                AND price_region = %s
                AND model = 'SE模型'
                GROUP BY date_time, price_region
            ) t ON s.date_time = t.date_time 
                  AND s.price_region = t.price_region 
                  AND s.fcst_time = t.latest_fcst_time
            WHERE s.model = 'SE模型'
            ORDER BY s.date_time
            """
            
            df_forecast = pd.read_sql_query(
                forecast_price_query, 
                engine, 
                params=(start_time, end_time, price_region_filter)
            )
            
            # 检查查询结果
            if df_actual.empty or df_forecast.empty:
                logger.warning(f"查询结果为空: actual={len(df_actual)}, forecast={len(df_forecast)}")
                return json.dumps({
                    "error": f"未找到{start_time}至{end_time}期间的{region}电价数据"
                }, ensure_ascii=False)
            
            # 合并数据集
            df_actual['date_time'] = pd.to_datetime(df_actual['date_time'])
            df_forecast['date_time'] = pd.to_datetime(df_forecast['date_time'])
            df_merged = pd.merge(df_actual, df_forecast, on='date_time', how='inner')
            
            if df_merged.empty:
                logger.warning("合并后的数据集为空，可能是实际价格和预测价格的时间戳不匹配")
                return json.dumps({
                    "error": f"实际价格和预测价格数据无法匹配，请检查数据"
                }, ensure_ascii=False)
            
            # 计算指标
            # 1. 整体偏差百分比
            overall_actual_avg = df_merged['actual_price'].mean()
            overall_forecast_avg = df_merged['forecast_price'].mean()
            deviation_percentage = round((overall_actual_avg - overall_forecast_avg) / overall_forecast_avg * 100, 2)
            
            # 计算每个时间点的价格差异
            df_merged['price_diff'] = df_merged['actual_price'] - df_merged['forecast_price']
            df_merged['price_diff_pct'] = (df_merged['price_diff'] / df_merged['forecast_price']) * 100
            
            # 计算绝对价差
            df_merged['abs_price_diff'] = abs(df_merged['price_diff'])
            df_merged['abs_price_diff_pct'] = abs(df_merged['price_diff_pct'])
            
            # 找出最大绝对价差和最小绝对价差
            max_abs_diff_row = df_merged.loc[df_merged['abs_price_diff'].idxmax()]
            min_abs_diff_row = df_merged.loc[df_merged['abs_price_diff'].idxmin()]
            
            # 提取最大价差信息
            max_abs_price_diff = round(max_abs_diff_row['price_diff'], 2)
            max_abs_diff_time = max_abs_diff_row['date_time'].strftime('%Y-%m-%d %H:%M:%S')
            max_abs_price_diff_pct = round(max_abs_diff_row['price_diff_pct'], 2)
            max_abs_direction = "高于" if max_abs_price_diff > 0 else "低于"
            
            # 提取最小价差信息
            min_abs_price_diff = round(min_abs_diff_row['price_diff'], 2)
            min_abs_diff_time = min_abs_diff_row['date_time'].strftime('%Y-%m-%d %H:%M:%S')
            min_abs_price_diff_pct = round(min_abs_diff_row['price_diff_pct'], 2)
            min_abs_direction = "高于" if min_abs_price_diff > 0 else "低于"
            
            # 2. 定义高峰和低谷时段
            df_merged['hour'] = df_merged['date_time'].dt.hour
            peak_hours = df_merged[(df_merged['hour'] >= 8) & (df_merged['hour'] < 20)]
            valley_hours = df_merged[(df_merged['hour'] < 8) | (df_merged['hour'] >= 20)]
            
            # 3. 计算高峰时段偏差
            if not peak_hours.empty:
                peak_actual_avg = peak_hours['actual_price'].mean()
                peak_forecast_avg = peak_hours['forecast_price'].mean()
                peak_hours_deviation = round((peak_actual_avg - peak_forecast_avg) / peak_forecast_avg * 100, 2)
            else:
                peak_hours_deviation = 0.0
            
            # 4. 计算低谷时段偏差
            if not valley_hours.empty:
                valley_actual_avg = valley_hours['actual_price'].mean()
                valley_forecast_avg = valley_hours['forecast_price'].mean()
                valley_hours_deviation = round((valley_actual_avg - valley_forecast_avg) / valley_forecast_avg * 100, 2)
            else:
                valley_hours_deviation = 0.0
            
            # 生成分析摘要
            deviation_direction = "高于" if deviation_percentage > 0 else "低于"
            deviation_magnitude = "较大" if abs(deviation_percentage) > 15 else "轻微"
            peak_valley_comparison = ""
            
            if abs(peak_hours_deviation) > abs(valley_hours_deviation):
                peak_valley_comparison = f"偏差在高峰期（{peak_hours_deviation}%）更为显著，而低谷期偏差较小（{valley_hours_deviation}%）。"
            elif abs(valley_hours_deviation) > abs(peak_hours_deviation):
                peak_valley_comparison = f"偏差在低谷期（{valley_hours_deviation}%）更为显著，而高峰期偏差较小（{peak_hours_deviation}%）。"
            
            max_min_info = f"最大价差发生在{max_abs_diff_time}，实际价格比预测{max_abs_direction}了{abs(max_abs_price_diff)}元/MWh（{abs(max_abs_price_diff_pct)}%）；最小价差发生在{min_abs_diff_time}，实际价格比预测{min_abs_direction}了{abs(min_abs_price_diff)}元/MWh（{abs(min_abs_price_diff_pct)}%）。"
            
            analysis_summary = f"该日期{region}电价出现{deviation_magnitude}偏差，平均实际价格比平均预测价格{deviation_direction}约{abs(deviation_percentage)}%。{peak_valley_comparison}{max_min_info}"
            
            # 计算市场价格（这里假设市场价格就是实际价格）
            market_price = round(overall_actual_avg, 2)
            
            # 构建报告
            report = {
                "date": date_str,
                "region": region,
                "market_price": market_price,
                "forecast_price": round(overall_forecast_avg, 2),
                "actual_price": round(overall_actual_avg, 2),
                "deviation_percentage": deviation_percentage,
                "peak_hours_deviation": peak_hours_deviation,
                "valley_hours_deviation": valley_hours_deviation,
                "max_price_diff": abs(max_abs_price_diff),
                "max_price_diff_pct": abs(max_abs_price_diff_pct),
                "max_diff_time": max_abs_diff_time,
                "max_diff_direction": max_abs_direction,
                "min_price_diff": abs(min_abs_price_diff),
                "min_price_diff_pct": abs(min_abs_price_diff_pct),
                "min_diff_time": min_abs_diff_time,
                "min_diff_direction": min_abs_direction,
                "analysis_summary": analysis_summary
            }
            
            return json.dumps(report, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"处理数据时出错: {e}", exc_info=True)
            return json.dumps({"error": f"处理数据时出错: {e}"}, ensure_ascii=False)

@register_tool('analyze_bidding_space_deviation')
class BiddingSpaceTool(BaseTool):
    """用于分析竞价空间偏差的工具。"""
    
    description = """分析特定日期和区域的电力市场竞价空间偏差。
    竞价空间是电力市场中的一个重要概念，电力市场的竞价空间（Bidding Space）可以被视为火电机组的“市场容量”或“博弈空间”。这个空间越大，火电的议价能力越强，电价越高；这个空间越小，火电的竞争越激烈，电价越低。因此，竞价空间对于现货电价的影响是显著的。
    内蒙省份的竞价空间的公式为：竞价空间 = 总负荷 - 总新能源发电量 - 非市场发电量 + 总东送电量（输往内蒙外部的电量）"""
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": True
    }, {
        "name": "time_period",
        "type": "string",
        "description": "分析的时段，例如'全天'、'高峰期'、'15:00-18:00'等",
        "required": False
    }]
    
    def parse_date(self, date_str: str) -> tuple:
        """
        解析日期字符串，返回用于查询的开始和结束时间
        
        Args:
            date_str: 日期字符串，可以是YYYY-MM-DD格式或描述性日期
            
        Returns:
            tuple: (start_time, end_time) 用于SQL查询的时间范围
        """
        from datetime import datetime, timedelta
        import re
        
        # 处理"昨天"、"今天"等描述性日期
        if date_str == "昨天":
            target_date = datetime.now() - timedelta(days=1)
        elif date_str == "今天":
            target_date = datetime.now()
        elif date_str == "前天":
            target_date = datetime.now() - timedelta(days=2)
        # 处理"6月24日"格式
        elif re.match(r'(\d+)月(\d+)日', date_str):
            match = re.match(r'(\d+)月(\d+)日', date_str)
            month, day = int(match.group(1)), int(match.group(2))
            current_year = datetime.now().year
            target_date = datetime(current_year, month, day)
        # 处理YYYY-MM-DD格式
        else:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # 如果无法解析，默认使用昨天
                logger.warning(f"无法解析日期'{date_str}'，使用昨天作为默认值")
                target_date = datetime.now() - timedelta(days=1)
        
        # 生成该日的起止时间
        start_time = target_date.replace(hour=0, minute=0, second=0)
        end_time = target_date.replace(hour=23, minute=59, second=59)
        
        return start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")

    def parse_time_period(self, time_period: str, start_time: str, end_time: str) -> tuple:
        """
        解析时间段参数，返回调整后的开始和结束时间
        
        Args:
            time_period: 时间段描述，如"全天"、"高峰期"或"08:00-16:00"
            start_time: 初始开始时间
            end_time: 初始结束时间
            
        Returns:
            tuple: 调整后的(start_time, end_time)
        """
        
        # 如果是"全天"或为空，直接返回原始时间范围
        if not time_period or time_period == "全天":
            return start_time, end_time
        
        # 解析日期部分
        base_date = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").date()
        
        # 处理"高峰期"(8:00-20:00)
        if time_period == "高峰期":
            new_start = datetime.combine(base_date, datetime.strptime("08:00", "%H:%M").time())
            new_end = datetime.combine(base_date, datetime.strptime("20:00", "%H:%M").time())
        # 处理"低谷期"(20:00-次日8:00)
        elif time_period == "低谷期":
            new_start = datetime.combine(base_date, datetime.strptime("20:00", "%H:%M").time())
            new_end = datetime.combine(base_date, datetime.strptime("08:00", "%H:%M").time())
        # 处理具体时间范围，如"08:00-16:00"
        elif re.match(r'(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})', time_period):
            match = re.match(r'(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})', time_period)
            start_h, start_m = int(match.group(1)), int(match.group(2))
            end_h, end_m = int(match.group(3)), int(match.group(4))
            
            new_start = datetime.combine(base_date, datetime.strptime(f"{start_h:02d}:{start_m:02d}", "%H:%M").time())
            new_end = datetime.combine(base_date, datetime.strptime(f"{end_h:02d}:{end_m:02d}", "%H:%M").time())
        else:
            # 无法解析时，返回原始时间范围
            logger.warning(f"无法解析时间段'{time_period}'，使用全天作为默认值")
            return start_time, end_time
        
        return new_start.strftime("%Y-%m-%d %H:%M:%S"), new_end.strftime("%Y-%m-%d %H:%M:%S")

    def call(self, params: str, **kwargs) -> str:
        """执行竞价空间偏差分析操作。"""
        logger.debug(f"BiddingSpaceTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 分析竞价空间偏差...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date_str = parsed_params.get("date", "")
            region = parsed_params.get("region", "内蒙全省")
            time_period = parsed_params.get("time_period", "全天")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # 解析日期参数
        try:
            start_time, end_time = self.parse_date(date_str)
            # 解析时间段
            start_time, end_time = self.parse_time_period(time_period, start_time, end_time)
        except Exception as e:
            logger.error(f"日期解析错误: {e}")
            return json.dumps({"error": f"日期解析错误: {e}"}, ensure_ascii=False)
        
        # 使用SQLAlchemy创建数据库连接
        engine = None
        try:
            db_url = f"mysql+pymysql://{DB_CONFIG_VPP_SERVICE['user']}:{DB_CONFIG_VPP_SERVICE['password']}@{DB_CONFIG_VPP_SERVICE['host']}:{DB_CONFIG_VPP_SERVICE['port']}/{DB_CONFIG_VPP_SERVICE['database']}"
            engine = create_engine(db_url)
            
            # 查询1: 预测的竞价空间数据
            forecast_query = """
            SELECT date_time, bidding_space AS bidding_space_forecast
            FROM neimeng_market_forecast
            WHERE date_time BETWEEN %s AND %s
            ORDER BY date_time
            """
            
            df_forecast = pd.read_sql_query(
                forecast_query, 
                engine, 
                params=(start_time, end_time)
            )
            
            # 查询2: 实际的竞价空间数据
            actual_query = """
            SELECT date_time, bidding_space AS bidding_space_actual
            FROM neimeng_market_actual
            WHERE date_time BETWEEN %s AND %s
            ORDER BY date_time
            """
            
            df_actual = pd.read_sql_query(
                actual_query, 
                engine, 
                params=(start_time, end_time)
            )
            
            # 检查查询结果
            if df_forecast.empty and df_actual.empty:
                logger.warning(f"查询结果为空: forecast={len(df_forecast)}, actual={len(df_actual)}")
                return json.dumps({
                    "error": f"未找到{start_time}至{end_time}期间的竞价空间数据"
                }, ensure_ascii=False)
            
            # 合并数据集 (外连接，保留所有时间点)
            df_forecast['date_time'] = pd.to_datetime(df_forecast['date_time'])
            df_actual['date_time'] = pd.to_datetime(df_actual['date_time'])
            df_merged = pd.merge(df_forecast, df_actual, on='date_time', how='outer')
            
            # 处理缺失值
            df_merged = df_merged.interpolate(method='linear')
            df_merged = df_merged.ffill().bfill()
            
            # 计算每个时间点的偏差
            df_merged['deviation'] = df_merged['bidding_space_actual'] - df_merged['bidding_space_forecast']
            df_merged['abs_deviation'] = abs(df_merged['deviation'])
            
            # 计算核心指标
            forecast_mean = round(df_merged['bidding_space_forecast'].mean(), 2)
            actual_mean = round(df_merged['bidding_space_actual'].mean(), 2)
            mean_deviation = round(abs(actual_mean - forecast_mean), 2)
            mean_deviation_percentage = round((mean_deviation / forecast_mean) * 100 if forecast_mean else 0, 2)
            
            # 找出最大和最小偏差点
            max_dev_idx = df_merged['abs_deviation'].idxmax()
            min_dev_idx = df_merged['abs_deviation'].idxmin()
            
            max_dev_time = df_merged.loc[max_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
            min_dev_time = df_merged.loc[min_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
            
            max_dev_value = round(df_merged.loc[max_dev_idx, 'abs_deviation'], 2)
            min_dev_value = round(df_merged.loc[min_dev_idx, 'abs_deviation'], 2)
            
            # 生成分析摘要
            deviation_direction = "高于" if actual_mean > forecast_mean else "低于"
            deviation_magnitude = "显著" if abs(mean_deviation_percentage) > 15 else "轻微"
            
            analysis_summary = (
                f"在{date_str}的{time_period}，{region}实际竞价空间均值为{actual_mean} MW，{deviation_magnitude}{deviation_direction}预测的{forecast_mean} MW，"
                f"平均偏差率达到{mean_deviation_percentage}%。偏差最大的时刻发生在{max_dev_time}，偏差值为{max_dev_value} MW；"
                f"偏差最小的时刻在{min_dev_time}，偏差值仅为{min_dev_value} MW。"
                f"根据公式'竞价空间 = 统调负荷 + 东送计划 - 新能源出力 - 非市场出力'，"
                f"本次偏差可能源于对统调负荷预测不足或新能源出力预测过高。"
            )
            
            # 构建报告
            report = {
                "date": date_str,
                "region": region,
                "time_period": time_period,
                "forecast_mean": forecast_mean,
                "actual_mean": actual_mean,
                "mean_deviation": mean_deviation,
                "mean_deviation_percentage": mean_deviation_percentage,
                "min_abs_deviation_point": {
                    "time": min_dev_time,
                    "deviation": min_dev_value
                },
                "max_abs_deviation_point": {
                    "time": max_dev_time,
                    "deviation": max_dev_value
                },
                "analysis_summary": analysis_summary
            }
            
            return json.dumps(report, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"处理数据时出错: {e}", exc_info=True)
            return json.dumps({"error": f"处理数据时出错: {e}"}, ensure_ascii=False)
        finally:
            if engine:
                engine.dispose()

@register_tool('analyze_power_generation_deviation')
class PowerGenerationTool(BaseTool):
    """用于分析电力生成偏差的工具。"""
    
    description = "分析特定日期和区域的电力生成偏差情况，尤其是新能源(风电、光伏)的实际出力与预测出力的差异。"
    
    parameters = [{
        "name": "date",
        "type": "string",
        "description": "分析日期，格式为YYYY-MM-DD或描述性日期如'昨天'、'6月24日'等",
        "required": True
    }, {
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'呼包东'、'呼包西'、'内蒙全省'等",
        "required": False
    }, {
        "name": "energy_type",
        "type": "string",
        "description": "能源类型，如'风电'、'光伏'、'水电'、'火电'或'全部'",
        "required": False
    }]
    
    def parse_date(self, date_str: str) -> tuple:
        """
        解析日期字符串，返回用于查询的开始和结束时间
        
        Args:
            date_str: 日期字符串，可以是YYYY-MM-DD格式或描述性日期
            
        Returns:
            tuple: (start_time, end_time) 用于SQL查询的时间范围
        """
        from datetime import datetime, timedelta
        import re
        
        # 处理"昨天"、"今天"等描述性日期
        if date_str == "昨天":
            target_date = datetime.now() - timedelta(days=1)
        elif date_str == "今天":
            target_date = datetime.now()
        elif date_str == "前天":
            target_date = datetime.now() - timedelta(days=2)
        # 处理"6月24日"格式
        elif re.match(r'(\d+)月(\d+)日', date_str):
            match = re.match(r'(\d+)月(\d+)日', date_str)
            month, day = int(match.group(1)), int(match.group(2))
            current_year = datetime.now().year
            target_date = datetime(current_year, month, day)
        # 处理YYYY-MM-DD格式
        else:
            try:
                target_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # 如果无法解析，默认使用昨天
                logger.warning(f"无法解析日期'{date_str}'，使用昨天作为默认值")
                target_date = datetime.now() - timedelta(days=1)
        
        # 生成该日的起止时间
        start_time = target_date.replace(hour=0, minute=0, second=0)
        end_time = target_date.replace(hour=23, minute=59, second=59)
        
        return start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def call(self, params: str, **kwargs) -> str:
        """执行电力生成偏差分析操作。"""
        logger.debug(f"PowerGenerationTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 分析电力生成偏差...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            date_str = parsed_params.get("date", "")
            region = parsed_params.get("region", "内蒙全省")  # 默认为内蒙全省
            energy_type = parsed_params.get("energy_type", "全部")  # 默认分析所有能源类型
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # 解析日期参数
        try:
            start_time, end_time = self.parse_date(date_str)
        except Exception as e:
            logger.error(f"日期解析错误: {e}")
            return json.dumps({"error": f"日期解析错误: {e}"}, ensure_ascii=False)
        
        # 使用SQLAlchemy创建数据库连接
        engine = None
        try:
            db_url = f"mysql+pymysql://{DB_CONFIG_VPP_SERVICE['user']}:{DB_CONFIG_VPP_SERVICE['password']}@{DB_CONFIG_VPP_SERVICE['host']}:{DB_CONFIG_VPP_SERVICE['port']}/{DB_CONFIG_VPP_SERVICE['database']}"
            engine = create_engine(db_url)
            
            # 查询1: 获取预测的风电和光伏数据
            forecast_query = """
            SELECT date_time, wind_power, solar_power
            FROM neimeng_market_forecast
            WHERE date_time BETWEEN %s AND %s
            ORDER BY date_time
            """
            
            df_forecast = pd.read_sql_query(
                forecast_query, 
                engine, 
                params=(start_time, end_time)
            )
            
            # 查询2: 获取实际的风电和光伏数据
            actual_query = """
            SELECT date_time, wind_power, solar_power
            FROM neimeng_market_actual
            WHERE date_time BETWEEN %s AND %s
            ORDER BY date_time
            """
            
            df_actual = pd.read_sql_query(
                actual_query, 
                engine, 
                params=(start_time, end_time)
            )
            
            # 检查查询结果
            if df_forecast.empty or df_actual.empty:
                logger.warning(f"查询结果为空: forecast={len(df_forecast)}, actual={len(df_actual)}")
                return json.dumps({
                    "error": f"未找到{start_time}至{end_time}期间的完整风电和光伏数据",
                    "forecast_empty": df_forecast.empty,
                    "actual_empty": df_actual.empty
                }, ensure_ascii=False)
            
            # 确保数据类型正确
            df_forecast['date_time'] = pd.to_datetime(df_forecast['date_time'])
            df_actual['date_time'] = pd.to_datetime(df_actual['date_time'])
            
            # 记录合并前数据状态
            logger.debug(f"合并前数据状态: forecast={df_forecast.shape}, actual={df_actual.shape}")
            logger.debug(f"预测数据列: {df_forecast.columns.tolist()}")
            logger.debug(f"实际数据列: {df_actual.columns.tolist()}")
            
            # 合并数据集 (内连接，只保留两边都有的时间点，避免过多NaN)
            df_merged = pd.merge(
                df_forecast, 
                df_actual, 
                on='date_time', 
                how='inner',  # 改为inner连接确保两边都有数据
                suffixes=('_forecast', '_actual')
            )
            
            # 检查合并后的数据
            if df_merged.empty:
                logger.warning("合并后数据为空，预测和实际数据时间点可能不匹配")
                return json.dumps({
                    "error": "预测和实际数据时间点不匹配，无法进行有效分析",
                    "forecast_times": df_forecast['date_time'].min().strftime('%Y-%m-%d %H:%M:%S') + " 至 " + df_forecast['date_time'].max().strftime('%Y-%m-%d %H:%M:%S') if not df_forecast.empty else "无数据",
                    "actual_times": df_actual['date_time'].min().strftime('%Y-%m-%d %H:%M:%S') + " 至 " + df_actual['date_time'].max().strftime('%Y-%m-%d %H:%M:%S') if not df_actual.empty else "无数据"
                }, ensure_ascii=False)
            
            # 检查合并后是否有NaN值
            nan_count = df_merged.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"合并后数据包含{nan_count}个NaN值，尝试插值处理")
                # 处理缺失值 - 逐列检查并处理
                for col in df_merged.columns:
                    if df_merged[col].isna().any():
                        # 跳过日期时间列
                        if col == 'date_time':
                            continue
                        # 对数值列进行插值
                        df_merged[col] = df_merged[col].interpolate(method='linear')
                
                # 检查处理后是否仍有NaN
                remaining_nan = df_merged.isna().sum().sum()
                if remaining_nan > 0:
                    logger.warning(f"插值后仍有{remaining_nan}个NaN值，使用向前和向后填充")
                    df_merged = df_merged.ffill().bfill()
                
                # 最后检查
                if df_merged.isna().sum().sum() > 0:
                    logger.warning("数据存在无法填充的NaN值，这可能影响分析结果")
            
            # 初始化分析结果字典
            analysis_result = {
                "date": date_str,
                "region": region,
                "energy_type": energy_type,
                "analysis": {}
            }
            
            try:
                # 处理风电数据
                wind_analysis = {}
                
                # 安全地计算风电平均值
                wind_forecast_avg = round(df_merged['wind_power_forecast'].mean(), 2)
                wind_actual_avg = round(df_merged['wind_power_actual'].mean(), 2)
                wind_deviation = round(wind_actual_avg - wind_forecast_avg, 2)
                # 安全处理除零情况
                if abs(wind_forecast_avg) < 0.01:
                    wind_deviation_pct = 0 if abs(wind_deviation) < 0.01 else 100.0
                    logger.warning("风电预测平均值接近零，无法计算准确的偏差百分比")
                else:
                    wind_deviation_pct = round((wind_deviation / wind_forecast_avg) * 100, 2)
                
                # 计算每个时间点的风电偏差并找出最大和最小偏差点
                df_merged['wind_deviation'] = df_merged['wind_power_actual'] - df_merged['wind_power_forecast']
                df_merged['wind_abs_deviation'] = abs(df_merged['wind_deviation'])
                
                # 安全获取最大和最小偏差点
                if not df_merged['wind_abs_deviation'].empty:
                    try:
                        wind_max_dev_idx = df_merged['wind_abs_deviation'].idxmax()
                        wind_min_dev_idx = df_merged['wind_abs_deviation'].idxmin()
                        
                        wind_max_dev_time = df_merged.loc[wind_max_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
                        wind_min_dev_time = df_merged.loc[wind_min_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
                        
                        wind_max_dev_value = round(df_merged.loc[wind_max_dev_idx, 'wind_deviation'], 2)
                        wind_min_dev_value = round(df_merged.loc[wind_min_dev_idx, 'wind_deviation'], 2)
                    except Exception as e:
                        logger.error(f"计算风电偏差点时出错: {e}")
                        wind_max_dev_time = "未知"
                        wind_min_dev_time = "未知"
                        wind_max_dev_value = 0
                        wind_min_dev_value = 0
                else:
                    wind_max_dev_time = "未知"
                    wind_min_dev_time = "未知"
                    wind_max_dev_value = 0
                    wind_min_dev_value = 0
                    
                # 保存风电分析结果
                wind_analysis = {
                    "forecast_avg_power": wind_forecast_avg,
                    "actual_avg_power": wind_actual_avg,
                    "deviation_absolute": wind_deviation,
                    "deviation_percentage": wind_deviation_pct,
                    "max_abs_deviation_point": {
                        "time": wind_max_dev_time,
                        "deviation": wind_max_dev_value
                    },
                    "min_abs_deviation_point": {
                        "time": wind_min_dev_time,
                        "deviation": wind_min_dev_value
                    }
                }
                
                # 处理光伏数据
                solar_analysis = {}
                
                # 安全地计算光伏平均值
                solar_forecast_avg = round(df_merged['solar_power_forecast'].mean(), 2)
                solar_actual_avg = round(df_merged['solar_power_actual'].mean(), 2)
                solar_deviation = round(solar_actual_avg - solar_forecast_avg, 2)
                # 安全处理除零情况
                if abs(solar_forecast_avg) < 0.01:
                    solar_deviation_pct = 0 if abs(solar_deviation) < 0.01 else 100.0
                    logger.warning("光伏预测平均值接近零，无法计算准确的偏差百分比")
                else:
                    solar_deviation_pct = round((solar_deviation / solar_forecast_avg) * 100, 2)
                
                # 计算每个时间点的光伏偏差并找出最大和最小偏差点
                df_merged['solar_deviation'] = df_merged['solar_power_actual'] - df_merged['solar_power_forecast']
                df_merged['solar_abs_deviation'] = abs(df_merged['solar_deviation'])
                
                # 安全获取最大和最小偏差点
                if not df_merged['solar_abs_deviation'].empty:
                    try:
                        solar_max_dev_idx = df_merged['solar_abs_deviation'].idxmax()
                        solar_min_dev_idx = df_merged['solar_abs_deviation'].idxmin()
                        
                        solar_max_dev_time = df_merged.loc[solar_max_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
                        solar_min_dev_time = df_merged.loc[solar_min_dev_idx, 'date_time'].strftime('%Y-%m-%d %H:%M:%S')
                        
                        solar_max_dev_value = round(df_merged.loc[solar_max_dev_idx, 'solar_deviation'], 2)
                        solar_min_dev_value = round(df_merged.loc[solar_min_dev_idx, 'solar_deviation'], 2)
                    except Exception as e:
                        logger.error(f"计算光伏偏差点时出错: {e}")
                        solar_max_dev_time = "未知"
                        solar_min_dev_time = "未知"
                        solar_max_dev_value = 0
                        solar_min_dev_value = 0
                else:
                    solar_max_dev_time = "未知"
                    solar_min_dev_time = "未知"
                    solar_max_dev_value = 0
                    solar_min_dev_value = 0
                
                # 保存光伏分析结果
                solar_analysis = {
                    "forecast_avg_power": solar_forecast_avg,
                    "actual_avg_power": solar_actual_avg,
                    "deviation_absolute": solar_deviation,
                    "deviation_percentage": solar_deviation_pct,
                    "max_abs_deviation_point": {
                        "time": solar_max_dev_time,
                        "deviation": solar_max_dev_value
                    },
                    "min_abs_deviation_point": {
                        "time": solar_min_dev_time,
                        "deviation": solar_min_dev_value
                    }
                }
            except KeyError as ke:
                logger.error(f"访问数据列时出错: {ke}")
                return json.dumps({
                    "error": f"数据处理过程中遇到列访问错误: {ke}",
                    "available_columns": df_merged.columns.tolist()
                }, ensure_ascii=False)
            except Exception as e:
                logger.error(f"数据分析过程中出错: {e}")
                return json.dumps({
                    "error": f"数据分析过程中出错: {e}"
                }, ensure_ascii=False)
            
            # 根据能源类型参数过滤分析结果
            if energy_type.lower() == "风电" or energy_type.lower() == "全部":
                analysis_result["analysis"]["wind_power"] = wind_analysis
            
            if energy_type.lower() == "光伏" or energy_type.lower() == "全部":
                analysis_result["analysis"]["solar_power"] = solar_analysis
            
            # 生成分析摘要
            wind_status = "高于" if wind_deviation_pct > 0 else "低于"
            solar_status = "高于" if solar_deviation_pct > 0 else "低于"
            
            wind_magnitude = "严重" if abs(wind_deviation_pct) > 20 else "轻微"
            solar_magnitude = "严重" if abs(solar_deviation_pct) > 20 else "轻微"
            
            # 安全计算风电和光伏偏差最大的时段统计
            try:
                wind_peak_hours = df_merged.loc[df_merged['wind_abs_deviation'] > df_merged['wind_abs_deviation'].quantile(0.75), 'date_time']
                wind_peak_period = "未能确定明显高偏差时段"
                
                if not wind_peak_hours.empty and len(wind_peak_hours) >= 3:
                    wind_peak_hours = [hour.hour for hour in wind_peak_hours]
                    if sum(1 for h in wind_peak_hours if 6 <= h <= 11) >= len(wind_peak_hours) * 0.5:
                        wind_peak_period = "上午时段"
                    elif sum(1 for h in wind_peak_hours if 12 <= h <= 17) >= len(wind_peak_hours) * 0.5:
                        wind_peak_period = "午后时段"
                    elif sum(1 for h in wind_peak_hours if 18 <= h <= 23) >= len(wind_peak_hours) * 0.5:
                        wind_peak_period = "晚间时段"
                    elif sum(1 for h in wind_peak_hours if 0 <= h <= 5) >= len(wind_peak_hours) * 0.5:
                        wind_peak_period = "深夜时段"
            except Exception as e:
                logger.error(f"计算风电峰值时段时出错: {e}")
                wind_peak_period = "未能确定明显高偏差时段"
            
            try:
                solar_peak_hours = df_merged.loc[df_merged['solar_abs_deviation'] > df_merged['solar_abs_deviation'].quantile(0.75), 'date_time']
                solar_peak_period = "未能确定明显高偏差时段"
                
                if not solar_peak_hours.empty and len(solar_peak_hours) >= 3:
                    solar_peak_hours = [hour.hour for hour in solar_peak_hours]
                    if sum(1 for h in solar_peak_hours if 8 <= h <= 11) >= len(solar_peak_hours) * 0.5:
                        solar_peak_period = "上午时段"
                    elif sum(1 for h in solar_peak_hours if 12 <= h <= 15) >= len(solar_peak_hours) * 0.5:
                        solar_peak_period = "正午时段"
                    elif sum(1 for h in solar_peak_hours if 16 <= h <= 19) >= len(solar_peak_hours) * 0.5:
                        solar_peak_period = "傍晚时段"
            except Exception as e:
                logger.error(f"计算光伏峰值时段时出错: {e}")
                solar_peak_period = "未能确定明显高偏差时段"
            
            # 构建摘要
            summary = ""
            if "wind_power" in analysis_result["analysis"] and "solar_power" in analysis_result["analysis"]:
                summary = f"该日{region}新能源出力{wind_magnitude if abs(wind_deviation_pct) > abs(solar_deviation_pct) else solar_magnitude}不足，" \
                         f"风电实际出力比预期{wind_status}约{abs(wind_deviation_pct)}%，" \
                         f"光伏出力{solar_status}约{abs(solar_deviation_pct)}%。" \
                         f"风电偏差最大时间点在{wind_max_dev_time}，光伏偏差最大时间点在{solar_max_dev_time}。" \
                         f"风电偏差主要集中在{wind_peak_period}，光伏偏差主要集中在{solar_peak_period}。"
            elif "wind_power" in analysis_result["analysis"]:
                summary = f"该日{region}风电出力{wind_magnitude}不足，" \
                         f"实际出力比预期{wind_status}约{abs(wind_deviation_pct)}%。" \
                         f"偏差最大的时间点在{wind_max_dev_time}，偏差主要集中在{wind_peak_period}。"
            elif "solar_power" in analysis_result["analysis"]:
                summary = f"该日{region}光伏出力{solar_magnitude}不足，" \
                         f"实际出力比预期{solar_status}约{abs(solar_deviation_pct)}%。" \
                         f"偏差最大的时间点在{solar_max_dev_time}，偏差主要集中在{solar_peak_period}。"
            
            analysis_result["analysis_summary"] = summary
            logger.debug(f"新能源出力分析结果: {summary}")
            
            # 最后进行JSON序列化，并处理可能的序列化错误
            try:
                result_json = json.dumps(analysis_result, ensure_ascii=False)
                logger.debug(f"成功生成分析结果JSON，长度: {len(result_json)}")
                return result_json
            except Exception as e:
                logger.error(f"JSON序列化错误: {e}")
                # 尝试一个简化的响应
                return json.dumps({
                    "error": "结果序列化失败，请检查数据格式",
                    "message": str(e)
                }, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"处理数据时出错: {e}", exc_info=True)
            return json.dumps({"error": f"处理数据时出错: {e}"}, ensure_ascii=False)
        finally:
            if engine:
                engine.dispose()

@register_tool('get_regional_capacity_info')
class RegionalCapacityTool(BaseTool):
    """用于获取区域发电容量信息的工具。"""
    
    description = "获取特定区域的发电容量结构信息，包括不同类型能源的装机容量、占比。"
    
    parameters = [{
        "name": "region",
        "type": "string",
        "description": "电力市场区域，例如'华东'、'华北'、'南方'等",
        "required": True
    }]
    
    # 硬编码的区域容量数据
    REGIONAL_CAPACITY_DATA = {
        '呼包东': {
            'capacity_by_type': {
                'wind': 21414.75,  # 单位: MW
                'solar': 4340.87,
                'thermal': 12319.00,
                'other': 70.36,
                'hydro': 1204.50
            },
            'total_capacity': 39349.43,
            'percentage_in_mengxi': {  # 占全蒙西地区的百分比
                'wind': 53.9,
                'solar': 18.1,
                'thermal': 28.0,
                'other': 15.3,
                'hydro': 57.8
            },
            'percentage_in_region': {  # 区内占比
                'wind': 54.4,
                'solar': 11.0,
                'thermal': 31.3,
                'other': 0.2,
                'hydro': 3.1
            }
        },
        '呼包西': {
            'capacity_by_type': {
                'wind': 18332.56,  # 单位: MW
                'solar': 19689.21,
                'thermal': 31648.00,
                'other': 390.70,
                'hydro': 880.91
            },
            'total_capacity': 70941.38,
            'percentage_in_mengxi': {  # 占全蒙西地区的百分比
                'wind': 46.1,
                'solar': 81.9,
                'thermal': 72.0,
                'other': 84.7,
                'hydro': 42.2
            },
            'percentage_in_region': {  # 区内占比
                'wind': 25.8,
                'solar': 27.8,
                'thermal': 44.6,
                'other': 0.6,
                'hydro': 1.2
            }
        }
    }
    
    def call(self, params: str, **kwargs) -> str:
        """执行区域容量信息获取操作。"""
        logger.debug(f"RegionalCapacityTool被调用，参数: {params}")
        print(f"{Colors.CYAN}[工具] 获取区域容量信息...{Colors.RESET}")
        
        # 解析参数
        try:
            parsed_params = json.loads(params)
            region = parsed_params.get("region", "")
        except Exception as e:
            logger.error(f"参数解析错误: {e}")
            return json.dumps({"error": f"参数解析错误: {e}"}, ensure_ascii=False)
        
        # 查找区域数据
        if region not in self.REGIONAL_CAPACITY_DATA:
            available_regions = ", ".join(self.REGIONAL_CAPACITY_DATA.keys())
            error_msg = f"未找到'{region}'地区的装机容量数据，可选地区为：{available_regions}"
            logger.warning(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        
        # 获取区域数据
        region_data = self.REGIONAL_CAPACITY_DATA[region]
        
        # 提取数据
        capacity_by_type = region_data['capacity_by_type']
        total_capacity = region_data['total_capacity']
        percentage_in_region = region_data['percentage_in_region']
        
        # 计算新能源渗透率（风电+光伏占比）
        new_energy_penetration = percentage_in_region['wind'] + percentage_in_region['solar']
        
        # 生成分析摘要
        primary_source = max(percentage_in_region.items(), key=lambda x: x[1])
        analysis_summary = f"{region}地区总装机容量为{total_capacity}MW，其中{primary_source[0]}为第一大电源，占比达{primary_source[1]}%。"
        
        if primary_source[0] == 'wind' or primary_source[0] == 'solar':
            analysis_summary += f"新能源（风电+光伏）装机占比达{new_energy_penetration}%，使得该区域电网对天气条件较为敏感。"
        else:
            analysis_summary += f"传统能源仍占主导，新能源（风电+光伏）渗透率为{new_energy_penetration}%。"
        
        # 构建响应
        response = {
            "region": region,
            "total_capacity": total_capacity,
            "capacity_by_type": {
                "wind": capacity_by_type['wind'],
                "solar": capacity_by_type['solar'],
                "hydro": capacity_by_type['hydro'],
                "thermal": capacity_by_type['thermal'],
                "other": capacity_by_type['other']
            },
            "percentage_by_type": {
                "wind": percentage_in_region['wind'],
                "solar": percentage_in_region['solar'],
                "hydro": percentage_in_region['hydro'],
                "thermal": percentage_in_region['thermal'],
                "other": percentage_in_region['other']
            },
            "new_energy_penetration": new_energy_penetration,
            "analysis_summary": analysis_summary
        }
        
        return json.dumps(response, ensure_ascii=False)

def find_final_assistant_message(response: Any) -> Optional[Message]:
    """
    从agent.run()返回的复杂response中找出代表最终助手回答的消息对象。
    
    Args:
        response: agent.run()返回的响应，可能是任何类型
        
    Returns:
        Message对象（如果找到），或者None（如果未找到）
    """
    # 处理列表类型响应（最常见的情况）
    if isinstance(response, list):
        # 从后向前遍历查找最终的助手回复
        for item in reversed(response):
            # 检查是否为Message对象
            if hasattr(item, 'role') and item.role == 'assistant':
                # 确保这不是工具调用消息
                if hasattr(item, 'content') and isinstance(item.content, str):
                    if '<tool_call>' not in item.content:
                        return item
            # 处理字典类型元素
            elif isinstance(item, dict) and item.get('role') == 'assistant':
                content = item.get('content', '')
                if isinstance(content, str) and '<tool_call>' not in content:
                    return item
    
    # 处理单个Message对象
    elif hasattr(response, 'role') and response.role == 'assistant':
        if hasattr(response, 'content') and '<tool_call>' not in response.content:
            return response
    
    # 处理字典类型响应
    elif isinstance(response, dict) and response.get('role') == 'assistant':
        content = response.get('content', '')
        if isinstance(content, str) and '<tool_call>' not in content:
            return response
    
    # 未找到合适的助手消息
    return None

def extract_content(content: str) -> str:
    """
    清洗单个消息内容的字符串，移除思考过程和工具调用标签。
    
    Args:
        content: 需要清洗的文本内容字符串
        
    Returns:
        清洗后的纯净字符串
    """
    # 确保输入是字符串
    if not isinstance(content, str):
        content = str(content)
    
    # 处理转义的换行符
    content = content.replace('\\n', '\n')
    
    # 1. 移除思考过程 - 使用更通用的模式
    # 移除<think>标签及其内容
    clean_text = re.sub(r'<think>[\s\S]*?</think>', '', content)
    
    # 处理其他常见思考格式（更简洁的模式）
    patterns = [
        r'\[thinking\][\s\S]*?\[/thinking\]',
        r'<思考>[\s\S]*?</思考>',
        r'思考[:：][\s\S]*?(?:思考结束|思考完毕)',
        r'<think>[\s\S]*', # 处理不闭合的think标签
        r'[\s\S]*</think>', # 处理不闭合的think标签结束
    ]
    
    for pattern in patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # 2. 移除工具调用和响应标签
    tool_patterns = [
        r'<tool_call>[\s\S]*?</tool_call>',
        r'<tool_response>[\s\S]*?</tool_response>'
    ]
    
    for pattern in tool_patterns:
        clean_text = re.sub(pattern, '', clean_text)
    
    # 3. 基础格式清理
    # 移除可能的前缀
    clean_text = re.sub(r'^(LEMMA|Assistant|AI): ', '', clean_text)
    
    # 清理多余换行和空白
    clean_text = re.sub(r'^\n+', '', clean_text)
    clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
    clean_text = clean_text.strip()
    
    # 如果提取结果为空，但看起来应该有内容(包含字母或数字字符)
    if not clean_text.strip() and re.search(r'[a-zA-Z0-9]', content):
        # 检查是否完整是工具调用
        if (('<tool_call>' in content and '</tool_call>' in content) or 
            ('<tool_response>' in content and '</tool_response>' in content)):
            # 此时确实是纯工具调用，不需要进行内容提取
            return ""
        
        # 试图提取任何可能的文本内容，跳过XML标签
        text_fragments = re.findall(r'>([^<>]+)<', content)
        if text_fragments:
            return ' '.join(text_fragments)
    
    return clean_text

def extract_tool_responses(messages: List[Message]) -> Dict[str, Any]:
    """
    一个更健壮的辅助函数，用于从消息历史中提取并分类所有工具的响应。

    Args:
        messages: 包含所有对话历史的消息列表。

    Returns:
        一个字典，键是工具名，值是该工具返回的已解析的JSON数据。
    """
    tool_outputs = {}
    # Qwen-Agent中，工具调用后，框架通常会添加一个 role='tool' 的消息
    # 我们遍历这个列表，寻找这样的消息
    for msg in messages:
        # Qwen-Agent 的标准做法是将工具调用的结果放在 role='tool' 的消息中
        if msg.role == 'tool':
            try:
                # msg.name 是工具名, msg.content 是返回的JSON字符串
                tool_name = msg.name
                tool_data = json.loads(msg.content)
                tool_outputs[tool_name] = tool_data
                logger.info(f"成功提取到工具 '{tool_name}' 的响应。")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"解析工具响应时出错: {e} - 消息内容: {getattr(msg, 'content', 'N/A')}")
    
    return tool_outputs

def generate_final_analysis(messages: List[Message]) -> str:
    """
    根据所有工具调用的结果，生成一个综合性的最终分析报告 (优化版)。
    
    Args:
        messages: 所有消息的历史记录。
    
    Returns:
        生成的分析报告字符串。
    """
    logger.info("开始生成最终分析报告...")
    start_time = time.time()
    
    # 第一步：调用辅助函数，结构化地提取所有工具的输出
    tool_results = extract_tool_responses(messages)
    
    if not tool_results:
        logger.warning("未能从历史记录中提取到任何有效的工具响应。")
        return "未能收集到足够的数据来生成分析报告。"

    # 第二步：基于提取好的数据，构建分析报告的各个部分
    analysis_parts = []
    
    # 价格偏差报告
    price_data = tool_results.get('get_price_deviation_report')
    if price_data:
        analysis_parts.append(
            f"根据价格偏差报告，在日期 {price_data.get('date', 'N/A')}，"
            f"{price_data.get('region', '该地区')}的实际电价与预测存在 {price_data.get('deviation_percentage', 0)}% 的显著偏差。"
            f"“{price_data.get('analysis_summary', '')}”"
        )
        
    # 竞价空间分析
    bidding_data = tool_results.get('analyze_bidding_space_deviation')
    if bidding_data:
        analysis_parts.append(
            f"竞价空间分析显示，偏差主要源于供应侧：{bidding_data.get('analysis_summary', '')}"
        )

    # 发电出力分析
    generation_data = tool_results.get('analyze_power_generation_deviation')
    if generation_data:
        wind_dev = generation_data.get('deviation_percentage', {}).get('wind', 0)
        solar_dev = generation_data.get('deviation_percentage', {}).get('solar', 0)
        analysis_parts.append(
            "发电出力分析找到了问题的核心：新能源出力严重不足。"
            f"风电实际出力比预期低了 {abs(wind_dev)}%，"
            f"光伏则低了 {abs(solar_dev)}%，这直接导致了供应紧张。"
        )

    # 区域容量结构分析
    capacity_data = tool_results.get('get_regional_capacity_info')
    if capacity_data:
        analysis_parts.append(
            "从区域能源结构来看，问题的深层原因在于："
            f"{capacity_data.get('analysis_summary', '未能获取到该地区的能源结构总结。')}"
        )

    # 第三步：组合所有分析部分，生成最终报告
    if not analysis_parts:
        final_report = "虽然调用了工具，但未能从返回数据中构建出有效的分析结论。"
        logger.warning(final_report)
    else:
        # 使用编号和换行符来组织报告，使其更清晰
        final_report = "综合分析报告如下：\n\n"
        for i, part in enumerate(analysis_parts, 1):
            final_report += f"{i}. {part}\n\n"
        final_report = final_report.strip()
    
    logger.info(f"生成分析报告完成，耗时: {time.time() - start_time:.2f}秒")
    return final_report


def typewriter_print(content, previous_content=""):
    """
    打印机风格的流式输出，不显示思考过程
    
    Args:
        content: 当前内容
        previous_content: 之前打印的内容
    
    Returns:
        返回当前打印的内容，用于下次比较
    """
    # 先提取内容 - 现在只需处理字符串
    if isinstance(content, str):
        clean_content = extract_content(content)
    else:
        # 如果是复杂对象，先尝试找出最终消息
        final_message = find_final_assistant_message(content)
        if final_message:
            if hasattr(final_message, 'content'):
                clean_content = extract_content(final_message.content)
            else:
                clean_content = extract_content(final_message.get('content', ''))
        else:
            # 如果找不到最终消息，退回到原来的处理方式
            clean_content = extract_content(str(content))
    
    # 如果提取的内容为空但原始内容不为空，可能是工具调用或未识别格式
    # 此时不输出任何内容，等待实际结果
    if not clean_content.strip() and content:
        return previous_content
    
    # 如果内容没有变化，直接返回
    if clean_content == previous_content:
        return previous_content
    
    # 计算上一次输出占用的行数
    if previous_content:
        # 安全起见，直接使用清屏方式重新打印
        # 这种方式不依赖于ANSI转义序列的行数计算，更可靠
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
    
    # 打印新内容
    prefix = f"{Colors.GREEN}LEMMA: {Colors.RESET}"
    print(f"{prefix}{clean_content}", flush=True)
    
    return clean_content


def print_welcome_banner():
    """打印欢迎横幅"""
    banner = f"""
{Colors.GREEN}╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║  {Colors.BOLD}LEMMA - 电力市场分析智能体{Colors.RESET}{Colors.GREEN}                                        ║
║                                                                    ║
║  基于Qwen-Agent框架和远程Ollama服务 (qwen3:32b)                    ║
║                                                                    ║
║  {Colors.YELLOW}核心功能:{Colors.GREEN}                                                         ║
║  • 价格偏差分析           • 竞价空间分析                           ║
║  • 电力生成偏差分析       • 区域容量信息                           ║
║                                                                    ║
║  {Colors.CYAN}输入'exit'或'quit'退出对话{Colors.GREEN}                                        ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def get_session_id():
    """生成唯一的会话ID"""
    return datetime.now().strftime("%Y%m%d%H%M%S")

if __name__ == "__main__":
    # 清屏和打印欢迎横幅
    os.system('cls' if os.name == 'nt' else 'clear')
    print_welcome_banner()

    session_id = get_session_id()
    logger.info(f"开始新会话，ID: {session_id}")

    # LLM配置
    llm_cfg = {
        'model': 'qwen3:32b',
        'model_server': 'http://10.5.0.100:11434/v1',
        'api_key': 'EMPTY',
        'generate_cfg': {
            'temperature': 0.7,
            'top_p': 0.8
        },
        'request_timeout': 120 
    }

    # 工具列表和系统提示词
    tools = [
        'get_price_deviation_report', 'analyze_bidding_space_deviation',
        'analyze_power_generation_deviation', 'get_regional_capacity_info'
    ]
    system_message = """# 角色
你是一个名为LEMMA的、顶级的电力市场AI Co-pilot。你的用户是售电公司电力交易员，他们负责帮助客户进行电力交易以套利。你的核心目标是通过快速、精准、数据驱动的分析，帮助交易员洞察市场动态、进行精准的电力市场复盘，最终实现利润最大化。
# 核心能力
- 市场复盘分析：深度复盘历史市场事件（如价格异常、新能源出力偏差），穿透表象，精准定位根本原因，形成从售电公司角度出发的、可供决策参考的、逻辑严谨的复盘报告。

# 工作原则
- 数据驱动：你的所有结论都必须基于通过工具获取的量化数据。避免主观臆断和模糊不清的表述。你的语言就是数据。
- 效率至上：交易员的时间极其宝贵。你的回答必须直击要点、简洁、结构化。使用项目符号或编号列表来呈现关键发现，使信息一目了然。
- 量化影响：在分析原因时，不仅要说明"是什么"，更要量化"影响有多大"。
- - 错误示范： "因为光伏出力不足导致价格上涨。"
- - 正确示范： "价格上涨的核心驱动因素是光伏出力严重低于预期（偏差-33.7%），这导致约XX兆瓦的电力缺口需要由成本更高的火电机组（成本约XXX元/兆瓦时）来填补。"
- 主动洞察：在完成分析报告后，在有实际数据支持的情况下，请主动提出1-2个关键的、面向未来的洞察或风险提示。例如："结论：本次事件暴露了该地区在午间高峰期对光伏的过度依赖。建议关注未来几日的天气预报，如果再次出现多云天气，类似的日前高价风险可能重现。"
- 回答语言：必须用简体中文进行思考以及回答。

# 可用工具
你有以下工具可用，每轮问答只能调用一个工具，并在对话末尾询问用户是否要进行下一个工具的调用（但是不能暴露工具的名字），如果用户同意，则继续调用下一个工具，直到用户不同意为止：
1. get_price_deviation_report: 用于获取价格偏差的基本情况和量化数据。
2. analyze_bidding_space_deviation: 用于分析市场供需情况。
3. analyze_power_generation_deviation: 用于分析各类电源（尤其是新能源）的实际出力与预测的偏差。
4. get_regional_capacity_info: 用于了解地区的能源结构和装机容量。

# 核心指令：
面对用户的复杂分析请求（例如"复盘一下昨天XX地区的价格异常"），你必须遵循以下高效的工作流程：
- 不能暴露具体的工具名称。
- 逐一问答、逐步执行的模式：首先调用get_price_deviation_report分析价格偏差，结束之后询问用户是否需要分析市场供需情况（竞价空间）。
- 如果用户同意，调用analyze_bidding_space_deviation分析市场供需情况，结束之后询问用户是否需要分析各类能源的实际出力与预测的偏差。
- 如果用户同意，调用analyze_power_generation_deviation分析各类电源（尤其是新能源）的实际出力与预测的偏差，结束之后询问用户是否需要分析地区的能源结构和装机容量。
- 如果用户同意，调用get_regional_capacity_info，根据地区的能源结构和装机容量进行简单分析，结束之后询问用户是否需要总结一份综合性的分析报告。
- 如果用户同意，根据上述四个工具的结果分析结果，生成一份从售电公司角度出发的综合性的最终分析报告。
    """

    # Agent初始化，使用 Assistant 类
    try:
        print(f"{Colors.BLUE}初始化LEMMA Agent...{Colors.RESET}")
        agent = Assistant(llm=llm_cfg,
                          system_message=system_message,
                          function_list=tools)
        print(f"{Colors.GREEN}初始化成功！{Colors.RESET}")
    except Exception as e:
        logger.error(f"Agent初始化失败: {e}", exc_info=True)
        sys.exit(1)

    # 初始化消息历史
    messages = []

    # 交互循环
    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}{Colors.YELLOW}用户: {Colors.RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break

        if user_input.lower() in ['exit', 'quit']:
            print(f"\n{Colors.GREEN}再见！感谢使用LEMMA电力市场分析智能体！{Colors.RESET}")
            break

        messages.append(Message(role="user", content=user_input))

        print(f"\n{Colors.BLUE}LEMMA思考中...{Colors.RESET}")

        response_generator = agent.run(messages=messages, stream=True)

        full_response_content = ""
        last_chunk_printed = ""
        final_message_object = None

        try:
            for response_chunk in response_generator:
                # Qwen-Agent的流式输出通常是一个消息列表
                if isinstance(response_chunk, list) and response_chunk:
                    # 我们只关心最新的消息，通常是列表的最后一个
                    latest_message = response_chunk[-1]
                    
                    # 检查是否是助手正在生成内容
                    if latest_message.role == 'assistant':
                        new_content = latest_message.content
                        # 计算增量部分并以流式打印
                        if new_content.startswith(last_chunk_printed):
                            delta = new_content[len(last_chunk_printed):]
                            print(f"{Colors.GREEN}{delta}{Colors.RESET}", end='', flush=True)
                            last_chunk_printed = new_content
                        else: # 如果内容不是增量的，直接覆盖打印
                            # 使用回车符\r和清行符\x1b[K来覆盖当前行
                            sys.stdout.write('\r\x1b[K')
                            print(f"{Colors.GREEN}LEMMA: {new_content}{Colors.RESET}", end='', flush=True)
                            last_chunk_printed = new_content

                        full_response_content = new_content

                    # 检查是否有工具调用，并在后台打印出来用于调试
                    elif latest_message.role == 'tool' or (isinstance(latest_message.content, str) and '<tool_call>' in latest_message.content):
                         # 工具调用时，在思考提示后换行，让界面更整洁
                        if not last_chunk_printed.endswith('\n'):
                            print()
                        print(f"{Colors.CYAN}[工具] Agent正在调用工具... ({latest_message.tool_name if hasattr(latest_message, 'tool_name') else ''}){Colors.RESET}")
                        last_chunk_printed = "" # 重置已打印内容，准备接收新一轮的助手回复
            
            # 循环结束后，打印一个换行符，让格式更美观
            print()

            # 使用新的find_final_assistant_message函数查找最终回答
            final_message = find_final_assistant_message(response_chunk)
            
            if final_message:
                # 如果找到了最终回答消息，提取并清洗它的内容
                if hasattr(final_message, 'content'):
                    clean_final_answer = extract_content(final_message.content)
                else:
                    clean_final_answer = extract_content(final_message.get('content', ''))
                
                if clean_final_answer:
                    # 如果有干净的回答，将它作为最终消息对象
                    final_message_object = Message(role='assistant', content=clean_final_answer)
                    logger.info("成功找到并提取了最终回答。")
                else:
                    logger.warning("找到了最终消息，但内容提取为空。")
            else:
                # 如果没有找到最终消息，尝试直接从full_response_content提取
                clean_final_answer = extract_content(full_response_content)
                
                if clean_final_answer:
                    # 如果提取成功，创建一个新的消息对象
                    final_message_object = Message(role='assistant', content=clean_final_answer)
                    logger.info("从完整响应中提取了最终回答。")
                else:
                    # 如果都失败了，尝试手动生成分析报告
                    logger.warning("未能提取到最终回答，尝试手动生成分析报告。")
                    manual_summary = generate_final_analysis(messages + [Message(role='assistant', content=full_response_content)])
                    if manual_summary:
                        print(f"{Colors.BOLD}{Colors.GREEN}LEMMA (分析总结):{Colors.RESET}\n{manual_summary}")
                        final_message_object = Message(role='assistant', content=manual_summary)
                    else:
                        logger.error("手动生成分析报告也失败了。")
                        print(f"{Colors.RED}无法生成最终分析报告。{Colors.RESET}")

            # 将最终有效的回复加入历史记录
            if final_message_object:
                messages.append(final_message_object)

        except Exception as e:
            logger.error(f"处理流式响应时出错: {e}", exc_info=True)
            print(f"{Colors.RED}\n处理过程中发生错误，请查看日志 lemma_agent.log。{Colors.RESET}")