"""
主程序入口，用于调用各个爬虫
"""

import os
import sys
import time
import argparse
import concurrent.futures
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from utils.config import CONCURRENCY, REQUEST_INTERVAL, get_api_cookie
from crawlers.week_ahead_load_crawler import WeekAheadLoadCrawler
from crawlers.day_ahead_load_crawler import DayAheadLoadCrawler
from crawlers.actual_load_crawler import ActualLoadCrawler
from crawlers.system_backup_crawler import SystemBackupCrawler
from crawlers.total_generation_forecast_crawler import TotalGenerationForecastCrawler
from crawlers.external_power_plan_crawler import ExternalPowerPlanCrawler
from crawlers.non_market_solar_forecast_crawler import NonMarketSolarForecastCrawler
from crawlers.non_market_wind_forecast_crawler import NonMarketWindForecastCrawler
from crawlers.non_market_nuclear_forecast_crawler import NonMarketNuclearForecastCrawler
from crawlers.non_market_hydro_forecast_crawler import NonMarketHydroForecastCrawler
from crawlers.day_ahead_solar_total_forecast_crawler import DayAheadSolarTotalForecastCrawler
from crawlers.day_ahead_wind_total_forecast_crawler import DayAheadWindTotalForecastCrawler
from crawlers.week_ahead_pumped_storage_forecast_crawler import WeekAheadPumpedStorageForecastCrawler
from crawlers.day_ahead_hydro_total_forecast_crawler import DayAheadHydroTotalForecastCrawler
from crawlers.day_ahead_pumped_storage_forecast_crawler import DayAheadPumpedStorageForecastCrawler
from crawlers.actual_total_generation_crawler import ActualTotalGenerationCrawler
from crawlers.actual_solar_output_crawler import ActualSolarOutputCrawler
from crawlers.actual_wind_output_crawler import ActualWindOutputCrawler
from crawlers.actual_hydro_output_crawler import ActualHydroOutputCrawler
from crawlers.actual_pumped_storage_output_crawler import ActualPumpedStorageOutputCrawler
from crawlers.non_market_total_output_crawler import NonMarketTotalOutputCrawler

# 设置日志记录器
logger = setup_logger('main')

# 爬虫列表
# 在这里添加所有需要运行的爬虫
CRAWLERS = [
    {
        'name': '周前负荷预测爬虫',
        'crawler': WeekAheadLoadCrawler,
        'params': {}
    },
    {
        'name': '日前负荷预测爬虫',
        'crawler': DayAheadLoadCrawler,
        'params': {}
    },
    {
        'name': '实际负荷爬虫',
        'crawler': ActualLoadCrawler,
        'params': {}
    },
    {
        'name': '系统备用爬虫',
        'crawler': SystemBackupCrawler,
        'params': {}
    },
    {
        'name': '发电总出力预测爬虫',
        'crawler': TotalGenerationForecastCrawler,
        'params': {}
    },
    {
        'name': '外来电受电计划爬虫',
        'crawler': ExternalPowerPlanCrawler,
        'params': {}
    },
    {
        'name': '非市场光伏出力预测爬虫',
        'crawler': NonMarketSolarForecastCrawler,
        'params': {}
    },
    {
        'name': '非市场风电出力预测爬虫',
        'crawler': NonMarketWindForecastCrawler,
        'params': {}
    },
    {
        'name': '非市场核电出力预测爬虫',
        'crawler': NonMarketNuclearForecastCrawler,
        'params': {}
    },
    {
        'name': '非市场水电出力预测爬虫',
        'crawler': NonMarketHydroForecastCrawler,
        'params': {}
    },
    {
        'name': '光伏总出力预测爬虫',
        'crawler': DayAheadSolarTotalForecastCrawler,
        'params': {}
    },
    {
        'name': '风电总出力预测爬虫',
        'crawler': DayAheadWindTotalForecastCrawler,
        'params': {}
    },
    {
        'name': '抽蓄总出力预测爬虫',
        'crawler': WeekAheadPumpedStorageForecastCrawler,
        'params': {}
    },
    {
        'name': '水电总出力预测爬虫',
        'crawler': DayAheadHydroTotalForecastCrawler,
        'params': {}
    },
    {
        'name': '日抽蓄总出力预测爬虫',
        'crawler': DayAheadPumpedStorageForecastCrawler,
        'params': {}
    },
    {
        'name': '发电实时总出力爬虫',
        'crawler': ActualTotalGenerationCrawler,
        'params': {}
    },
    {
        'name': '光伏实时总出力爬虫',
        'crawler': ActualSolarOutputCrawler,
        'params': {}
    },
    {
        'name': '风电实时总出力爬虫',
        'crawler': ActualWindOutputCrawler,
        'params': {}
    },
    {
        'name': '水电实时总出力爬虫',
        'crawler': ActualHydroOutputCrawler,
        'params': {}
    },
    {
        'name': '抽蓄实时总出力爬虫',
        'crawler': ActualPumpedStorageOutputCrawler,
        'params': {}
    },
    {
        'name': '非市场机组实时总出力爬虫',
        'crawler': NonMarketTotalOutputCrawler,
        'params': {}
    }
]

def run_crawler(crawler_info, start_date, end_date, retry_days=3):
    """
    运行单个爬虫

    Args:
        crawler_info: 爬虫信息
        start_date: 开始日期
        end_date: 结束日期
        retry_days: 重试天数

    Returns:
        bool: 爬虫是否成功运行
    """
    name = crawler_info['name']
    crawler_class = crawler_info['crawler']
    params = crawler_info.get('params', {})
    
    logger.info(f'开始运行爬虫: {name}')
    
    try:
        # 创建爬虫实例
        crawler = crawler_class(**params)
        
        # 运行爬虫
        logger.info(f'时间范围: {start_date} 至 {end_date}')
        result = crawler.run(start_date, end_date)
        
        if result:
            logger.info(f'爬虫 {name} 运行成功')
            return True
        else:
            logger.warning(f'爬虫 {name} 运行失败或未获取数据')
            return False
    except Exception as e:
        logger.error(f'爬虫 {name} 运行异常: {e}', exc_info=True)
        return False


def run_all_crawlers_parallel(crawlers, start_date, end_date, retry_days=3, max_workers=CONCURRENCY):
    """
    并行运行所有爬虫

    Args:
        crawlers: 爬虫列表
        start_date: 开始日期
        end_date: 结束日期
        retry_days: 重试天数
        max_workers: 最大工作线程数

    Returns:
        dict: 爬虫运行结果
    """
    results = {}
    
    logger.info(f'使用并行模式运行爬虫，最大工作线程数: {max_workers}')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_crawler = {
            executor.submit(run_crawler, crawler, start_date, end_date, retry_days): crawler['name']
            for crawler in crawlers
        }
        
        for future in concurrent.futures.as_completed(future_to_crawler):
            crawler_name = future_to_crawler[future]
            try:
                result = future.result()
                results[crawler_name] = result
                logger.info(f'爬虫 {crawler_name} 已完成')
            except Exception as e:
                results[crawler_name] = False
                logger.error(f'爬虫 {crawler_name} 运行异常: {e}', exc_info=True)
    
    return results


def run_all_crawlers_serial(crawlers, start_date, end_date, retry_days=3):
    """
    串行运行所有爬虫

    Args:
        crawlers: 爬虫列表
        start_date: 开始日期
        end_date: 结束日期
        retry_days: 重试天数

    Returns:
        dict: 爬虫运行结果
    """
    results = {}
    
    logger.info('使用串行模式运行爬虫')
    
    for crawler in crawlers:
        crawler_name = crawler['name']
        result = run_crawler(crawler, start_date, end_date, retry_days)
        results[crawler_name] = result
        logger.info(f'爬虫 {crawler_name} 已完成')
    
    return results


def parse_arguments():
    """
    解析命令行参数

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='运行爬虫，获取数据')
    
    # 日期范围参数
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--days', type=int, default=1,
                       help='获取最近几天的数据，默认为1天')
    group.add_argument('--start-date', type=str,
                       help='开始日期，格式为YYYY-MM-DD')
    
    parser.add_argument('--end-date', type=str,
                       help='结束日期，格式为YYYY-MM-DD，默认为当天')
    
    # 其他参数
    parser.add_argument('--parallel', action='store_true',
                       help='使用并行模式运行爬虫')
    parser.add_argument('--retry', type=int, default=3,
                       help='重试天数，默认为3天')
    parser.add_argument('--crawler', type=str,
                       help='只运行指定的爬虫，使用爬虫名称')
    
    return parser.parse_args()


def main():
    """主程序入口"""
    args = parse_arguments()
    
    # 设置重试天数
    retry_days = args.retry
    logger.info(f'设置重试天数: {retry_days}')
    
    # 计算开始和结束日期
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    if args.start_date:
        start_date = args.start_date
    else:
        start_date_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=args.days - 1)
        start_date = start_date_dt.strftime('%Y-%m-%d')
    
    logger.info(f'开始运行爬虫，时间范围: {start_date} 至 {end_date}')
    
    # 根据命令行参数筛选爬虫
    selected_crawlers = CRAWLERS
    if args.crawler:
        selected_crawlers = [c for c in CRAWLERS if c['name'] == args.crawler]
        if not selected_crawlers:
            logger.error(f'未找到名为 {args.crawler} 的爬虫')
            return
    
    # 统计爬虫数量
    crawler_count = len(selected_crawlers)
    logger.info(f'共有 {crawler_count} 个爬虫需要运行')
    
    # 记录开始时间
    start_time = time.time()
    
    # 运行爬虫
    if args.parallel:
        results = run_all_crawlers_parallel(selected_crawlers, start_date, end_date, retry_days)
    else:
        results = run_all_crawlers_serial(selected_crawlers, start_date, end_date, retry_days)
    
    # 计算运行时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f'爬虫运行完成，耗时: {elapsed_time:.2f} 秒')
    
    # 统计结果
    success_count = sum(1 for result in results.values() if result)
    failure_count = crawler_count - success_count
    logger.info(f'成功: {success_count}, 失败: {failure_count}')
    
    # 打印失败的爬虫
    if failure_count > 0:
        logger.warning('以下爬虫运行失败:')
        for crawler_name, result in results.items():
            if not result:
                logger.warning(f'  - {crawler_name}: 未知错误')


if __name__ == '__main__':
    main() 