"""
主程序入口，用于调用各个爬虫
"""

import os
import sys
import time
import argparse
import concurrent.futures
from datetime import datetime, timedelta

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
from crawlers.non_market_solar_output_crawler import NonMarketSolarOutputCrawler
from crawlers.non_market_wind_output_crawler import NonMarketWindOutputCrawler
from crawlers.non_market_nuclear_output_crawler import NonMarketNuclearOutputCrawler
from crawlers.non_market_hydro_output_crawler import NonMarketHydroOutputCrawler
from crawlers.day_ahead_price_crawler import DayAheadPriceCrawler
from crawlers.day_ahead_cleared_volume_crawler import DayAheadClearedVolumeCrawler
from crawlers.real_time_market_price_crawler import RealTimeMarketPriceCrawler
from crawlers.spot_cleared_volume_crawler import SpotClearedVolumeCrawler
from crawlers.fixed_unit_generation_plan_crawler import FixedUnitGenerationPlanCrawler

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
    },
    {
        'name': '非市场光伏实时总出力爬虫',
        'crawler': NonMarketSolarOutputCrawler,
        'params': {}
    },
    {
        'name': '非市场风电实时总出力爬虫',
        'crawler': NonMarketWindOutputCrawler,
        'params': {}
    },
    {
        'name': '非市场核电实时总出力爬虫',
        'crawler': NonMarketNuclearOutputCrawler,
        'params': {}
    },
    {
        'name': '非市场水电实时总出力爬虫',
        'crawler': NonMarketHydroOutputCrawler,
        'params': {}
    },
    {
        'name': '日前市场出清负荷侧电价爬虫',
        'crawler': DayAheadPriceCrawler,
        'params': {}
    },
    {
        'name': '日前市场出清总电量爬虫',
        'crawler': DayAheadClearedVolumeCrawler,
        'params': {}
    },
    {
        'name': '实时市场出清负荷侧电价爬虫',
        'crawler': RealTimeMarketPriceCrawler,
        'params': {}
    },
    {
        'name': '实时市场出清总电量爬虫',
        'crawler': SpotClearedVolumeCrawler,
        'params': {}
    },
    {
        'name': '固定出力机组发电计划爬虫',
        'crawler': FixedUnitGenerationPlanCrawler,
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
        DataFrame: 爬虫获取的数据
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
        df = crawler.fetch_data(start_date, end_date)
        
        if not df.empty:
            logger.info(f'爬虫 {name} 运行成功，获取 {len(df)} 条数据')
            return df
        else:
            logger.warning(f'爬虫 {name} 运行失败，未获取到数据')
            return df
    except Exception as e:
        logger.error(f'爬虫 {name} 运行异常: {e}', exc_info=True)
        return None
    finally:
        logger.info(f'爬虫 {name} 已完成')


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
                results[crawler_name] = None
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
    """
    parser = argparse.ArgumentParser(description='浙江电力市场数据爬虫')
    parser.add_argument('--start-date', help='开始日期，格式为YYYY-MM-DD')
    parser.add_argument('--end-date', help='结束日期，格式为YYYY-MM-DD')
    crawler_help = '要运行的爬虫名称，支持的值：all, week_ahead, day_ahead, actual_load, system_backup, total_generation_forecast, external_power_plan, non_market_solar_forecast, non_market_wind_forecast, non_market_nuclear_forecast, non_market_hydro_forecast, day_ahead_solar_total_forecast, day_ahead_wind_total_forecast, week_ahead_pumped_storage_forecast, day_ahead_hydro_total_forecast, day_ahead_pumped_storage_forecast, actual_total_generation, actual_solar_output, actual_wind_output, actual_hydro_output, actual_pumped_storage_output, non_market_total_output, non_market_solar_output, non_market_wind_output, non_market_nuclear_output, non_market_hydro_output, day_ahead_price, day_ahead_cleared_volume, real_time_market_price, spot_cleared_volume, fixed_plan'
    parser.add_argument('--crawler', help=crawler_help)
    parser.add_argument('--crawlers', dest='crawler', help=crawler_help)  # 添加别名
    parser.add_argument('--retry-days', type=int, default=3, help='重试天数')
    parser.add_argument('--serial', action='store_false', dest='parallel', help='是否串行运行爬虫')
    parser.add_argument('--max-workers', type=int, default=CONCURRENCY, help='最大并行工作线程数')
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 添加开始计时
    start_time = time.time()
    
    args = parse_arguments()
    
    # 设置开始日期和结束日期
    start_date = args.start_date
    end_date = args.end_date
    retry_days = args.retry_days
    
    logger.info(f"设置重试天数: {retry_days}")
    
    # 如果未指定开始日期，则使用当前日期减去重试天数
    if not start_date:
        start_date = (datetime.now() - timedelta(days=retry_days)).strftime('%Y-%m-%d')
    
    # 如果未指定结束日期，则使用当前日期加上retry_days天数
    if not end_date:
        end_date = (datetime.now() + timedelta(days=retry_days)).strftime('%Y-%m-%d')
    
    logger.info(f"开始运行爬虫，时间范围: {start_date} 至 {end_date}")
    
    # 选择要运行的爬虫
    selected_crawlers = []
    if not args.crawler or args.crawler == 'all':
        selected_crawlers = CRAWLERS
    else:
        # 根据爬虫名称选择要运行的爬虫
        crawler_name_map = {
            'week_ahead_load': '周前负荷预测爬虫',
            'day_ahead_load': '日前负荷预测爬虫',
            'actual_load': '实际负荷爬虫',
            'system_backup': '系统备用爬虫',
            'total_generation_forecast': '发电总出力预测爬虫',
            'external_power_plan': '外来电受电计划爬虫',
            'non_market_solar_forecast': '非市场光伏出力预测爬虫',
            'non_market_wind_forecast': '非市场风电出力预测爬虫',
            'non_market_nuclear_forecast': '非市场核电出力预测爬虫',
            'non_market_hydro_forecast': '非市场水电出力预测爬虫',
            'day_ahead_solar_total_forecast': '光伏总出力预测爬虫',
            'day_ahead_wind_total_forecast': '风电总出力预测爬虫',
            'week_ahead_pumped_storage_forecast': '抽蓄总出力预测爬虫',
            'day_ahead_hydro_total_forecast': '水电总出力预测爬虫',
            'day_ahead_pumped_storage_forecast': '日抽蓄总出力预测爬虫',
            'actual_total_generation': '发电实时总出力爬虫',
            'actual_solar_output': '光伏实时总出力爬虫',
            'actual_wind_output': '风电实时总出力爬虫',
            'actual_hydro_output': '水电实时总出力爬虫',
            'actual_pumped_storage_output': '抽蓄实时总出力爬虫',
            'non_market_total_output': '非市场机组实时总出力爬虫',
            'non_market_solar_output': '非市场光伏实时总出力爬虫',
            'non_market_wind_output': '非市场风电实时总出力爬虫',
            'non_market_nuclear_output': '非市场核电实时总出力爬虫',
            'non_market_hydro_output': '非市场水电实时总出力爬虫',
            'day_ahead_price': '日前市场出清负荷侧电价爬虫',
            'day_ahead_cleared_volume': '日前市场出清总电量爬虫',
            'real_time_market_price': '实时市场出清负荷侧电价爬虫',
            'spot_cleared_volume': '实时市场出清总电量爬虫',
            'fixed_plan': '固定出力机组发电计划爬虫'
        }
        
        if args.crawler in crawler_name_map:
            crawler_name = crawler_name_map[args.crawler]
            selected_crawler = None
            for crawler in CRAWLERS:
                if crawler['name'] == crawler_name:
                    selected_crawler = crawler
                    break
            
            if selected_crawler:
                selected_crawlers = [selected_crawler]
            else:
                logger.error(f"未找到名为 {crawler_name} 的爬虫")
                return
        else:
            logger.error(f"未找到名为 {args.crawler} 的爬虫")
            return
    
    # 记录运行模式
    run_mode = "并行" if args.parallel else "串行"
    
    # 并行运行爬虫
    if args.parallel:
        logger.info(f"并行运行 {len(selected_crawlers)} 个爬虫，最大并行数: {args.max_workers}")
        run_all_crawlers_parallel(selected_crawlers, start_date, end_date, retry_days=retry_days, max_workers=args.max_workers)
    else:
        logger.info(f"串行运行 {len(selected_crawlers)} 个爬虫")
        run_all_crawlers_serial(selected_crawlers, start_date, end_date, retry_days=retry_days)
    
    # 添加结束计时
    end_time = time.time()
    total_time = end_time - start_time
    
    # 记录总运行时间
    logger.info(f"爬虫运行完成 - 运行模式: {run_mode}, 总运行时间: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")


def main_test(crawler_name, start_date, end_date):
    """
    测试运行单个爬虫

    Args:
        crawler_name: 爬虫名称
        start_date: 开始日期
        end_date: 结束日期
    """
    # 查找匹配的爬虫
    crawler_info = None
    for c in CRAWLERS:
        if c['crawler'].__name__.lower() == crawler_name.lower():
            crawler_info = c
            break
    
    if crawler_info is None:
        logger.error(f'未找到爬虫: {crawler_name}')
        print(f'未找到爬虫: {crawler_name}')
        print('可用爬虫:')
        for c in CRAWLERS:
            print(f'- {c["crawler"].__name__}')
        return
    
    # 运行爬虫
    df = run_crawler(crawler_info, start_date, end_date)
    
    if df is not None and not df.empty:
        print(f'成功获取 {len(df)} 条数据')
        print('数据预览:')
        print(df.head())
        print('数据统计:')
        print(df.describe())
    else:
        print('未获取到数据')


if __name__ == '__main__':
    args = parse_arguments()
    
    if args.crawler:
        # 测试运行单个爬虫
        main_test(args.crawler, args.start_date, args.end_date)
    else:
        main() 