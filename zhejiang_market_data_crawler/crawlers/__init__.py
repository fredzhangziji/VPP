"""
爬虫模块，包含各个数据源的爬虫实现
"""

from .base_crawler import BaseCrawler
from .json_crawler import JSONCrawler
from .actual_load_crawler import ActualLoadCrawler
from .day_ahead_load_crawler import DayAheadLoadCrawler
from .week_ahead_load_crawler import WeekAheadLoadCrawler
from .system_backup_crawler import SystemBackupCrawler
from .total_generation_forecast_crawler import TotalGenerationForecastCrawler
from .external_power_plan_crawler import ExternalPowerPlanCrawler 