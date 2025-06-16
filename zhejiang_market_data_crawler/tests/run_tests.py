#!/usr/bin/env python3
"""
运行所有爬虫测试
"""

import os
import sys
import asyncio
import argparse

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger

# 设置日志记录器
logger = setup_logger('run_tests')

# 测试模块导入
try:
    import tests.test_day_ahead_load as test_day_ahead_load
    import tests.test_actual_load as test_actual_load
    import tests.test_system_backup as test_system_backup
    import tests.test_total_generation_forecast as test_total_generation_forecast
    import tests.test_external_power_plan as test_external_power_plan
    # 如果有周前负荷预测的测试模块，也可以在这里导入
    HAS_WEEK_AHEAD = False
    try:
        import tests.test_week_ahead_load as test_week_ahead_load
        HAS_WEEK_AHEAD = True
    except ImportError:
        logger.warning("未找到周前负荷预测测试模块")
    
    logger.info("成功导入测试模块")
except ImportError as e:
    logger.error(f"导入测试模块失败: {e}")
    sys.exit(1)

async def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有爬虫测试")
    
    # 运行日前负荷预测测试
    logger.info("===== 运行日前负荷预测测试 =====")
    await test_day_ahead_load.main()
    
    # 运行实际负荷测试
    logger.info("===== 运行实际负荷测试 =====")
    await test_actual_load.main()
    
    # 运行系统备用测试
    logger.info("===== 运行系统备用测试 =====")
    await test_system_backup.main()
    
    # 运行发电总出力预测测试
    logger.info("===== 运行发电总出力预测测试 =====")
    await test_total_generation_forecast.main()
    
    # 运行外来电受电计划测试
    logger.info("===== 运行外来电受电计划测试 =====")
    await test_external_power_plan.main()
    
    # 如果有周前负荷预测，也运行它
    if HAS_WEEK_AHEAD:
        logger.info("===== 运行周前负荷预测测试 =====")
        await test_week_ahead_load.main()
    
    logger.info("所有测试执行完毕")

async def run_selected_test(test_name):
    """运行选定的测试"""
    if test_name == "day_ahead":
        logger.info("===== 运行日前负荷预测测试 =====")
        await test_day_ahead_load.main()
    elif test_name == "actual_load":
        logger.info("===== 运行实际负荷测试 =====")
        await test_actual_load.main()
    elif test_name == "system_backup":
        logger.info("===== 运行系统备用测试 =====")
        await test_system_backup.main()
    elif test_name == "total_generation_forecast":
        logger.info("===== 运行发电总出力预测测试 =====")
        await test_total_generation_forecast.main()
    elif test_name == "external_power_plan":
        logger.info("===== 运行外来电受电计划测试 =====")
        await test_external_power_plan.main()
    elif test_name == "week_ahead" and HAS_WEEK_AHEAD:
        logger.info("===== 运行周前负荷预测测试 =====")
        await test_week_ahead_load.main()
    else:
        logger.error(f"未知的测试名称: {test_name}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行爬虫测试")
    parser.add_argument('--test', '-t', choices=['all', 'day_ahead', 'actual_load', 'week_ahead', 'system_backup', 'total_generation_forecast', 'external_power_plan'],
                       default='all', help='要运行的测试 (默认: all)')
    args = parser.parse_args()
    
    if args.test == 'all':
        asyncio.run(run_all_tests())
    else:
        asyncio.run(run_selected_test(args.test))

if __name__ == '__main__':
    main() 