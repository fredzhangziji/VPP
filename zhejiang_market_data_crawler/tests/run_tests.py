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
    import tests.test_non_market_solar_forecast as test_non_market_solar_forecast
    import tests.test_non_market_wind_forecast as test_non_market_wind_forecast
    import tests.test_non_market_nuclear_forecast as test_non_market_nuclear_forecast
    import tests.test_non_market_hydro_forecast as test_non_market_hydro_forecast
    import tests.test_day_ahead_solar_total_forecast as test_day_ahead_solar_total_forecast
    import tests.test_day_ahead_wind_total_forecast as test_day_ahead_wind_total_forecast
    import tests.test_week_ahead_pumped_storage_forecast as test_week_ahead_pumped_storage_forecast
    import tests.test_day_ahead_hydro_total_forecast as test_day_ahead_hydro_total_forecast
    import tests.test_actual_solar_output as test_actual_solar_output
    import tests.test_actual_wind_output as test_actual_wind_output
    import tests.test_actual_hydro_output as test_actual_hydro_output
    import tests.test_actual_pumped_storage_output as test_actual_pumped_storage_output
    import tests.test_non_market_total_output as test_non_market_total_output
    import tests.test_non_market_solar_output as test_non_market_solar_output
    import tests.test_non_market_wind_output as test_non_market_wind_output
    import tests.test_non_market_nuclear_output as test_non_market_nuclear_output
    import tests.test_non_market_hydro_output as test_non_market_hydro_output
    import tests.test_day_ahead_price as test_day_ahead_price
    import tests.test_actual_total_generation as test_actual_total_generation
    import tests.test_day_ahead_pumped_storage_forecast as test_day_ahead_pumped_storage_forecast
    import tests.test_week_ahead_load as test_week_ahead_load
    import tests.test_day_ahead_cleared_volume as test_day_ahead_cleared_volume
    import tests.test_real_time_market_price as test_real_time_market_price
    import tests.test_spot_cleared_volume as test_spot_cleared_volume
    import tests.test_fixed_unit_generation_plan_crawler as test_fixed_unit_generation_plan
    
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
    
    # 运行非市场光伏出力预测测试
    logger.info("===== 运行非市场光伏出力预测测试 =====")
    await test_non_market_solar_forecast.main()
    
    # 运行非市场风电出力预测测试
    logger.info("===== 运行非市场风电出力预测测试 =====")
    await test_non_market_wind_forecast.main()
    
    # 运行非市场核电出力预测测试
    logger.info("===== 运行非市场核电出力预测测试 =====")
    await test_non_market_nuclear_forecast.main()
    
    # 运行非市场水电出力预测测试
    logger.info("===== 运行非市场水电出力预测测试 =====")
    await test_non_market_hydro_forecast.main()
    
    # 运行光伏总出力预测测试
    logger.info("===== 运行光伏总出力预测测试 =====")
    await test_day_ahead_solar_total_forecast.main()
    
    # 运行风电总出力预测测试
    logger.info("===== 运行风电总出力预测测试 =====")
    await test_day_ahead_wind_total_forecast.main()
    
    # 运行抽蓄总出力预测测试
    logger.info("===== 运行抽蓄总出力预测测试 =====")
    await test_week_ahead_pumped_storage_forecast.main()
    
    # 运行水电总出力预测测试
    logger.info("===== 运行水电总出力预测测试 =====")
    await test_day_ahead_hydro_total_forecast.main()
    
    # 运行光伏实时总出力测试
    logger.info("===== 运行光伏实时总出力测试 =====")
    await test_actual_solar_output.main()
    
    # 运行风电实时总出力测试
    logger.info("===== 运行风电实时总出力测试 =====")
    await test_actual_wind_output.main()
    
    # 运行水电实时总出力测试
    logger.info("===== 运行水电实时总出力测试 =====")
    await test_actual_hydro_output.main()
    
    # 运行抽蓄实时总出力测试
    logger.info("===== 运行抽蓄实时总出力测试 =====")
    await test_actual_pumped_storage_output.main()
    
    # 运行非市场机组实时总出力测试
    logger.info("===== 运行非市场机组实时总出力测试 =====")
    await test_non_market_total_output.main()
    
    # 运行非市场光伏实时总出力测试
    logger.info("===== 运行非市场光伏实时总出力测试 =====")
    await test_non_market_solar_output.main()
    
    # 运行非市场风电实时总出力测试
    logger.info("===== 运行非市场风电实时总出力测试 =====")
    await test_non_market_wind_output.main()
    
    # 运行非市场核电实时总出力测试
    logger.info("===== 运行非市场核电实时总出力测试 =====")
    await test_non_market_nuclear_output.main()
    
    # 运行非市场水电实时总出力测试
    logger.info("===== 运行非市场水电实时总出力测试 =====")
    await test_non_market_hydro_output.main()
    
    # 运行日前市场出清负荷侧电价测试
    logger.info("===== 运行日前市场出清负荷侧电价测试 =====")
    await test_day_ahead_price.main()
    
    # 运行实际发电总出力测试
    logger.info("===== 运行实际发电总出力测试 =====")
    await test_actual_total_generation.main()
    
    # 运行日前抽蓄出力预测测试
    logger.info("===== 运行日前抽蓄出力预测测试 =====")
    await test_day_ahead_pumped_storage_forecast.main()
    
    # 运行周前负荷预测测试
    logger.info("===== 运行周前负荷预测测试 =====")
    await test_week_ahead_load.main()
    
    # 运行日前市场出清总电量测试
    logger.info("===== 运行日前市场出清总电量测试 =====")
    await test_day_ahead_cleared_volume.main()
    
    # 运行实时市场出清负荷侧电价测试
    logger.info("===== 运行实时市场出清负荷侧电价测试 =====")
    await test_real_time_market_price.main()
    
    # 运行实时市场出清总电量测试
    logger.info("===== 运行实时市场出清总电量测试 =====")
    await test_spot_cleared_volume.main()
    
    # 运行固定出力机组发电计划测试
    logger.info("===== 运行固定出力机组发电计划测试 =====")
    await test_fixed_unit_generation_plan.main()
    
    logger.info("所有测试执行完毕")

async def run_selected_test(test_name):
    """运行选定的测试"""
    if test_name == "day_ahead_load":
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
    elif test_name == "non_market_solar_forecast":
        logger.info("===== 运行非市场光伏出力预测测试 =====")
        await test_non_market_solar_forecast.main()
    elif test_name == "non_market_wind_forecast":
        logger.info("===== 运行非市场风电出力预测测试 =====")
        await test_non_market_wind_forecast.main()
    elif test_name == "non_market_nuclear_forecast":
        logger.info("===== 运行非市场核电出力预测测试 =====")
        await test_non_market_nuclear_forecast.main()
    elif test_name == "non_market_hydro_forecast":
        logger.info("===== 运行非市场水电出力预测测试 =====")
        await test_non_market_hydro_forecast.main()
    elif test_name == "day_ahead_solar_total_forecast":
        logger.info("===== 运行光伏总出力预测测试 =====")
        await test_day_ahead_solar_total_forecast.main()
    elif test_name == "day_ahead_wind_total_forecast":
        logger.info("===== 运行风电总出力预测测试 =====")
        await test_day_ahead_wind_total_forecast.main()
    elif test_name == "week_ahead_pumped_storage_forecast":
        logger.info("===== 运行抽蓄总出力预测测试 =====")
        await test_week_ahead_pumped_storage_forecast.main()
    elif test_name == "day_ahead_hydro_total_forecast":
        logger.info("===== 运行水电总出力预测测试 =====")
        await test_day_ahead_hydro_total_forecast.main()
    elif test_name == "actual_solar_output":
        logger.info("===== 运行光伏实时总出力测试 =====")
        await test_actual_solar_output.main()
    elif test_name == "actual_wind_output":
        logger.info("===== 运行风电实时总出力测试 =====")
        await test_actual_wind_output.main()
    elif test_name == "actual_hydro_output":
        logger.info("===== 运行水电实时总出力测试 =====")
        await test_actual_hydro_output.main()
    elif test_name == "actual_pumped_storage_output":
        logger.info("===== 运行抽蓄实时总出力测试 =====")
        await test_actual_pumped_storage_output.main()
    elif test_name == "non_market_total_output":
        logger.info("===== 运行非市场机组实时总出力测试 =====")
        await test_non_market_total_output.main()
    elif test_name == "non_market_solar_output":
        logger.info("===== 运行非市场光伏实时总出力测试 =====")
        await test_non_market_solar_output.main()
    elif test_name == "non_market_wind_output":
        logger.info("===== 运行非市场风电实时总出力测试 =====")
        await test_non_market_wind_output.main()
    elif test_name == "non_market_nuclear_output":
        logger.info("===== 运行非市场核电实时总出力测试 =====")
        await test_non_market_nuclear_output.main()
    elif test_name == "non_market_hydro_output":
        logger.info("===== 运行非市场水电实时总出力测试 =====")
        await test_non_market_hydro_output.main()
    elif test_name == "day_ahead_price":
        logger.info("===== 运行日前市场出清负荷侧电价测试 =====")
        await test_day_ahead_price.main()
    elif test_name == "actual_total_generation":
        logger.info("===== 运行实际发电总出力测试 =====")
        await test_actual_total_generation.main()
    elif test_name == "fixed_plan":
        logger.info("===== 运行固定出力机组发电计划测试 =====")
        await test_fixed_unit_generation_plan.main()
    elif test_name == "day_ahead_pumped_storage_forecast":
        logger.info("===== 运行日前抽蓄出力预测测试 =====")
        await test_day_ahead_pumped_storage_forecast.main()
    elif test_name == "week_ahead_load":
        logger.info("===== 运行周前负荷预测测试 =====")
        await test_week_ahead_load.main()
    elif test_name == "day_ahead_cleared_volume":
        logger.info("===== 运行日前市场出清总电量测试 =====")
        await test_day_ahead_cleared_volume.main()
    elif test_name == "real_time_market_price":
        logger.info("===== 运行实时市场出清负荷侧电价测试 =====")
        await test_real_time_market_price.main()
    elif test_name == "spot_cleared_volume":
        logger.info("===== 运行实时市场出清总电量测试 =====")
        await test_spot_cleared_volume.main()
    else:
        logger.error(f"未知的测试名称: {test_name}")

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行爬虫测试')
    parser.add_argument('test', nargs='?', help='指定要运行的测试，如果不指定则运行所有测试')
    parser.add_argument('--test', dest='test_opt', help='指定要运行的测试（使用选项方式）')
    args = parser.parse_args()
    
    # 确定要运行的测试（优先使用位置参数）
    test_name = args.test or args.test_opt
    
    # 运行测试
    if test_name:
        asyncio.run(run_selected_test(test_name))
    else:
        asyncio.run(run_all_tests())

if __name__ == "__main__":
    main()