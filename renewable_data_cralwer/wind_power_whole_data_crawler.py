import crawler_tools
from pub_tools import const
from datetime import date
import argparse

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Wind Power Data Crawler")
    parser.add_argument(
        "--start_date",
        type=str,
        default=date.today().strftime("%Y-%m-%d"),
        help="起始日期，格式为 YYYY-MM-DD，默认为今天的日期"
    )

    args = parser.parse_args()
    start_date = args.start_date

    crawler_tools.fetch_multi_day_wind_power_data_for_each_city(const.NEIMENG_RENEWABLE_ENERGY_URL,
                                                          start_date=start_date)