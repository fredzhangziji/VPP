import crawler_tools
from pub_tools import const
from datetime import date
import pandas as pd


if __name__ == '__main__': 
    today_date = pd.to_datetime(date.today(), format="%Y-%m-%d")
    start_date = (today_date + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end_date = (today_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    crawler_tools.fetch_multi_day_wind_power_data_for_each_city(const.NEIMENG_RENEWABLE_ENERGY_URL,
                                                                 start_date,
                                                                 end_date)