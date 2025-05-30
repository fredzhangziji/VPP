"""
所有项目公用的常量配置。
"""

# 数据库连接配置
DB_CONFIG_VPP_SERVICE = {
    'host': '10.5.0.10',
    'user': 'root',
    'password': 'kunyu2023rds',
    'database': 'vpp_service',
    'port': 3306
}

DB_CONFIG_VPP_USER = {
    'host': '10.5.0.10',
    'user': 'root',
    'password': 'kunyu2023rds',
    'database': 'vpp_user',
    'port': 3306
}

WEATHER_FEATURE = [
    't2m',      # temperature at 2 meters
    'ws100m',    # wind speed at 100 meters
    'wd100m',    # wind direction at 100 meters
    'sp'        # surface pressure
]

REGIONS_FOR_DB = [
    '呼和浩特',
    '包头',
    '乌海',
    '鄂尔多斯+薛家湾',
    '巴彦淖尔',
    '乌兰察布',
    '锡林郭勒',
    '阿拉善盟'
]

REGIONS_FOR_CRAWLER = [
    "'呼和浩特'",
    "'包头'",
    "'乌海'",
    "'鄂尔多斯','薛家湾'",
    "'巴彦淖尔'",
    "'乌兰察布'",
    "'锡林郭勒'",
    "'阿拉善盟'"
]

REGIONS_FOR_WEATHER = [
    '呼和浩特市',
    '包头市',
    '乌海市',
    '鄂尔多斯市',
    '巴彦淖尔市',
    '乌兰察布市',
    '锡林郭勒盟',
    '阿拉善盟'
]

CITY_POS = {
    '呼和浩特': [111.755509, 40.848423],
    '包头': [109.846544, 40.662929],
    '乌海': [106.800391, 39.662006],
    '鄂尔多斯+薛家湾': [109.787443, 39.614482],
    '巴彦淖尔': [107.394398, 40.749359],
    '乌兰察布': [113.139468, 41.000748],
    '锡林郭勒': [116.054391, 43.939423],
    '阿拉善盟': [105.735377, 38.858276]
}

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "zh-CN,zh;q=0.9,zh-TW;q=0.8,en;q=0.7",
    "Authorization": "",
    "Connection": "keep-alive",
    "Content-Type": "application/json;charset=UTF-8",
    "Cookie": "Token=; NMJY=2207.61701.194.0000; sidebarStatus=1",
    "Host": "www.imptc.com",
    "Origin": "https://www.imptc.com",
    "Referer": "https://www.imptc.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "sec-ch-ua": "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\""
}

WIND_POWER_NAME = '1'
SOLAR_POWER_NAME = '2'

NEIMENG_WIND_TABLE_NAME = 'neimeng_wind_power'
NEIMENG_SOLAR_TABLE_NAME = 'neimeng_solar_power'

NEIMENG_RENEWABLE_ENERGY_URL = 'https://www.imptc.com/api/sctjfxyyc/crqwxxfb/getXnyfdnlycData'