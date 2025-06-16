# 浙江电力市场数据爬虫系统

该系统用于从浙江电力交易中心网站获取电力市场相关数据，包括周前负荷预测、日前负荷预测、实际负荷等关键指标。

## 功能特点

- 支持获取浙江全省的负荷预测和实际负荷数据
- 可配置的时间范围，支持获取指定日期区间的数据
- 支持并行和串行两种运行模式
- 使用INSERT...ON DUPLICATE KEY UPDATE语法处理重复数据
- 模块化的爬虫设计，便于扩展新的数据源

## 系统结构

```
VPP/zhejiang_market_data_crawler/
├── config.yaml.example   # 配置文件模板
├── crawlers/             # 爬虫模块
│   ├── __init__.py
│   ├── actual_load_crawler.py      # 实际负荷爬虫
│   ├── base_crawler.py             # 基础爬虫类
│   ├── day_ahead_load_crawler.py   # 日前负荷预测爬虫
│   ├── json_crawler.py             # JSON爬虫基类
│   ├── sample_json_crawler.py      # 示例JSON爬虫
│   └── week_ahead_load_crawler.py  # 周前负荷预测爬虫
├── data/                # 数据目录
├── logs/                # 日志目录
├── main.py              # 主程序
├── pub_tools/           # 公共工具
│   ├── __init__.py
│   └── db_tools.py      # 数据库工具
├── README.md            # 项目说明文档
├── requirements.txt     # 依赖包列表
├── tests/               # 测试目录
│   ├── __init__.py
│   ├── run_tests.py              # 测试运行脚本
│   ├── test_actual_load.py       # 实际负荷爬虫测试
│   ├── test_day_ahead_load.py    # 日前负荷预测爬虫测试
│   └── test_week_ahead_load.py   # 周前负荷预测爬虫测试
└── utils/               # 工具模块
    ├── __init__.py
    ├── config.py        # 配置工具
    ├── db_helper.py     # 数据库操作辅助工具
    ├── http_client.py   # HTTP客户端
    └── logger.py        # 日志工具
```

## 安装指南

1. 克隆代码库:
```bash
git clone <repository-url>
cd VPP/zhejiang_market_data_crawler
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 创建配置文件:
```bash
cp config.yaml.example config.yaml
```

4. 编辑配置文件 `config.yaml`，填写数据库连接信息和API Cookie。

## 使用说明

### 运行主程序

获取最近一天的数据:
```bash
python main.py
```

获取指定天数的数据:
```bash
python main.py --days 7
```

获取指定日期范围的数据:
```bash
python main.py --start-date 2024-06-01 --end-date 2024-06-07
```

使用并行模式:
```bash
python main.py --parallel
```

### 运行测试

运行所有测试:
```bash
python tests/run_tests.py
```

运行特定爬虫的测试:
```bash
python tests/run_tests.py -t actual_load
```

## 爬虫说明

### 实际负荷爬虫 (ActualLoadCrawler)

用于获取浙江省实际负荷数据，每15分钟一个数据点。

数据源: `https://zjpx.com.cn/px-settlement-infpubquery-phbzj/schedule/realLoad`

数据字段: `actual_load`

### 日前负荷预测爬虫 (DayAheadLoadCrawler)

用于获取浙江省日前负荷预测数据，每15分钟一个数据点。

数据源: `https://zjpx.com.cn/px-settlement-infpubquery-phbzj/supplyAndDemand/dailyLoad`

数据字段: `day_ahead_load_forecast`

### 周前负荷预测爬虫 (WeekAheadLoadCrawler)

用于获取浙江省周前负荷预测数据，每15分钟一个数据点。

数据源: 内部API (具体见代码)

数据字段: `week_ahead_load_forecast`

## 注意事项

- 系统需要有效的API Cookie才能正常获取数据，请在配置文件中设置。
- 数据库表需要有date_time字段作为唯一键，以便处理重复数据。
- 日志文件保存在logs目录下，按日期和模块分类。 