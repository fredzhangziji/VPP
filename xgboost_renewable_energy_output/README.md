# XGBoost 风力发电预测项目

本项目旨在使用 XGBoost 模型预测单个城市的风力发电输出。

## 项目结构

```
xgboost_renewable_energy_output/
├── data/                             # 存放数据 (需要您手动创建并添加数据)
│   └── wind_data.csv                 # 示例数据文件名
├── notebooks/                        # 存放 Jupyter notebooks (可选)
├── src/                              # 存放源代码
│   ├── __init__.py
│   ├── data_preprocessing.py         # 数据预处理模块
│   ├── model_training.py           # 模型训练模块
│   ├── predict.py                    # 预测脚本
│   └── utils.py                      # 工具函数
├── main.py                           # 主执行脚本
├── requirements.txt                  # 项目依赖
└── README.md                         # 项目说明
```

## 安装

1. 克隆仓库 (如果适用)
2. 创建并激活虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   ```
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据

你需要准备一个包含以下特征的时间序列数据集 (`wind_data.csv`)：

*   `timestamp`: 时间戳
*   `wind_speed`: 风速 (m/s)
*   `wind_direction`: 风向 (度)
*   `temperature`: 温度 (°C)
*   `pressure`: 气压 (hPa)
*   `humidity`: 湿度 (%)
*   `power_output`: 风力发电输出 (MW) - **这是我们要预测的目标变量**

请将数据文件放在 `data/` 目录下。

## 使用

1.  **数据预处理**: (在 `src/data_preprocessing.py` 中实现)
    *   加载数据
    *   处理缺失值
    *   特征工程 (例如，从时间戳提取小时、月份等)
    *   数据标准化/归一化
2.  **模型训练**: (在 `src/model_training.py` 中实现)
    *   划分训练集和测试集
    *   使用 XGBoost 训练模型
    *   超参数调优 (可选)
    *   保存训练好的模型
3.  **预测**: (在 `src/predict.py` 和 `main.py` 中实现)
    *   加载训练好的模型
    *   使用新数据进行预测

运行主脚本：
```bash
python main.py
```

## 注意

这只是一个项目骨架。你需要根据你的具体数据和需求来实现各个模块的功能。 