# Informer 风力发电预测模型

基于 Informer 时间序列预测模型的风力发电输出预测系统，专为内蒙古各城市的风电场设计。

## 项目介绍

本项目使用 Informer 模型进行风力发电输出的预测。Informer 是一种高效的长序列时间序列预测模型，相比传统的 Transformer 模型，它在处理长序列数据时具有更高的效率和准确性，特别适合风电这种受多种因素影响的时间序列数据。

## 主要功能

- 基于历史天气和风力发电数据进行模型训练
- 结合未来5天天气预报数据预测风力发电输出
- 支持内蒙古多个城市的风力发电输出预测
- 提供数据可视化和预测结果导出功能
- 自动将预测结果存储到数据库

## 使用方法

### 训练模型

```bash
python src/train.py
```

### 预测未来发电量

```bash
python src/predict.py
```

## 技术栈

- PyTorch
- Informer (基于 Transformer 的时间序列预测模型)
- MLflow (实验跟踪)
- Pandas & NumPy (数据处理)
- Matplotlib & Seaborn (可视化)
- SQLAlchemy (数据库交互) 