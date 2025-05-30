# MLflow 集成说明

本项目已集成 MLflow，实现了全流程的实验追踪、参数记录、模型管理。

## 快速开始

1. 安装依赖

```bash
conda env update -f conda.yaml
conda activate xgboost
```

2. 启动 MLflow Tracking Server

本地
```bash
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  > /Users/fredzhang/Documents/git_repo/VPP/xgboost_renewable_energy_output/src/logs/mlflow_server.log 2>&1 &
```

远端
```bash
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  > /home/elu/VPP/xgboost_renewable_energy_output/src/logs/mlflow_server.log 2>&1 &
```

3. 运行主流程并自动追踪

```bash
python main.py
```

4. 通过浏览器访问 http://127.0.0.1:5001 查看实验结果

## 远程 MLflow Tracking

如需连接远程 MLflow Tracking Server，请设置环境变量：

```bash
export MLFLOW_TRACKING_URI=http://your-mlflow-server:5001
```

## 复现实验

推荐使用 MLflow Project 方式：

```bash
mlflow run .
```

## 主要追踪内容
- 数据路径、模型路径、训练参数
- 训练过程指标（MSE、RMSE、R2等）
- 训练好的模型（artifact）
- 训练特征列

---
如需自定义追踪内容，请参考 `main.py` 和 `src/model_training.py`。
