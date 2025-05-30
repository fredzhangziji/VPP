import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib # 用于保存和加载模型
import os
import mlflow # 新增导入
import mlflow.xgboost # 新增导入

MODEL_DIR = 'models' # 模型保存的目录
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_wind_power_model.json')

def train_model(X_train, y_train, X_test, y_test):
    """训练 XGBoost 模型并评估"""
    if X_train.empty or y_train.empty:
        print("警告: 训练数据为空。无法训练模型。")
        return None

    # 启动 MLflow run
    with mlflow.start_run():
        print("启动 MLflow Run...")
        # 开启 XGBoost 自动记录功能
        # 这会自动记录参数、指标、模型等
        mlflow.xgboost.autolog()

        print("开始训练 XGBoost 模型...")
        model = xgb.XGBRegressor(
            objective='reg:squarederror', # 回归任务
            n_estimators=1000,             # 树的数量
            learning_rate=0.05,            # 学习率
            max_depth=5,                  # 树的最大深度
            subsample=0.8,                # 训练每棵树的样本比例
            colsample_bytree=0.8,         # 构建每棵树时特征的比例
            random_state=42,
            n_jobs=-1,                     # 使用所有可用核心
        )
        
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  early_stopping_rounds=10, # 如果验证集上的性能在10轮内没有改善，则提前停止
                  verbose=False) # 可以设置为 True 以查看训练过程
        
        print("模型训练完成。")
        
        # 评估模型
        if not X_test.empty and not y_test.empty:
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            print(f"测试集评估结果:")
            print(f"  均方误差 (MSE): {mse:.4f}")
            print(f"  均方根误差 (RMSE): {rmse:.4f}")
            print(f"  R2 分数: {r2:.4f}")
        else:
            print("警告: 测试数据为空，跳过评估。")

    # mlflow.xgboost.autolog() 会自动记录模型，通常不需要显式调用 mlflow.xgboost.log_model()
    # 如果您想额外控制或记录其他产物，可以在这里添加
    # 例如: mlflow.log_artifact("some_local_file.txt")
    print("MLflow Run 结束。模型、参数和指标已自动记录（如果启用了autolog）。")
    return model

def save_model(model, model_path=MODEL_PATH):
    """保存训练好的模型"""
    if model is None:
        print("没有模型可保存。")
        return
    try:
        os.makedirs(MODEL_DIR, exist_ok=True) # 创建模型目录（如果不存在）
        model.save_model(model_path)
        print(f"模型已保存到 {model_path}")
    except Exception as e:
        print(f"保存模型时出错: {e}")

def load_model(model_path=MODEL_PATH):
    """加载已保存的本地 XGBoost JSON 模型"""
    try:
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 未找到。请先训练并保存模型。")
            return None
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        print(f"模型已从 {model_path} 加载")
        return model
    except Exception as e:
        print(f"加载本地模型时出错: {e}")
        return None 