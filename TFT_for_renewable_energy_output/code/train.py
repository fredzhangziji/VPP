"""
此模块实现了TFT（Temporal Fusion Transformer）模型的训练流程。

主要功能：
- 对每个城市的风电数据进行模型训练
- 使用Optuna进行超参数优化
- 使用最佳超参数训练最终模型
- 保存模型检查点
- 验证模型预测性能

该脚本作为独立程序运行，可以自动处理多个城市的风电预测模型训练。
"""

import TFT_model_tool
import optuna
import os

import logging
import pub_tools.logging_config
logger = logging.getLogger('tft_model')

if __name__ == "__main__":
    city_num = len(TFT_model_tool.CITIES_FOR_POWER)
    for i in range(city_num):
        city_for_power = TFT_model_tool.CITIES_FOR_POWER[i]
        city_for_weather = TFT_model_tool.CITIES_FOR_WEATHER[i]
        # Load and merge data
        weather_df = TFT_model_tool.get_history_weather_data_for_city(city_for_weather)
        output_df = TFT_model_tool.get_history_wind_power_for_city(city_for_power)
        merged_df = TFT_model_tool.merge_weather_and_power_df(weather_df, output_df)

        # Create the Optuna study
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        # Run optimization
        study.optimize(lambda trial: TFT_model_tool.objective(trial, city_for_power, merged_df), n_trials=30, timeout=3600)

        # Print and save best parameters
        logger.info("Best trial:")
        logger.info("  Value: %f", study.best_value)
        logger.info("  Params: ")
        for key, value in study.best_params.items():
            logger.info("    %s: %s", key, value)

        # Train final model using best parameters
        best_params = study.best_params
        final_trainer = TFT_model_tool.TFTModelTrainer(
            city=city_for_power,
            learning_rate=best_params["learning_rate"],
            hidden_size=best_params["hidden_size"],
            head_size=best_params["attention_head_size"],
            drop_out=best_params["dropout"],
            hidden_continuous_size=best_params["hidden_continuous_size"],
            max_epochs=best_params["max_epochs"]
        )
        df_preprocessed = final_trainer.preprocess_and_set_features(merged_df)
        logger.info('final dataset:')
        logger.info(df_preprocessed)
        final_trainer.build_datasets(df_preprocessed)
        final_trainer.create_loaders()
        final_model, _ = final_trainer.train()
        final_trainer.tft_model = final_model

        # 在 train.py 中训练完之后（或在 callback 中）
        os.makedirs("optuna_tft_checkpoints", exist_ok=True)
        ckpt_path = f"optuna_tft_checkpoints/{city_for_power}_best_trial.ckpt"
        final_trainer.trainer.save_checkpoint(ckpt_path)
        logger.info("Saved full model checkpoint to %s", ckpt_path)

        pred_val, actual_val = final_trainer.predict_validation()
