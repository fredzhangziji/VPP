"""
此模块实现了TFT（Temporal Fusion Transformer）模型的预测应用。

主要功能：
- 为每个城市加载训练好的模型
- 获取历史天气和风电数据用于特征生成
- 获取未来天气预报数据
- 预测未来5天的风电输出
- 将预测结果保存到CSV文件
- 将预测结果写入数据库wind_output_forecast表

该脚本作为独立程序运行，可以自动处理多个城市的风电预测。
"""

import os
import TFT_model_tool
from datetime import datetime
from pub_tools import db_tools, const, pub_tools
from datetime import datetime, timezone
from sqlalchemy import text
from pytorch_forecasting import TemporalFusionTransformer
import pandas as pd
import numpy as np

import logging
import pub_tools.logging_config
logger = logging.getLogger('tft_model')


if __name__ == "__main__":
    # list of cities to predict for
    cities = TFT_model_tool.CITIES_FOR_POWER
    # ensure output directory exists
    os.makedirs("predictions", exist_ok=True)

    for city in cities:
        logger.info("Processing city: %s", city)

        if city in TFT_model_tool.CITY_NAME_MAPPING_DICT:
            weather_city = TFT_model_tool.CITY_NAME_MAPPING_DICT[city]
        else:
            weather_city = city

        # load predicted future weather
        weather_df = TFT_model_tool.get_predicted_weather_data_for_city(weather_city)
        if weather_df.empty:
            logger.warning("No weather data for city: %s. Skipping.", city)
            continue

        # load history weather and power to compute lag features
        history_weather_df = TFT_model_tool.get_history_weather_data_for_city(weather_city)
        history_power_df = TFT_model_tool.get_history_wind_power_for_city(city)
        history_df = TFT_model_tool.merge_weather_and_power_df(history_weather_df, history_power_df)
        # add all engineered features including lag_7d and lag_30d
        history_features_df = TFT_model_tool._add_more_features(history_df)

        # prepare future rows with predicted weather and placeholder wind_output
        future_weather_df = weather_df.copy()
        # avoid timestamp overlap: only use future times beyond last history index
        last_history_time = history_features_df.index.max()
        future_weather_df = future_weather_df[future_weather_df.index > last_history_time]
        future_weather_df['wind_output'] = np.nan
        # ensure same columns and index
        df_for_predict = pd.concat([history_features_df, future_weather_df])

        trainer = TFT_model_tool.TFTModelTrainer(city)
        # preprocess full dataset (history + future) to set features, time_idx, group_id, etc.
        trainer.preprocess_and_set_features(df_for_predict)

        ckpt_path = os.path.join("optuna_tft_checkpoints", f"{city}_best_trial.ckpt")
        trainer.tft_model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

        future_df = trainer.predict_future(24 * 5)
        if hasattr(future_df, "iterrows") is False:
            # 假设 future_df 的索引为时间，内容为预测风电出力
            future_df = future_df.to_frame(name="wind_output").reset_index().rename(columns={"index": "date_time"})

        out_file = os.path.join("predictions", f"{city}_future_predictions.csv")
        future_df.to_csv(out_file, index=False)
        logger.info("Saved predictions for %s to %s", city, out_file)

        engine, _ = db_tools.get_db_connection(const.DB_CONFIG_VPP_SERVICE)
        try:
            now = datetime.now(timezone.utc).replace(tzinfo=None)  # 去掉时区信息
            model_name = 'TFT_wind_power'
            records = []
            for _, row in future_df.iterrows():
                sf_id = pub_tools.generate_snowflake_id()
                records.append({
                    "id": sf_id,
                    "city_name": city,
                    "date_time": row["date_time"],
                    "model": model_name,
                    "wind_output": row["wind_output"],
                    "fcst_time": now,
                    "similar_date": None
                })
            
            # 将记录转换为DataFrame并使用upsert_to_db函数
            records_df = pd.DataFrame(records)
            db_tools.upsert_to_db(engine, records_df, 'wind_output_forecast', update_column='wind_output')
            logger.info("Successfully upserted %d records to wind_output_forecast table.", len(records))
        finally:
            db_tools.release_db_connection(engine)