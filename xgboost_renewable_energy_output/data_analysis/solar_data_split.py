import sys
import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the src directory
src_path = os.path.abspath(os.path.join(script_dir, '../src'))
# Add to sys.path
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
import data_preprocessing as dp
from sklearn.model_selection import train_test_split

def main():
    CITY_FOR_POWER_DATA = '阿拉善盟'
    # Ensure CITY_NAME_MAPPING_DICT is accessible from dp
    CITY_FOR_WEATHER_DATA = dp.CITY_NAME_MAPPING_DICT[CITY_FOR_POWER_DATA] 
    TARGET_VARIABLE = 'solar_output'

    print("1. Loading data...")
    # 1. Data Loading
    weather_df = dp.get_history_weather_data_for_city(CITY_FOR_WEATHER_DATA)
    solar_power_df = dp.get_history_solar_power_for_city(CITY_FOR_POWER_DATA)
    merged_df = dp.merge_weather_and_power_df(weather_df, solar_power_df)

    print("2. Preprocessing data...")
    # 2. Data Preprocessing
    preprocessed_df = dp.preprocess_data(merged_df, CITY_FOR_POWER_DATA)
    time_wise_df = dp.set_time_wise_feature(preprocessed_df.copy())

    # 3. Prepare features and target variable
    y = time_wise_df[TARGET_VARIABLE]
    X_raw = time_wise_df.drop(columns=[TARGET_VARIABLE, 'time_idx', 'group_id'], errors='ignore')

    # Define and one-hot encode categorical features
    categorical_features_def = ['wind_season', 'year', 'month', 'day', 'hour']
    actual_categorical_features = [col for col in categorical_features_def if col in X_raw.columns]

    print("3. Encoding categorical features...")
    X_encoded = pd.get_dummies(X_raw, columns=actual_categorical_features, drop_first=True)
    
    print("4. Splitting data...")
    # 4. Data Splitting (Train-Validation-Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=False # 0.25 * 0.8 = 0.2 of total
    )

    # 5. Combine features and labels for analysis
    solar_train_df = X_train.copy()
    solar_train_df['solar_output'] = y_train

    solar_val_df = X_val.copy()
    solar_val_df['solar_output'] = y_val

    solar_test_df = X_test.copy()
    solar_test_df['solar_output'] = y_test

    # Print dataset sizes
    print(f"太阳能数据集大小:")
    print(f"训练集: {solar_train_df.shape}")
    print(f"验证集: {solar_val_df.shape}")
    print(f"测试集: {solar_test_df.shape}")
    
    # 6. Save datasets to CSV files
    print("5. Saving datasets to CSV files...")
    solar_train_df.to_csv('solar_train_df.csv')
    solar_val_df.to_csv('solar_val_df.csv')
    solar_test_df.to_csv('solar_test_df.csv')
    
    print("Done! Datasets saved to CSV files.")
    
    return solar_train_df, solar_val_df, solar_test_df

if __name__ == "__main__":
    solar_train_df, solar_val_df, solar_test_df = main() 