"""
此模块提供了风电出力Baseline模型的数据处理功能。
"""

import pandas as pd

def read_from_file(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def export_data_to_file(df: pd.DataFrame, file_path: str) -> None:
    df.to_csv(file_path, index=False)

if __name__ == '__main__':
    pass