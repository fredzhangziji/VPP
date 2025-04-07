import pandas as pd
import glob
import os

# 获取当前目录下所有 CSV 文件
csv_files = glob.glob("../data/tmp*.csv")

# 创建 ExcelWriter 对象，指定输出文件名和引擎（这里使用 xlsxwriter）
with pd.ExcelWriter("../data/weather_merged_csv.xlsx", engine="xlsxwriter") as writer:
    for csv_file in csv_files:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        # 如果文件名包含下划线，则取最后一个下划线后的内容，否则使用整个文件名
        if '_' in base_name:
            sheet_name = base_name.split('_')[-1][:31]
        else:
            sheet_name = base_name[:31]
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        # 将 DataFrame 写入 Excel 文件中的对应工作表，且不写入行索引
        df.to_excel(writer, sheet_name=sheet_name, index=False)