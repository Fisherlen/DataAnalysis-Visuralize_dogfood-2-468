"""
将SPSS文件转换为CSV文件
"""

import pandas as pd
import pyreadstat
import os

input_file = r"Data\TOP100 DATA 25年11月.SAV"
output_file = r"Data\TOP100_DATA_25年11月.csv"

df, meta = pyreadstat.read_sav(input_file)

df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"转换完成！文件已保存至: {output_file}")
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

