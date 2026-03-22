import pandas as pd
import pyreadstat
import os

# 输入文件路径
input_file = r"Data\TOP100 DATA 25年11月.SAV"
# 输出文件路径
output_file = r"Data\TOP100_DATA_25年11月.csv"

# 读取SPSS文件
df, meta = pyreadstat.read_sav(input_file)

# 显示数据基本信息
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n前5行数据:")
print(df.head())

# 保存为CSV文件
df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n转换完成！")
print(f"CSV文件已保存至: {output_file}")
