"""
将SPSS文件转换为CSV文件
"""

import pandas as pd
import pyreadstat
import os

# 输入文件路径
input_file = r"Data\TOP100 DATA 25年11月.SAV"
# 输出文件路径
output_file = r"Data\TOP100_DATA_25年11月.csv"

try:
    # 读取SPSS文件
    df, meta = pyreadstat.read_sav(input_file)
    
    # 保存为CSV文件
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"文件已成功转换并保存至: {output_file}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
except Exception as e:
    print(f"转换过程中出现错误: {e}")

