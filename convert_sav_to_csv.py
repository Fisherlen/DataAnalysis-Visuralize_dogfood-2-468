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

# 读取SPSS文件
df, meta = pyreadstat.read_sav(input_file)

