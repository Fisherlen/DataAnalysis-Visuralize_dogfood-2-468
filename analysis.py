"""
数据分析脚本：计算IncreaseYTD与其他指标的相关性分析
包括：1. Pearson相关系数计算
2. 典型相关分析
3. 相关性可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_csv(r"Data\TOP100_DATA_25年11月.csv", encoding='utf-8-sig')

print("数据预览:")
print(df.head())
print("\n数据形状:", df.shape)
print("\n数据列:", df.columns.tolist())

# 数据清洗：处理缺失值
print("\n数据清洗前的缺失值情况:")
print(df.isnull().sum())

# 仅选择有实际经济意义的指标进行分析
analysis_columns = ['VALUE', 'IncreaseThisYear', 'IncreaseYTD', 'MarketValue', 'MARKETVALUEGroup', 'TOP100MarketValue']

# 删除包含缺失值的行
df_clean = df[analysis_columns].dropna()
print(f"\n清洗后的数据形状: {df_clean.shape}")

# 检查数据类型
print("\n数据类型:")
print(df_clean.dtypes)

# 计算Pearson相关系数
corr_matrix = df_clean.corr()

print("\n" + "="*60)
print("Pearson相关系数矩阵")
print("="*60)
print(corr_matrix)

# 提取IncreaseYTD与其他指标的相关系数
increaseytd_corr = corr_matrix['IncreaseYTD'].sort_values(ascending=False)
print("\n" + "="*60)
print("IncreaseYTD与各指标的相关系数")
print("="*60)
print(increaseytd_corr)

# 创建相关程度判断函数
def judge_correlation(corr):
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "高相关"
    elif abs_corr >= 0.3:
        return "中等相关"
    elif abs_corr >= 0.1:
        return "低相关"
    else:
        return "不相关"

# 生成相关系数表格
print("\n" + "="*60)
print("IncreaseYTD与各指标的相关系数及判断")
print("="*60)
print(f"{'指标':<20} {'相关系数':<10} {'相关性判断'}")
print("-"*40)
for col, corr in increaseytd_corr.items():
    if col != 'IncreaseYTD':
        judge = judge_correlation(corr)
        print(f"{col:<20} {corr:<10.4f} {judge}")

# 典型相关分析 (Canonical Correlation Analysis
print("\n" + "="*60)
print("典型相关分析 (Canonical Correlation Analysis)")
print("="*60)

# 将IncreaseYTD作为因变量集合，其他指标作为自变量集合（共5个有实际经济意义的自变量）
X = df_clean[['VALUE', 'IncreaseThisYear', 'MarketValue', 'MARKETVALUEGroup', 'TOP100MarketValue']].values
Y = df_clean[['IncreaseYTD']].values

# 标准化数据
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

# 执行CCA
cca = CCA(n_components=1)
cca.fit(X_scaled, Y_scaled)

# 获取典型变量
X_c, Y_c = cca.transform(X_scaled, Y_scaled)

# 计算典型相关系数
canonical_corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
print(f"典型相关系数: {canonical_corr:.4f}")

# 获取权重
print("\n自变量的典型权重:")
for i, col in enumerate(['VALUE', 'IncreaseThisYear', 'MarketValue', 'MARKETVALUEGroup', 'TOP100MarketValue']):
    print(f"{col}: {cca.x_weights_[i][0]:.4f}")

print(f"\n因变量IncreaseYTD的典型权重:")
print(f"IncreaseYTD: {cca.y_weights_[0][0]:.4f}")

# 绘制IncreaseYTD与其他指标的Pearson系数柱状图
print("\n" + "="*60)
print("绘制相关性可视化")
print("="*60)

# 设置风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 准备绘图数据
plot_data = increaseytd_corr.drop('IncreaseYTD')
# 归一化绝对值用于颜色映射
norm = plt.Normalize(vmin=0, vmax=1)
colors = plt.cm.RdYlGn(norm(abs(plot_data.values)))

# 绘制柱状图
bars = ax.bar(plot_data.index, plot_data.values, color=colors)

# 添加颜色条
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('相关强度 (绝对值)')

# 设置标题和标签
ax.set_title('IncreaseYTD与各指标的Pearson相关系数', fontsize=16, pad=20)
ax.set_ylabel('Pearson相关系数', fontsize=12)
ax.set_xlabel('指标', fontsize=12)
ax.set_ylim(-1, 1)

# 在柱子上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom' if height > 0 else 'top')

# 旋转x轴标签
plt.xticks(rotation=45, ha='right')

# 显示网格
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 紧凑布局
plt.tight_layout()

# 保存图片
plt.savefig(r"Data\IncreaseYTD相关性分析.png", dpi=300, bbox_inches='tight')
print("图表已保存至: Data\\IncreaseYTD相关性分析.png")

# 显示图表
plt.show()

# 最终分析总结
print("\n" + "="*60)
print("分析总结")
print("="*60)
max_corr_col = plot_data.abs().idxmax()
max_corr_val = plot_data.abs().max()
print(f"IncreaseYTD相关性最高的指标是: {max_corr_col} (相关系数: {max_corr_val:.4f})")
print(f"相关性程度: {judge_correlation(max_corr_val)}")
