"""
相关性分析：IncreaseYTD与其他指标的关系
使用Pearson相关系数和典型相关分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cross_decomposition import CCA
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r"Data\TOP100_DATA_25年11月.csv")
print(f"数据形状: {df.shape}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n数据预览:\n{df.head()}")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n数值型列: {numeric_cols}")

if 'IncreaseYTD' not in numeric_cols:
    print("IncreaseYTD 不是数值型，尝试转换...")
    df['IncreaseYTD'] = pd.to_numeric(df['IncreaseYTD'], errors='coerce')

target_col = 'IncreaseYTD'
other_numeric_cols = [col for col in numeric_cols if col != target_col and col not in ['序', 'STOCKID']]

print(f"\n分析目标: {target_col}")
print(f"其他数值指标: {other_numeric_cols}")

analysis_df = df[[target_col] + other_numeric_cols].dropna()
print(f"\n去除缺失值后数据量: {len(analysis_df)}")

print("\n" + "="*60)
print("Pearson相关系数分析")
print("="*60)

correlations = {}
for col in other_numeric_cols:
    corr, p_value = stats.pearsonr(analysis_df[target_col], analysis_df[col])
    correlations[col] = {'Pearson系数': corr, 'P值': p_value}
    print(f"{col}: r = {corr:.4f}, p = {p_value:.4e}")

corr_df = pd.DataFrame(correlations).T
corr_df = corr_df.dropna(subset=['Pearson系数'])
corr_df['相关强度'] = corr_df['Pearson系数'].abs().apply(
    lambda x: '高相关' if x >= 0.5 else ('中等相关' if x >= 0.3 else ('低相关' if x >= 0.1 else '不相关'))
)
corr_df = corr_df.sort_values('Pearson系数', key=abs, ascending=False)

print("\n" + "="*60)
print("相关系数汇总表")
print("="*60)
print(corr_df.to_string())

most_correlated = corr_df.index[0]
most_corr_value = corr_df.loc[most_correlated, 'Pearson系数']
print(f"\n与IncreaseYTD最相关的指标: {most_correlated}")
print(f"相关系数: {most_corr_value:.4f}")

print("\n" + "="*60)
print("典型相关分析 (Canonical Correlation Analysis)")
print("="*60)

X_cols = [col for col in other_numeric_cols if col != 'IncreaseThisYear']
Y_col = [target_col]

X = analysis_df[X_cols].values
Y = analysis_df[[target_col]].values

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

n_components = min(len(X_cols), 1)
cca = CCA(n_components=n_components)
X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)

canonical_corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
print(f"典型相关系数: {canonical_corr:.4f}")

print("\n典型变量载荷 (X变量对典型变量的贡献):")
for i, col in enumerate(X_cols):
    print(f"  {col}: {cca.x_weights_[i, 0]:.4f}")

print("\n" + "="*60)
print("相关性分析总结")
print("="*60)

def get_correlation_level(r):
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "高相关"
    elif abs_r >= 0.4:
        return "中等相关"
    elif abs_r >= 0.2:
        return "低相关"
    else:
        return "不相关"

print(f"\nIncreaseYTD与各指标的相关性分析结果:")
print("-"*50)
for idx, row in corr_df.iterrows():
    level = get_correlation_level(row['Pearson系数'])
    direction = "正相关" if row['Pearson系数'] > 0 else "负相关"
    print(f"{idx:25s}: r = {row['Pearson系数']:+.4f} ({level}, {direction})")

print(f"\n最相关指标: {most_correlated}")
print(f"典型相关系数: {canonical_corr:.4f} ({get_correlation_level(canonical_corr)})")

fig, ax = plt.subplots(figsize=(12, 8))

plot_data = corr_df.sort_values('Pearson系数', key=abs, ascending=True).copy()
plot_data = plot_data[plot_data['Pearson系数'].notna()]
indicators = plot_data.index.tolist()
pearson_values = plot_data['Pearson系数'].values
abs_values = np.abs(pearson_values)

max_abs = max(abs_values) if max(abs_values) > 0 else 1
colors = plt.cm.RdYlGn(abs_values / max_abs)

bars = ax.barh(indicators, pearson_values, color=colors, edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, pearson_values):
    if np.isfinite(val):
        ax.text(val + 0.02 if val >= 0 else val - 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=10, fontweight='bold')

ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5, label='中等相关阈值(0.3)')
ax.axvline(x=-0.3, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='高相关阈值(0.5)')
ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.7)

ax.set_xlabel('Pearson相关系数', fontsize=12)
ax.set_ylabel('指标', fontsize=12)
ax.set_title('IncreaseYTD与各指标的Pearson相关系数\n(颜色深浅表示相关强度)', fontsize=14, fontweight='bold')

sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='相关系数绝对值')

ax.legend(loc='lower right')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(r'Data\correlation_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存至: Data\correlation_analysis.png")
plt.show()
