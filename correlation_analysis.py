"""
分析IncreaseYTD与其他指标的相关系数
使用Canonical Correlation和Pearson相关系数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv(r"Data\TOP100_DATA_25年11月.csv")

print("=" * 80)
print("数据概览")
print("=" * 80)
print(f"数据形状: {df.shape}")
print(f"\n列名: {list(df.columns)}")
print(f"\n数据类型:\n{df.dtypes}")

# 选择数值型列进行分析
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\n数值型列: {numeric_cols}")

# 目标变量
target_col = 'IncreaseYTD'

# 检查目标变量是否存在
if target_col not in df.columns:
    print(f"错误: 找不到目标变量 {target_col}")
    exit(1)

# 获取其他数值型列（排除目标变量和非数值列）
other_cols = [col for col in numeric_cols if col != target_col]

# 移除标识符类列（通常相关性无意义）和日期列
id_cols = ['序', 'STOCKID', 'DATE']
analysis_cols = [col for col in other_cols if col not in id_cols]

print(f"\n分析列（排除ID类）: {analysis_cols}")

# 创建分析用的数据框（去除缺失值）
analysis_df = df[analysis_cols + [target_col]].dropna()
print(f"\n去除缺失值后的数据形状: {analysis_df.shape}")

# ============================================
# 1. Pearson相关系数计算
# ============================================
print("\n" + "=" * 80)
print("1. Pearson相关系数分析")
print("=" * 80)

correlations = []
for col in analysis_cols:
    corr, p_value = pearsonr(analysis_df[col], analysis_df[target_col])
    correlations.append({
        '指标': col,
        'Pearson相关系数': corr,
        'P值': p_value,
        '绝对值': abs(corr)
    })

# 创建相关系数DataFrame并排序
corr_df = pd.DataFrame(correlations)
corr_df = corr_df.sort_values('绝对值', ascending=False)

print("\nIncreaseYTD与各指标的Pearson相关系数（按绝对值降序）:")
print(corr_df.to_string(index=False))

# ============================================
# 2. 相关性分析总结
# ============================================
print("\n" + "=" * 80)
print("2. 相关性分析总结")
print("=" * 80)

def classify_correlation(corr_val):
    abs_corr = abs(corr_val)
    if abs_corr >= 0.7:
        return '高相关'
    elif abs_corr >= 0.3:
        return '中等相关'
    elif abs_corr >= 0.1:
        return '低相关'
    else:
        return '不相关'

corr_df['相关程度'] = corr_df['Pearson相关系数'].apply(classify_correlation)

# 按相关程度分组
high_corr = corr_df[corr_df['相关程度'] == '高相关']
medium_corr = corr_df[corr_df['相关程度'] == '中等相关']
low_corr = corr_df[corr_df['相关程度'] == '低相关']
no_corr = corr_df[corr_df['相关程度'] == '不相关']

print(f"\n【高相关】(|r| >= 0.7): {len(high_corr)}个指标")
if len(high_corr) > 0:
    for _, row in high_corr.iterrows():
        print(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}")

print(f"\n【中等相关】(0.3 <= |r| < 0.7): {len(medium_corr)}个指标")
if len(medium_corr) > 0:
    for _, row in medium_corr.iterrows():
        print(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}")

print(f"\n【低相关】(0.1 <= |r| < 0.3): {len(low_corr)}个指标")
if len(low_corr) > 0:
    for _, row in low_corr.iterrows():
        print(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}")

print(f"\n【不相关】( |r| < 0.1): {len(no_corr)}个指标")
if len(no_corr) > 0:
    for _, row in no_corr.iterrows():
        print(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}")

# 找出最相关的指标
most_related = corr_df.iloc[0]
print(f"\n【最相关指标】: {most_related['指标']}")
print(f"   Pearson相关系数: {most_related['Pearson相关系数']:.4f}")
print(f"   P值: {most_related['P值']:.2e}")

# ============================================
# 3. Canonical Correlation Analysis (CCA)
# ============================================
print("\n" + "=" * 80)
print("3. Canonical Correlation Analysis (CCA)")
print("=" * 80)

# 准备数据
X = analysis_df[analysis_cols].values  # 自变量
Y = analysis_df[[target_col]].values   # 因变量

# 标准化数据
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0)

# 执行CCA分析
n_components = min(X.shape[1], Y.shape[1])
cca = CCA(n_components=n_components)
cca.fit(X_std, Y_std)

# 获取canonical相关系数
X_c, Y_c = cca.transform(X_std, Y_std)
canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]

print(f"\nCanonical Correlation系数:")
for i, corr in enumerate(canonical_corrs):
    print(f"   第{i+1}对典型变量: r = {corr:.4f}")

# CCA载荷分析
print(f"\nX变量（自变量）的Canonical Loadings:")
x_loadings = cca.x_loadings_
for i, col in enumerate(analysis_cols):
    loadings_str = ", ".join([f"CV{j+1}: {x_loadings[i, j]:.4f}" for j in range(min(3, n_components))])
    print(f"   {col}: {loadings_str}")

print(f"\nY变量（因变量）的Canonical Loadings:")
y_loadings = cca.y_loadings_
print(f"   {target_col}: " + ", ".join([f"CV{j+1}: {y_loadings[0, j]:.4f}" for j in range(min(3, n_components))]))

# ============================================
# 4. 生成可视化图表
# ============================================
print("\n" + "=" * 80)
print("4. 生成可视化图表")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 8))

# 准备数据
indicators = corr_df['指标'].tolist()
corr_values = corr_df['Pearson相关系数'].tolist()
abs_values = corr_df['绝对值'].tolist()

# 根据相关系数绝对值着色
colors = []
for val in abs_values:
    if val >= 0.7:
        colors.append('#d62728')  # 深红色 - 高相关
    elif val >= 0.3:
        colors.append('#ff7f0e')  # 橙色 - 中等相关
    elif val >= 0.1:
        colors.append('#2ca02c')  # 绿色 - 低相关
    else:
        colors.append('#1f77b4')  # 蓝色 - 不相关

# 创建横向条形图
bars = ax.barh(indicators, corr_values, color=colors, edgecolor='black', linewidth=0.5)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, corr_values)):
    width = bar.get_width()
    label_x = width + 0.01 if width >= 0 else width - 0.01
    ha = 'left' if width >= 0 else 'right'
    ax.text(label_x, bar.get_y() + bar.get_height()/2, 
            f'{val:.4f}', va='center', ha=ha, fontsize=10, fontweight='bold')

# 设置图表属性
ax.set_xlabel('Pearson Correlation Coefficient', fontsize=12)
ax.set_ylabel('Indicators', fontsize=12)
ax.set_title(f'IncreaseYTD与各指标的Pearson相关系数\n(按相关强度着色)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlim(-1, 1)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='高相关 (|r| ≥ 0.7)'),
    Patch(facecolor='#ff7f0e', label='中等相关 (0.3 ≤ |r| < 0.7)'),
    Patch(facecolor='#2ca02c', label='低相关 (0.1 ≤ |r| < 0.3)'),
    Patch(facecolor='#1f77b4', label='不相关 (|r| < 0.1)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# 添加网格线
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('correlation_bar_chart.png', dpi=300, bbox_inches='tight')
print("图表已保存为: correlation_bar_chart.png")
plt.show()

# ============================================
# 5. 保存完整结果
# ============================================
print("\n" + "=" * 80)
print("5. 保存分析结果")
print("=" * 80)

# 保存相关系数表格
corr_df.to_csv('correlation_results.csv', index=False, encoding='utf-8-sig')
print("相关系数表格已保存为: correlation_results.csv")

# 生成分析报告
with open('correlation_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("IncreaseYTD相关性分析报告\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("【数据概览】\n")
    f.write(f"数据形状: {df.shape}\n")
    f.write(f"分析样本数: {analysis_df.shape[0]}\n\n")
    
    f.write("【Pearson相关系数表】\n")
    f.write(corr_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("【相关性分析总结】\n")
    f.write(f"高相关指标 (|r| >= 0.7): {len(high_corr)}个\n")
    for _, row in high_corr.iterrows():
        f.write(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}\n")
    
    f.write(f"\n中等相关指标 (0.3 <= |r| < 0.7): {len(medium_corr)}个\n")
    for _, row in medium_corr.iterrows():
        f.write(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}\n")
    
    f.write(f"\n低相关指标 (0.1 <= |r| < 0.3): {len(low_corr)}个\n")
    for _, row in low_corr.iterrows():
        f.write(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}\n")
    
    f.write(f"\n不相关指标 (|r| < 0.1): {len(no_corr)}个\n")
    for _, row in no_corr.iterrows():
        f.write(f"   - {row['指标']}: r = {row['Pearson相关系数']:.4f}\n")
    
    f.write(f"\n【最相关指标】\n")
    f.write(f"指标名称: {most_related['指标']}\n")
    f.write(f"Pearson相关系数: {most_related['Pearson相关系数']:.4f}\n")
    f.write(f"P值: {most_related['P值']:.2e}\n")
    
    f.write("\n【Canonical Correlation分析】\n")
    for i, corr in enumerate(canonical_corrs):
        f.write(f"第{i+1}对典型变量: r = {corr:.4f}\n")

print("分析报告已保存为: correlation_analysis_report.txt")
print("\n" + "=" * 80)
print("分析完成!")
print("=" * 80)
