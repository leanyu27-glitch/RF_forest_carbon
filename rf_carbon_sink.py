#!/usr/bin/env python3
"""
随机森林碳汇模拟与修正分析 (Random Forest Carbon Sink Modeling & Correction)
===================================================================
基于CFCCD数据集, 利用随机森林模型:
  1. 构建气象因子 → 碳汇通量 (NEP/NPP/GPP) 的非线性关系
  2. 交叉验证评估模型精度
  3. 生成修正后的碳汇估算值
  4. 输出InVEST碳储量模型所需的碳库数据
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import os, json

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

OUT_DIR = "/home/claude/output"
os.makedirs(OUT_DIR, exist_ok=True)

DATA_BASE = "/home/claude/data/CFCCD datasets/CFCCD Data file"

# ===================== 1. 数据加载与融合 =====================
print("=" * 60)
print("Step 1: 数据加载与预处理")
print("=" * 60)

# 站点基本信息
site_info = {
    'CBF': {'name': 'Changbai Mountain', 'lat': 42.40, 'lon': 128.09, 'type': 'Temperate'},
    'BJF': {'name': 'Beijing',           'lat': 39.95, 'lon': 115.41, 'type': 'Warm Temperate'},
    'MXF': {'name': 'Maoxian',           'lat': 31.70, 'lon': 103.90, 'type': 'Warm Temperate'},
    'SNF': {'name': 'Shennongjia',       'lat': 31.32, 'lon': 110.49, 'type': 'Subtropical'},
    'HTF': {'name': 'Huitong',           'lat': 26.85, 'lon': 109.60, 'type': 'Subtropical'},
    'QYF': {'name': 'Qianyanzhou',       'lat': 26.73, 'lon': 115.07, 'type': 'Subtropical'},
    'ALF': {'name': 'Ailao Mountain',    'lat': 24.55, 'lon': 101.02, 'type': 'Subtropical'},
    'DHF': {'name': 'Dinghu Mountain',   'lat': 23.18, 'lon': 112.54, 'type': 'Subtropical'},
    'HSF': {'name': 'Heshan',            'lat': 22.67, 'lon': 112.88, 'type': 'Subtropical'},
    'BNF': {'name': 'Xishuangbanna',     'lat': 21.95, 'lon': 101.20, 'type': 'Tropical'},
}

# 1a. 加载所有站点的模型输入数据 (日尺度气象因子)
model_input_all = []
input_dir = os.path.join(DATA_BASE, "model input dataset")
for f in sorted(os.listdir(input_dir)):
    if f.endswith('.xlsx'):
        df = pd.read_excel(os.path.join(input_dir, f))
        # 统一列名
        if 'Site_ID' in df.columns:
            df = df.rename(columns={'Site_ID': 'Site_code'})
        model_input_all.append(df)
        print(f"  Loaded {f}: {df.shape[0]} daily records, site={df['Site_code'].iloc[0]}")

df_daily = pd.concat(model_input_all, ignore_index=True)
print(f"\n  Total daily records: {df_daily.shape[0]}")

# 1b. 聚合为年尺度气象特征 (生成丰富特征)
def agg_yearly_features(group):
    """从日尺度气象数据构建年尺度特征"""
    ta = group['Ta(℃)'].astype(float)
    par = group['PAR(mol/day)'].astype(float)
    rh = group['RH(%)'].astype(float)

    features = {
        # 温度特征
        'Ta_mean': ta.mean(),
        'Ta_std': ta.std(),
        'Ta_max': ta.max(),
        'Ta_min': ta.min(),
        'Ta_range': ta.max() - ta.min(),
        'Ta_growing_days': (ta > 5).sum(),  # 生长季天数 (>5°C)
        'Ta_warm_days': (ta > 20).sum(),
        'Ta_frost_days': (ta < 0).sum(),
        'GDD': ta[ta > 5].sum(),  # 积温 (Growing Degree Days)

        # 光合有效辐射特征
        'PAR_mean': par.mean(),
        'PAR_std': par.std(),
        'PAR_sum': par.sum(),
        'PAR_max': par.max(),

        # 相对湿度特征
        'RH_mean': rh.mean(),
        'RH_std': rh.std(),
        'RH_min': rh.min(),
        'RH_dry_days': (rh < 50).sum(),

        # 季节特征 (按月份近似)
        'Ta_spring': group.loc[group['Month'].isin([3,4,5]), 'Ta(℃)'].astype(float).mean() if len(group.loc[group['Month'].isin([3,4,5])]) > 0 else np.nan,
        'Ta_summer': group.loc[group['Month'].isin([6,7,8]), 'Ta(℃)'].astype(float).mean() if len(group.loc[group['Month'].isin([6,7,8])]) > 0 else np.nan,
        'Ta_autumn': group.loc[group['Month'].isin([9,10,11]), 'Ta(℃)'].astype(float).mean() if len(group.loc[group['Month'].isin([9,10,11])]) > 0 else np.nan,
        'Ta_winter': group.loc[group['Month'].isin([12,1,2]), 'Ta(℃)'].astype(float).mean() if len(group.loc[group['Month'].isin([12,1,2])]) > 0 else np.nan,

        'PAR_spring': group.loc[group['Month'].isin([3,4,5]), 'PAR(mol/day)'].astype(float).mean() if len(group.loc[group['Month'].isin([3,4,5])]) > 0 else np.nan,
        'PAR_summer': group.loc[group['Month'].isin([6,7,8]), 'PAR(mol/day)'].astype(float).mean() if len(group.loc[group['Month'].isin([6,7,8])]) > 0 else np.nan,
    }

    # 加入地理信息
    site_code = group['Site_code'].iloc[0]
    if site_code in site_info:
        features['Latitude'] = site_info[site_code]['lat']
        features['Longitude'] = site_info[site_code]['lon']

    return pd.Series(features)

df_yearly_meteo = df_daily.groupby(['Site_code', 'Year'], group_keys=False).apply(agg_yearly_features).reset_index()
print(f"  Yearly meteorological features: {df_yearly_meteo.shape}")

# 1c. 加载同化碳汇数据 (年尺度)
cs_file = os.path.join(DATA_BASE, "assimilation dataset", "assimilated time-continous carbon sequestration data.xlsx")
wb_cs = pd.ExcelFile(cs_file)

carbon_all = []
for sheet in wb_cs.sheet_names:
    df_c = pd.read_excel(wb_cs, sheet_name=sheet)
    df_c['Site_code'] = sheet
    # 重命名列, 去除std列
    cols_keep = {
        'year': 'Year',
        'biomass_C (g C m-2)': 'biomass_C',
        'SOC (g C m-2)': 'SOC',
        'Total_C (g C m-2)': 'Total_C',
        'NEP (g C m-2 yr-1)': 'NEP',
        'GPP(g C m-2 yr-1)': 'GPP',
        'NPP(g C m-2 yr-1)': 'NPP',
        'Ra(g C m-2 yr-1)': 'Ra',
        'Rh(g C m-2 yr-1)': 'Rh',
    }
    df_c = df_c.rename(columns=cols_keep)
    df_c = df_c[['Site_code', 'Year'] + [v for v in cols_keep.values() if v not in ['Site_code', 'Year']]]
    carbon_all.append(df_c)

df_carbon = pd.concat(carbon_all, ignore_index=True)
print(f"  Carbon sequestration records: {df_carbon.shape}")

# 1d. 合并气象-碳汇数据
df_merged = pd.merge(df_yearly_meteo, df_carbon, on=['Site_code', 'Year'], how='inner')
print(f"  Merged dataset: {df_merged.shape}")
print(f"  Sites: {sorted(df_merged['Site_code'].unique())}")
print(f"  Years: {sorted(df_merged['Year'].unique())}")
print(f"  Samples per site: {df_merged.groupby('Site_code').size().to_dict()}")

# ===================== 2. 随机森林建模 =====================
print("\n" + "=" * 60)
print("Step 2: 随机森林模型训练与评估")
print("=" * 60)

# 目标变量
targets = ['NEP', 'NPP', 'GPP', 'biomass_C', 'SOC', 'Total_C']
# 特征列
feature_cols = [c for c in df_merged.columns if c not in 
                ['Site_code', 'Year'] + targets + ['Ra', 'Rh']]

# 处理缺失值
df_model = df_merged.dropna(subset=feature_cols + targets)
print(f"  Valid samples after dropping NaN: {df_model.shape[0]}")
print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

X = df_model[feature_cols].values
site_codes = df_model['Site_code'].values
years = df_model['Year'].values

# 存储结果
results = {}
models = {}
predictions_all = {}

for target in targets:
    print(f"\n  --- Modeling {target} ---")
    y = df_model[target].values

    # 超参数优化
    from sklearn.model_selection import KFold, RandomizedSearchCV
    from scipy.stats import randint, uniform

    param_dist = {
        'n_estimators': [100, 150, 200],
        'max_depth': [5, 8, 12, None],
        'min_samples_split': [2, 3, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 0.5, 0.8],
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # 使用Leave-One-Out交叉验证 (因为样本量不大)
    loo = LeaveOneOut()

    kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    gs = RandomizedSearchCV(rf_base, param_dist, n_iter=20, cv=kf, scoring='r2',
                            n_jobs=-1, refit=True, random_state=42)
    gs.fit(X, y)

    best_rf = gs.best_estimator_
    print(f"    Best params: {gs.best_params_}")
    print(f"    Best CV R²: {gs.best_score_:.4f}")

    # LOO 预测以获得每个样本的修正值
    y_pred_loo = cross_val_predict(best_rf, X, y, cv=loo)

    # 最终模型在全部数据上训练
    best_rf.fit(X, y)

    # 评估指标
    r2 = r2_score(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    mae = mean_absolute_error(y, y_pred_loo)
    bias = np.mean(y_pred_loo - y)

    print(f"    LOO R²: {r2:.4f}")
    print(f"    LOO RMSE: {rmse:.2f}")
    print(f"    LOO MAE: {mae:.2f}")
    print(f"    Mean Bias: {bias:.2f}")

    # 特征重要性
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    results[target] = {
        'r2': r2, 'rmse': rmse, 'mae': mae, 'bias': bias,
        'y_true': y, 'y_pred': y_pred_loo,
        'feat_imp': feat_imp,
        'best_params': gs.best_params_,
    }
    models[target] = best_rf
    predictions_all[target] = y_pred_loo

# ===================== 3. 碳汇修正 =====================
print("\n" + "=" * 60)
print("Step 3: 碳汇模拟修正")
print("=" * 60)

# 3a. 计算修正系数 (基于RF预测与原始同化值的偏差)
df_corrected = df_model[['Site_code', 'Year']].copy()

for target in targets:
    y_true = df_model[target].values
    y_pred = predictions_all[target]

    # 修正方案: RF预测值作为模拟值, 用偏差比率修正
    # correction_ratio = mean(observed/predicted) per site
    df_temp = pd.DataFrame({
        'Site_code': site_codes,
        'y_true': y_true,
        'y_pred': y_pred
    })

    # 按站点计算修正系数
    site_corrections = {}
    for site in df_temp['Site_code'].unique():
        mask = df_temp['Site_code'] == site
        obs = df_temp.loc[mask, 'y_true'].values
        pred = df_temp.loc[mask, 'y_pred'].values
        # 避免除以0
        valid = pred != 0
        if valid.sum() > 0:
            ratio = np.mean(obs[valid] / pred[valid])
        else:
            ratio = 1.0
        site_corrections[site] = ratio

    print(f"\n  {target} site correction ratios:")
    for s, r in sorted(site_corrections.items()):
        print(f"    {s}: {r:.4f}")

    # 修正后的值 = RF预测 * 修正系数
    corrected = np.array([
        predictions_all[target][i] * site_corrections.get(site_codes[i], 1.0)
        for i in range(len(y_pred))
    ])

    df_corrected[f'{target}_original'] = y_true
    df_corrected[f'{target}_RF_pred'] = y_pred
    df_corrected[f'{target}_corrected'] = corrected

# ===================== 4. InVEST碳储量模型输入 =====================
print("\n" + "=" * 60)
print("Step 4: 生成InVEST碳储量模型输入数据")
print("=" * 60)

# InVEST Carbon模型需要4个碳库: 地上生物量碳(C_above), 地下生物量碳(C_below), 土壤碳(C_soil), 死有机质碳(C_dead)
# 基于CFCCD数据, 我们可以估算:
#   - C_above + C_below ≈ biomass_C (需要按比例分配, 一般地上:地下 ≈ 3:1~4:1)
#   - C_soil ≈ SOC
#   - C_dead 可从凋落物和 Rh 近似

# 加载凋落物数据
try:
    df_litter = pd.read_excel(os.path.join(DATA_BASE, "observation-based basic element dataset", "observation-based litterfall.xlsx"))
    has_litter = True
    print("  Loaded litterfall data.")
except:
    has_litter = False
    print("  No litterfall data loaded, using approximation.")

# 生成InVEST碳池查找表 (按站点/植被类型的多年均值)
invest_table = []
for site_code in sorted(site_info.keys()):
    site_data = df_corrected[df_corrected['Site_code'] == site_code]
    if len(site_data) == 0:
        continue

    # 使用修正后的均值
    biomass_c = site_data['biomass_C_corrected'].mean()
    soc = site_data['SOC_corrected'].mean()
    total_c = site_data['Total_C_corrected'].mean()
    nep = site_data['NEP_corrected'].mean()

    # 碳库分配 (单位: Mg C/ha = g C/m² × 0.01)
    c_above = biomass_c * 0.75 * 0.01   # 地上生物量碳 (75% of biomass)
    c_below = biomass_c * 0.25 * 0.01   # 地下生物量碳 (25% of biomass)
    c_soil  = soc * 0.01                # 土壤有机碳
    c_dead  = (total_c - biomass_c - soc) * 0.01 if total_c > biomass_c + soc else 0.5  # 死有机质碳

    invest_table.append({
        'lucode': site_code,
        'LULC_name': site_info[site_code]['name'],
        'Vegetation_type': site_info[site_code]['type'],
        'Latitude': site_info[site_code]['lat'],
        'Longitude': site_info[site_code]['lon'],
        'C_above (Mg C/ha)': round(c_above, 2),
        'C_below (Mg C/ha)': round(c_below, 2),
        'C_soil (Mg C/ha)':  round(c_soil, 2),
        'C_dead (Mg C/ha)':  round(c_dead, 2),
        'C_total (Mg C/ha)': round(c_above + c_below + c_soil + c_dead, 2),
        'NEP_mean (g C/m²/yr)': round(nep, 2),
        'NPP_mean (g C/m²/yr)': round(site_data['NPP_corrected'].mean(), 2),
        'GPP_mean (g C/m²/yr)': round(site_data['GPP_corrected'].mean(), 2),
    })

df_invest = pd.DataFrame(invest_table)
print("\n  InVEST Carbon Pool Lookup Table:")
print(df_invest[['lucode', 'LULC_name', 'C_above (Mg C/ha)', 'C_below (Mg C/ha)', 
                  'C_soil (Mg C/ha)', 'C_dead (Mg C/ha)', 'C_total (Mg C/ha)']].to_string(index=False))

# ===================== 5. 可视化 =====================
print("\n" + "=" * 60)
print("Step 5: 生成可视化报告")
print("=" * 60)

# 配色方案
COLORS = {
    'primary': '#2C5F7C',
    'secondary': '#E07B54',
    'accent1': '#5B9A6F',
    'accent2': '#9B7ED8',
    'accent3': '#D4A843',
    'bg': '#FAFBFC',
    'text': '#2D3436',
    'grid': '#E8ECEF',
}

site_colors = {
    'ALF': '#E74C3C', 'BJF': '#3498DB', 'BNF': '#2ECC71', 'CBF': '#9B59B6',
    'DHF': '#E67E22', 'HSF': '#1ABC9C', 'HTF': '#F39C12', 'MXF': '#E91E63',
    'QYF': '#00BCD4', 'SNF': '#8BC34A'
}

# --- Figure 1: 模型性能总览 ---
fig = plt.figure(figsize=(20, 16), facecolor=COLORS['bg'])
fig.suptitle('Random Forest Carbon Sink Modeling — Performance Overview',
             fontsize=18, fontweight='bold', color=COLORS['text'], y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                       left=0.07, right=0.95, top=0.93, bottom=0.06)

# 1-6: 散点图 observed vs predicted
for idx, target in enumerate(targets):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])

    y_true = results[target]['y_true']
    y_pred = results[target]['y_pred']
    r2 = results[target]['r2']
    rmse = results[target]['rmse']

    plotted_sites = set()
    for i, site in enumerate(df_model['Site_code'].values):
        lbl = site if site not in plotted_sites else ''
        plotted_sites.add(site)
        ax.scatter(y_true[i], y_pred[i], c=site_colors.get(site, '#666'),
                   s=50, alpha=0.8, edgecolor='white', linewidth=0.5, label=lbl)

    # 1:1 line
    lims = [min(y_true.min(), y_pred.min()) * 0.9, max(y_true.max(), y_pred.max()) * 1.1]
    ax.plot(lims, lims, '--', color=COLORS['secondary'], linewidth=1.5, alpha=0.7, label='1:1 line')

    # Regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(x_line, p(x_line), '-', color=COLORS['primary'], linewidth=2, alpha=0.8)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f'Observed {target}', fontsize=10)
    ax.set_ylabel(f'Predicted {target}', fontsize=10)
    ax.set_title(f'{target}', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])

    # Metrics text
    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.1f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLORS['grid'], alpha=0.9))

# Legend
handles, labels = [], []
for site in sorted(site_colors.keys()):
    if site in df_model['Site_code'].values:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=site_colors[site], markersize=8))
        labels.append(site)
fig.legend(handles, labels, loc='lower center', ncol=10, fontsize=9,
           frameon=True, fancybox=True, framealpha=0.9)

plt.savefig(os.path.join(OUT_DIR, 'fig1_model_performance.png'), dpi=150)
print("  Saved fig1_model_performance.png")
plt.close()

# --- Figure 2: 特征重要性 ---
fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=COLORS['bg'])
fig.suptitle('Feature Importance — Top 10 Features per Carbon Variable',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)

for idx, target in enumerate(targets):
    ax = axes[idx // 3, idx % 3]
    feat_imp = results[target]['feat_imp'].head(10)

    bars = ax.barh(range(len(feat_imp)), feat_imp['importance'].values,
                   color=COLORS['primary'], alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(feat_imp)))
    ax.set_yticklabels(feat_imp['feature'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_title(f'{target}', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, axis='x', alpha=0.3, color=COLORS['grid'])

    # 添加数值标签
    for bar, val in zip(bars, feat_imp['importance'].values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=8, color=COLORS['text'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'fig2_feature_importance.png'), dpi=150)
print("  Saved fig2_feature_importance.png")
plt.close()

# --- Figure 3: 碳汇时间序列 (原始 vs RF修正) ---
fig, axes = plt.subplots(2, 5, figsize=(24, 10), facecolor=COLORS['bg'])
fig.suptitle('NEP Time Series: Original vs RF-Corrected (by Site)',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)

for idx, site_code in enumerate(sorted(df_corrected['Site_code'].unique())):
    ax = axes[idx // 5, idx % 5]
    site_data = df_corrected[df_corrected['Site_code'] == site_code].sort_values('Year')

    ax.plot(site_data['Year'], site_data['NEP_original'], 'o-',
            color=COLORS['primary'], linewidth=2, markersize=5, label='Assimilated', alpha=0.9)
    ax.plot(site_data['Year'], site_data['NEP_RF_pred'], 's--',
            color=COLORS['secondary'], linewidth=1.5, markersize=4, label='RF Predicted', alpha=0.8)
    ax.plot(site_data['Year'], site_data['NEP_corrected'], '^-',
            color=COLORS['accent1'], linewidth=1.5, markersize=4, label='RF Corrected', alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_title(f'{site_code} ({site_info[site_code]["name"]})',
                 fontsize=10, fontweight='bold', color=COLORS['text'])
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('NEP (g C m⁻² yr⁻¹)', fontsize=9)
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    if idx == 0:
        ax.legend(fontsize=7, loc='best')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'fig3_nep_timeseries.png'), dpi=150)
print("  Saved fig3_nep_timeseries.png")
plt.close()

# --- Figure 4: InVEST碳库对比 ---
fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS['bg'])
fig.suptitle('InVEST Carbon Pool Distribution by Site',
             fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)

sites_list = df_invest['lucode'].values
x = np.arange(len(sites_list))
width = 0.2

bars1 = ax.bar(x - 1.5*width, df_invest['C_above (Mg C/ha)'], width,
               label='C_above', color=COLORS['accent1'], edgecolor='white')
bars2 = ax.bar(x - 0.5*width, df_invest['C_below (Mg C/ha)'], width,
               label='C_below', color=COLORS['primary'], edgecolor='white')
bars3 = ax.bar(x + 0.5*width, df_invest['C_soil (Mg C/ha)'], width,
               label='C_soil', color=COLORS['secondary'], edgecolor='white')
bars4 = ax.bar(x + 1.5*width, df_invest['C_dead (Mg C/ha)'], width,
               label='C_dead', color=COLORS['accent2'], edgecolor='white')

ax.set_xlabel('Site', fontsize=12)
ax.set_ylabel('Carbon Pool (Mg C/ha)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n({site_info[s]["name"][:8]})' for s in sites_list], fontsize=9)
ax.legend(fontsize=10, loc='upper left')
ax.set_facecolor(COLORS['bg'])
ax.grid(True, axis='y', alpha=0.3, color=COLORS['grid'])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT_DIR, 'fig4_invest_carbon_pools.png'), dpi=150)
print("  Saved fig4_invest_carbon_pools.png")
plt.close()

# --- Figure 5: 碳汇与环境因子关系热力图 ---
fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
corr_cols = ['Ta_mean', 'PAR_mean', 'RH_mean', 'GDD', 'Ta_range', 'Latitude',
             'NEP', 'NPP', 'GPP', 'biomass_C', 'SOC', 'Total_C']
corr_data = df_model[corr_cols].corr()
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', center=0,
            cmap='RdBu_r', ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix: Climate Factors vs Carbon Variables',
             fontsize=14, fontweight='bold', color=COLORS['text'])
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_correlation_heatmap.png'), dpi=150)
print("  Saved fig5_correlation_heatmap.png")
plt.close()


# ===================== 6. 导出数据 =====================
print("\n" + "=" * 60)
print("Step 6: 导出结果数据")
print("=" * 60)

# 6a. 详细修正结果
df_corrected.to_csv(os.path.join(OUT_DIR, 'carbon_sink_corrected.csv'), index=False)
print("  Saved carbon_sink_corrected.csv")

# 6b. InVEST碳库查找表
df_invest.to_csv(os.path.join(OUT_DIR, 'invest_carbon_pool_table.csv'), index=False)
print("  Saved invest_carbon_pool_table.csv")

# 6c. 模型评估汇总
eval_summary = []
for target in targets:
    r = results[target]
    eval_summary.append({
        'Variable': target,
        'R²': round(r['r2'], 4),
        'RMSE': round(r['rmse'], 2),
        'MAE': round(r['mae'], 2),
        'Bias': round(r['bias'], 2),
        'Best_Params': str(r['best_params']),
    })
df_eval = pd.DataFrame(eval_summary)
df_eval.to_csv(os.path.join(OUT_DIR, 'model_evaluation_summary.csv'), index=False)
print("  Saved model_evaluation_summary.csv")

# 6d. 特征重要性
with pd.ExcelWriter(os.path.join(OUT_DIR, 'feature_importance_all.xlsx')) as writer:
    for target in targets:
        results[target]['feat_imp'].to_excel(writer, sheet_name=target, index=False)
print("  Saved feature_importance_all.xlsx")

# 6e. 年度碳汇通量 (用于InVEST碳储量与碳汇动态模型)
invest_flux = df_corrected[['Site_code', 'Year']].copy()
for t in ['NEP', 'NPP', 'GPP']:
    invest_flux[f'{t}_corrected_gCm2yr'] = df_corrected[f'{t}_corrected']
    # 转换为 Mg C/ha/yr
    invest_flux[f'{t}_corrected_MgCha_yr'] = df_corrected[f'{t}_corrected'] * 0.01
invest_flux.to_csv(os.path.join(OUT_DIR, 'invest_annual_carbon_flux.csv'), index=False)
print("  Saved invest_annual_carbon_flux.csv")

print("\n" + "=" * 60)
print("ALL DONE! Summary:")
print("=" * 60)
print(f"\nModel Performance (LOO Cross-Validation):")
print(df_eval[['Variable', 'R²', 'RMSE']].to_string(index=False))
print(f"\nInVEST Carbon Pool Table (Mg C/ha):")
print(df_invest[['lucode', 'C_above (Mg C/ha)', 'C_below (Mg C/ha)', 'C_soil (Mg C/ha)', 'C_dead (Mg C/ha)']].to_string(index=False))
print(f"\nOutput files saved to: {OUT_DIR}")
