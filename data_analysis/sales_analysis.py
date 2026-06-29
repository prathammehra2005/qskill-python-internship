"""
Sales Data Analysis - Internship Project
Dataset: Superstore Sales Dataset (Kaggle)
Tools used: Pandas, Matplotlib, Seaborn, NumPy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ─────────────────────────────────────────────
# STEP 1: Paths — works on any machine / OS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "sales_data.csv")
png_path = os.path.join(BASE_DIR, "sales_analysis.png")

# ─────────────────────────────────────────────
# STEP 2: Load & Inspect the CSV with Pandas
# ─────────────────────────────────────────────
df = pd.read_csv(csv_path, encoding='latin1')

print("✔ Dataset loaded from sales_data.csv")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("── df.head() ──────────────────────────────")
print(df.head())

print("\n── df.info() ──────────────────────────────")
print(df.dtypes.to_string())
print(f"\nNull values:\n{df.isnull().sum().to_string()}")

# ─────────────────────────────────────────────
# STEP 3: Basic Statistics (describe + custom)
# ─────────────────────────────────────────────
print("\n── df.describe() ──────────────────────────")
print(df[['Quantity', 'Sales', 'Profit', 'Discount']].describe().round(2))

print("\n── Column averages ────────────────────────")
numeric_cols = ['Quantity', 'Sales', 'Profit', 'Discount']
for col in numeric_cols:
    print(f"  Mean {col:15s}: {df[col].mean():.2f}")

print("\n── Average Sales by Region ─────────────────")
region_avg = df.groupby('Region')['Sales'].mean().sort_values(ascending=False)
print(region_avg.round(2).to_string())

print("\n── Average Sales by Category ───────────────")
cat_avg = df.groupby('Category')['Sales'].mean().sort_values(ascending=False)
print(cat_avg.round(2).to_string())

# ─────────────────────────────────────────────
# STEP 4: Visualizations
# ─────────────────────────────────────────────
PALETTE = ['#3266ad', '#d85a30', '#1d9e75', '#ba7517']
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': '#fafafa'})

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Superstore Sales Analysis', fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Chart 1: Bar Chart — Avg Sales by Region ──
ax1 = fig.add_subplot(gs[0, 0])
colors_map = dict(zip(region_avg.index, PALETTE))
bars = ax1.bar(region_avg.index, region_avg.values,
               color=[colors_map[r] for r in region_avg.index],
               edgecolor='white', linewidth=0.8, width=0.55)
for bar, val in zip(bars, region_avg.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'${val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='500')
ax1.set_title('Avg Sales by Region', fontweight='bold')
ax1.set_ylabel('Sales ($)')
ax1.set_xlabel('Region')
ax1.tick_params(bottom=False)
ax1.set_ylim(0, region_avg.max() * 1.18)

# ── Chart 2: Bar Chart — Avg Sales by Category ──
ax2 = fig.add_subplot(gs[0, 1])
cat_colors = ['#534ab7', '#993c1d', '#0f6e56']
bars2 = ax2.bar(cat_avg.index, cat_avg.values,
                color=cat_colors, edgecolor='white', linewidth=0.8, width=0.55)
for bar, val in zip(bars2, cat_avg.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'${val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='500')
ax2.set_title('Avg Sales by Category', fontweight='bold')
ax2.set_ylabel('Sales ($)')
ax2.set_xlabel('Category')
ax2.tick_params(bottom=False, labelrotation=10)
ax2.set_ylim(0, cat_avg.max() * 1.18)

# ── Chart 3: Scatter Plot — Sales vs Quantity ──
ax3 = fig.add_subplot(gs[1, 0])
for i, region in enumerate(df['Region'].unique()):
    sub = df[df['Region'] == region]
    ax3.scatter(sub['Quantity'], sub['Sales'],
                label=region, color=PALETTE[i % 4], alpha=0.5, s=25, edgecolors='none')
m, b = np.polyfit(df['Quantity'], df['Sales'], 1)
x_line = np.linspace(df['Quantity'].min(), df['Quantity'].max(), 100)
ax3.plot(x_line, m*x_line + b, color='#333', linewidth=1.4,
         linestyle='--', label='Trend line')
ax3.set_title('Sales vs Quantity', fontweight='bold')
ax3.set_xlabel('Quantity')
ax3.set_ylabel('Sales ($)')
ax3.legend(fontsize=9, framealpha=0.7)

# ── Chart 4: Scatter Plot — Discount vs Profit ──
ax4 = fig.add_subplot(gs[1, 1])
scatter = ax4.scatter(df['Discount'], df['Profit'],
                      c=df['Sales'], cmap='YlOrRd', alpha=0.5, s=25, edgecolors='none')
plt.colorbar(scatter, ax=ax4, label='Sales ($)', shrink=0.85)
ax4.set_title('Discount vs Profit', fontweight='bold')
ax4.set_xlabel('Discount')
ax4.set_ylabel('Profit ($)')

# ── Chart 5: Heatmap — Correlation Matrix ──
ax5 = fig.add_subplot(gs[2, :])
corr_matrix = df[['Quantity', 'Sales', 'Profit', 'Discount']].corr()
sns.heatmap(corr_matrix, ax=ax5, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'weight': '500'},
            cbar_kws={'shrink': 0.6, 'label': 'Pearson r'})
ax5.set_title('Correlation Heatmap — Numeric Features', fontweight='bold')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=20, ha='right')
ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0)

plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✔ Chart saved to {png_path}")

# ─────────────────────────────────────────────
# STEP 5: Summary Insights
# ─────────────────────────────────────────────
corr_sales_qty    = df['Sales'].corr(df['Quantity'])
corr_discount_profit = df['Discount'].corr(df['Profit'])
top_region        = region_avg.idxmax()
top_category      = cat_avg.idxmax()

print("\n── Insights ────────────────────────────────")
print(f"  Top region by avg sales    : {top_region} (${region_avg[top_region]:.0f})")
print(f"  Top category by avg sales  : {top_category} (${cat_avg[top_category]:.0f})")
print(f"  Sales ↔ Quantity corr (r)  : {corr_sales_qty:.2f}")
print(f"  Discount ↔ Profit corr (r) : {corr_discount_profit:.2f}  → higher discount = lower profit")
print(f"  Average order quantity     : {df['Quantity'].mean():.1f} units")
print(f"  Average discount given     : {df['Discount'].mean()*100:.1f}%")

print("\nBusiness Recommendations:")
print(f"- Focus marketing efforts in the {top_region} region.")
print(f"- {top_category} generates the highest average sales.")
print("- Reducing discounts is strongly associated with higher profit.")
