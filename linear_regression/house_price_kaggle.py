"""
QSkill Internship - Python Development
Task 2: Linear Regression for House Price Prediction
Dataset: Kaggle - House Prices (Ames, Iowa) - train.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA

print("House Price Prediction using Linear Regression")
print("Dataset: Ames Housing Dataset")
print("-" * 40)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "train.csv")
out_img = os.path.join(BASE_DIR, "kaggle_house_price_regression.png")

df = pd.read_csv(csv_path)
print(f"\n Dataset Shape: {df.shape}")
print(f"   Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")

# Select useful columns
# Handle missing values

features = [
    'OverallQual',   # Overall material and finish quality (1-10)
    'GrLivArea',     # Above grade living area (sq ft)
    'GarageCars',    # Garage size in car capacity
    'GarageArea',    # Garage area in sq ft
    'TotalBsmtSF',   # Total basement area (sq ft)
    '1stFlrSF',      # First floor sq ft
    'FullBath',      # Full bathrooms above grade
    'TotRmsAbvGrd',  # Total rooms above grade
    'YearBuilt',     # House construction year
    'YearRemodAdd',  # Remodel year
    'LotArea',       # Lot size in sq ft
    'Fireplaces',    # Number of fireplaces
    'BedroomAbvGr',  # Bedrooms above grade
    'LotFrontage',   # Street-connected lot feet
    'MasVnrArea',    # Masonry veneer area (sq ft)
]

target = 'SalePrice'

# Fill missing values with median
df_model = df[features + [target]].copy()
for col in features:
    df_model[col].fillna(df_model[col].median(), inplace=True)
# Drop any remaining rows with NaN
df_model.dropna(inplace=True)

print(f"\n Features selected : {len(features)}")
print(f" Target            : SalePrice (USD)")
print(f" Missing values    : filled with column median")
print(f"\n Target (SalePrice) Summary:")
print(df_model[target].describe().apply(lambda x: f"${x:,.0f}").to_string())


# 3. TRAIN / TEST SPLIT & SCALING

X = df_model[features]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\n Train size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")


# 4. TRAIN MODEL

model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)


# 5. EVALUATION

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 20)
print(f"  R²   Score  : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"  MAE         : ${mae:,.0f}")
print(f"  RMSE        : ${rmse:,.0f}")
print("=" * 60)

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

coef_df['AbsCoeff'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('AbsCoeff', ascending=False)
print("\nMost Important Features:")
print(coef_df.to_string(index=False))


# 6. VISUALIZATIONS

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 16))
fig.suptitle("House Price Prediction Analysis",
             fontsize=17, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1: Actual vs Predicted ───────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test/1000, y_pred/1000, alpha=0.4, color='steelblue', s=20, edgecolors='none')
lims = [min(y_test.min(), y_pred.min())/1000, max(y_test.max(), y_pred.max())/1000]
ax1.plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
ax1.set_xlabel('Actual Price ($000s)')
ax1.set_ylabel('Predicted Price ($000s)')
ax1.set_title('Actual vs Predicted')
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f'R² = {r2:.3f}', transform=ax1.transAxes,
         fontsize=11, color='darkred', fontweight='bold')

# ── 2: Residuals Distribution ────────────
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test - y_pred
ax2.hist(residuals/1000, bins=40, color='coral', edgecolor='white', alpha=0.85)
ax2.axvline(0, color='black', linestyle='--', lw=1.5)
ax2.set_xlabel('Residual ($000s)')
ax2.set_ylabel('Count')
ax2.set_title('Residuals Distribution')

# ── 3: Residuals vs Predicted ────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_pred/1000, residuals/1000, alpha=0.4, color='mediumseagreen', s=20)
ax3.axhline(0, color='red', linestyle='--', lw=1.5)
ax3.set_xlabel('Predicted Price ($000s)')
ax3.set_ylabel('Residual ($000s)')
ax3.set_title('Residuals vs Predicted')

# ── 4: Feature Coefficients ──────────────
ax4 = fig.add_subplot(gs[1, :2])
colors = ['tomato' if c < 0 else 'steelblue' for c in coef_df['Coefficient']]
bars = ax4.barh(coef_df['Feature'], coef_df['Coefficient']/1000, color=colors)
ax4.axvline(0, color='black', linewidth=0.8)
ax4.set_title('Feature Coefficients (Impact on Price, $000s per SD)')
ax4.set_xlabel('Coefficient ($000s)')
for bar, val in zip(bars, coef_df['Coefficient']/1000):
    offset = 0.3 if val >= 0 else -0.3
    ax4.text(val + offset, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}k', va='center',
             ha='left' if val >= 0 else 'right', fontsize=8)

# ── 5: Correlation Heatmap (top 8 features + target) ─
ax5 = fig.add_subplot(gs[1, 2])
top8 = coef_df['Feature'].head(8).tolist()
corr = df_model[top8 + [target]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax5, annot_kws={'size': 7}, linewidths=0.4,
            cbar_kws={'shrink': 0.75})
ax5.set_title('Top Feature Correlations')
ax5.tick_params(axis='x', rotation=45, labelsize=7)
ax5.tick_params(axis='y', rotation=0,  labelsize=7)

# ── 6: SalePrice Distribution ────────────
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(df[target]/1000, bins=50, color='mediumpurple', edgecolor='white', alpha=0.85)
ax6.axvline(df[target].mean()/1000, color='red', linestyle='--', lw=1.5,
            label=f'Mean=${df[target].mean()/1000:.0f}k')
ax6.set_xlabel('Sale Price ($000s)')
ax6.set_ylabel('Count')
ax6.set_title('Sale Price Distribution')
ax6.legend(fontsize=9)

# ── 7: OverallQual vs SalePrice ──────────
ax7 = fig.add_subplot(gs[2, 1])
qual_price = df.groupby('OverallQual')['SalePrice'].median() / 1000
ax7.bar(qual_price.index, qual_price.values, color='darkorange', alpha=0.85, edgecolor='white')
ax7.set_xlabel('Overall Quality (1–10)')
ax7.set_ylabel('Median Sale Price ($000s)')
ax7.set_title('Quality vs Median Price')

# ── 8: Metrics Panel ─────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
metrics_text = (
    f"  MODEL METRICS\n"
    f"{'─'*32}\n"
    f"R² Score      :  {r2:.4f}\n"
    f"              ({r2*100:.1f}% variance explained)\n\n"
    f"MAE           :  ${mae:,.0f}\n"
    f"RMSE          :  ${rmse:,.0f}\n\n"
    f"Train Samples :  {X_train.shape[0]}\n"
    f"Test  Samples :  {X_test.shape[0]}\n"
    f"Features Used :  {len(features)}\n\n"
    f"Dataset       :  Kaggle Ames Housing\n"
    f"               Iowa, USA (2006–2010)"
)
ax8.text(0.05, 0.97, metrics_text, transform=ax8.transAxes,
         fontsize=10.5, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))


plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n Chart saved → {out_img}")
print("\nObservations:")
print("- OverallQual is the strongest predictor of house prices.")
print("- Larger living areas generally increase sale price.")
print("- Newer homes tend to sell for higher prices.")
print("- Homes with larger garages and basements command higher prices.")
print(f"- The model explains {r2*100:.1f}% of house price variation.")

# 7. SAMPLE PREDICTIONS

print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS (10 test houses)")
print("=" * 60)
sample = pd.DataFrame({
    'Actual ($)'   : y_test.values[:10],
    'Predicted ($)': y_pred[:10].astype(int),
    'Error ($)'    : (y_test.values[:10] - y_pred[:10]).astype(int)
})
print(sample.to_string(index=False))
print("\n All done!")
