"""
Real-world Data Project — Retail Domain
==========================================
Applies the full data-science workflow (cleaning -> EDA -> modeling/forecasting)
to a retail sales dataset spanning 2 years across 4 product categories.

Since a live retail POS export isn't available in this environment, a
realistic synthetic daily-sales dataset is generated with the properties
real retail data actually has: an upward trend, weekly + yearly seasonality,
holiday spikes, missing values, and a few extreme outliers (e.g. a data-entry
error, a Black Friday spike) -- then cleaned, explored, and forecast.

Run: python3 retail_project.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/outputs_assets4"
os.makedirs(OUT, exist_ok=True)
np.random.seed(7)

# ---------------------------------------------------------------------------
# 1. GENERATE A REALISTIC RAW RETAIL DATASET
# ---------------------------------------------------------------------------
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
categories = ["Electronics", "Clothing", "Grocery", "Home & Garden"]

rows = []
for cat, base, trend, weekend_boost in zip(
    categories, [1200, 800, 1500, 600], [0.6, 0.3, 0.2, 0.4], [1.35, 1.5, 1.1, 1.25]
):
    for i, d in enumerate(dates):
        weekly = weekend_boost if d.dayofweek >= 5 else 1.0
        yearly_season = 1 + 0.25 * np.sin(2 * np.pi * (d.dayofyear / 365) + 1.2)
        # Holiday spikes: late Nov (Black Friday) and December
        holiday = 1.0
        if d.month == 11 and 24 <= d.day <= 30:
            holiday = 2.2
        elif d.month == 12 and d.day <= 24:
            holiday = 1.6
        noise = np.random.normal(0, 60)
        sales = base + trend * i + (base * (weekly * yearly_season * holiday - 1)) + noise
        rows.append([d, cat, max(sales, 0)])

df = pd.DataFrame(rows, columns=["date", "category", "sales"])
df["sales"] = df["sales"].round(2)

# --- Inject realistic messiness ---
# Missing values (sensor/export gaps)
missing_idx = df.sample(frac=0.02, random_state=1).index
df.loc[missing_idx, "sales"] = np.nan
# Duplicate rows (double-logged transactions on export)
dupes = df.sample(15, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)
# Outliers: a few data-entry errors (accidental extra zero)
outlier_idx = df.dropna(subset=["sales"]).sample(5, random_state=3).index
df.loc[outlier_idx, "sales"] = df.loc[outlier_idx, "sales"] * 12

print(f"Raw dataset shape: {df.shape}")
print(f"Missing values: {df['sales'].isna().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

df.to_csv(f"{OUT}/retail_sales_raw.csv", index=False)

# ---------------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------------
# 2a. Duplicates
n_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

# 2b. Missing values: forward/backward-fill within each category's time series
# (standard approach for daily sales gaps -- assume similar to recent days)
df = df.sort_values(["category", "date"])
df["sales"] = df.groupby("category")["sales"].transform(lambda s: s.interpolate().ffill().bfill())

# 2c. Outliers: cap using IQR *within each category* (fair comparison across
# categories with very different sales volumes)
def cap_outliers(group):
    q1, q3 = group["sales"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr  # generous bound -- retail data legitimately has spikes (holidays)
    n_capped = (group["sales"] > upper).sum()
    group["sales_capped"] = np.where(group["sales"] > upper, upper, group["sales"])
    return group, n_capped

capped_parts = []
total_capped = 0
for cat, g in df.groupby("category"):
    g2, n = cap_outliers(g)
    capped_parts.append(g2)
    total_capped += n
df = pd.concat(capped_parts).sort_values(["category", "date"]).reset_index(drop=True)

print(f"\nDuplicates removed: {n_dupes}")
print(f"Missing values after cleaning: {df['sales'].isna().sum()}")
print(f"Outliers capped: {total_capped}")

df.to_csv(f"{OUT}/retail_sales_cleaned.csv", index=False)

# ---------------------------------------------------------------------------
# 3. EDA — TRENDS & PATTERNS
# ---------------------------------------------------------------------------
df["month"] = df["date"].dt.to_period("M").astype(str)
df["dow"] = df["date"].dt.day_name()
df["year"] = df["date"].dt.year

# 3a. Daily sales trend by category
fig, ax = plt.subplots(figsize=(11, 5))
for cat in categories:
    sub = df[df.category == cat].sort_values("date")
    ax.plot(sub["date"], sub["sales_capped"].rolling(7).mean(), label=cat, linewidth=1.5)
ax.set_title("7-Day Rolling Average Sales by Category")
ax.set_ylabel("Sales ($)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_sales_trend.png", dpi=150)
plt.close()

# 3b. Total sales by category
fig, ax = plt.subplots(figsize=(7, 5))
totals = df.groupby("category")["sales_capped"].sum().sort_values(ascending=False)
sns.barplot(x=totals.index, y=totals.values, ax=ax, palette="crest")
ax.set_title("Total Sales by Category (2024-2025)")
ax.set_ylabel("Total Sales ($)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/02_total_by_category.png", dpi=150)
plt.close()

# 3c. Day-of-week pattern
fig, ax = plt.subplots(figsize=(8, 5))
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.boxplot(data=df, x="dow", y="sales_capped", order=dow_order, ax=ax, palette="viridis")
ax.set_title("Sales Distribution by Day of Week (All Categories)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/03_day_of_week.png", dpi=150)
plt.close()

# 3d. Monthly heatmap (category x month totals)
pivot = df.pivot_table(index="category", columns="month", values="sales_capped", aggfunc="sum")
fig, ax = plt.subplots(figsize=(14, 4.5))
sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Total Sales ($)"})
ax.set_title("Monthly Sales Heatmap by Category")
plt.tight_layout()
plt.savefig(f"{OUT}/04_monthly_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. FORECASTING MODEL
# ---------------------------------------------------------------------------
# Aggregate to total daily sales across all categories, then forecast using
# time-based features (day index, day-of-week, month, holiday flag).
daily = df.groupby("date", as_index=False)["sales_capped"].sum().rename(columns={"sales_capped": "total_sales"})
daily = daily.sort_values("date").reset_index(drop=True)
daily["day_idx"] = np.arange(len(daily))
daily["dow"] = daily["date"].dt.dayofweek
daily["month"] = daily["date"].dt.month
daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
daily["is_holiday_season"] = ((daily["date"].dt.month == 11) & (daily["date"].dt.day >= 24) |
                               (daily["date"].dt.month == 12) & (daily["date"].dt.day <= 24)).astype(int)

feature_cols = ["day_idx", "dow", "month", "is_weekend", "is_holiday_season"]
X = daily[feature_cols]
y = daily["total_sales"]

# Time-based split: train on first 20 months, test on last 4 months (no shuffling -- this is a forecast)
split_date = daily["date"].quantile(0.833)
train_mask = daily["date"] <= split_date
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
dates_test = daily.loc[~train_mask, "date"]

print(f"\nForecast train size: {len(X_train)}  |  test size: {len(X_test)}")

fc_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
}

fc_results = {}
for name, model in fc_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    fc_results[name] = {
        "model": model, "pred": pred,
        "mae": mean_absolute_error(y_test, pred),
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred),
    }
    print(f"{name}: MAE=${fc_results[name]['mae']:.0f}  RMSE=${fc_results[name]['rmse']:.0f}  R2={fc_results[name]['r2']:.3f}")

best_name = min(fc_results, key=lambda k: fc_results[k]["rmse"])

# Plot actual vs forecast for both models
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(daily["date"], daily["total_sales"], color="#95a5a6", alpha=0.5, label="Actual (full history)")
ax.plot(dates_test, y_test.values, color="#2c3e50", linewidth=2, label="Actual (test period)")
for name, r in fc_results.items():
    ax.plot(dates_test, r["pred"], linewidth=2, linestyle="--", label=f"{name} forecast")
ax.axvline(split_date, color="red", linestyle=":", alpha=0.6, label="Train/Test split")
ax.set_title("Total Daily Sales — Actual vs. Forecast")
ax.set_ylabel("Total Sales ($)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/05_forecast_vs_actual.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. SUMMARY
# ---------------------------------------------------------------------------
top_cat = totals.idxmax()
summary = f"""
RETAIL PROJECT SUMMARY
========================
Raw rows: {df.shape[0] + n_dupes}  |  Cleaned rows: {df.shape[0]}
Duplicates removed: {n_dupes}
Outliers capped (per-category IQR): {total_capped}

Top-selling category: {top_cat} (${totals[top_cat]:,.0f} total)
Weekend sales boost observed across all categories (esp. Clothing).
Clear holiday-season spike late November (Black Friday) through December.

FORECAST MODEL COMPARISON (last ~4 months held out)
{pd.DataFrame({k: {'MAE': f"${v['mae']:.0f}", 'RMSE': f"${v['rmse']:.0f}", 'R2': f"{v['r2']:.3f}"} for k, v in fc_results.items()}).T.to_string()}

Best forecasting model: {best_name}
"""
print(summary)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(summary)

print(f"\nAll charts and data saved to {OUT}")
