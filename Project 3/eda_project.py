"""
Exploratory Data Analysis (EDA) Project
=========================================
Dataset: Diamonds (public dataset, via seaborn) — ~54,000 diamonds with
         cut, color, clarity, carat, dimensions, and price.

Goal:
  - Statistical summaries of the dataset
  - Uncover patterns, trends, and correlations
  - Identify the key factors that influence diamond price
  - Present insights visually

Run: python3 eda_project.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

sns.set_theme(style="whitegrid", palette="deep")
OUT = "/home/claude/outputs_assets3"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = sns.load_dataset("diamonds")
print(f"Dataset shape: {df.shape}")
print(df.info())

# ---------------------------------------------------------------------------
# 2. STATISTICAL SUMMARY
# ---------------------------------------------------------------------------
numeric_cols = ["carat", "depth", "table", "price", "x", "y", "z"]
summary_stats = df[numeric_cols].describe().T.round(2)
print("\nSummary statistics:\n", summary_stats)
summary_stats.to_csv(f"{OUT}/summary_statistics.csv")

# Quick data-quality check (this dataset is already clean, but we verify)
print(f"\nMissing values total: {df.isna().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
# A few rows have 0 for x/y/z (physically impossible dimensions) -- flag as
# data-quality outliers typical of a real EDA pass.
zero_dim_rows = (df[["x", "y", "z"]] == 0).any(axis=1).sum()
print(f"Rows with a zero dimension (likely data errors): {zero_dim_rows}")
df_eda = df[~(df[["x", "y", "z"]] == 0).any(axis=1)].copy()
print(f"Shape used for analysis after dropping those rows: {df_eda.shape}")

# ---------------------------------------------------------------------------
# 3. DISTRIBUTIONS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df_eda["price"], bins=50, ax=axes[0, 0], color="#2980b9")
axes[0, 0].set_title("Price Distribution")
sns.histplot(df_eda["carat"], bins=50, ax=axes[0, 1], color="#8e44ad")
axes[0, 1].set_title("Carat Distribution")
sns.histplot(df_eda["depth"], bins=50, ax=axes[1, 0], color="#16a085")
axes[1, 0].set_title("Depth % Distribution")
sns.histplot(df_eda["table"], bins=50, ax=axes[1, 1], color="#d35400")
axes[1, 1].set_title("Table % Distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/01_distributions.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
corr = df_eda[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap — Numeric Features")
plt.tight_layout()
plt.savefig(f"{OUT}/02_correlation_heatmap.png", dpi=150)
plt.close()

price_corr = corr["price"].drop("price").sort_values(ascending=False)
print("\nCorrelation with price:\n", price_corr)

# ---------------------------------------------------------------------------
# 5. CARAT VS PRICE (the strongest relationship) — colored by cut
# ---------------------------------------------------------------------------
sample = df_eda.sample(4000, random_state=42)  # sample for a readable scatter
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=sample, x="carat", y="price", hue="cut", alpha=0.6, s=25, ax=ax, palette="viridis")
ax.set_title("Carat vs Price (colored by Cut Quality)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_carat_vs_price.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. PRICE BY CATEGORICAL QUALITY FACTORS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order = sorted(df_eda["color"].unique())
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

sns.boxplot(data=df_eda, x="cut", y="price", order=cut_order, ax=axes[0], palette="Blues")
axes[0].set_title("Price by Cut")
axes[0].tick_params(axis="x", rotation=30)

sns.boxplot(data=df_eda, x="color", y="price", order=color_order, ax=axes[1], palette="Purples")
axes[1].set_title("Price by Color Grade (D=best, J=worst)")

sns.boxplot(data=df_eda, x="clarity", y="price", order=[c for c in clarity_order if c in df_eda["clarity"].unique()], ax=axes[2], palette="Greens")
axes[2].set_title("Price by Clarity Grade")
axes[2].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT}/04_price_by_quality.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. AVERAGE PRICE PER CARAT BY CUT (normalizing out size effect)
# ---------------------------------------------------------------------------
df_eda["price_per_carat"] = df_eda["price"] / df_eda["carat"]
ppc_by_cut = df_eda.groupby("cut", observed=True)["price_per_carat"].mean().reindex(cut_order)
print("\nAvg price-per-carat by cut:\n", ppc_by_cut)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=ppc_by_cut.index, y=ppc_by_cut.values, ax=ax, palette="mako")
ax.set_ylabel("Avg Price per Carat ($)")
ax.set_title("Average Price-per-Carat by Cut Quality")
plt.tight_layout()
plt.savefig(f"{OUT}/05_price_per_carat_by_cut.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. KEY FINDINGS
# ---------------------------------------------------------------------------
top_factor = price_corr.abs().idxmax()
findings = f"""
KEY EDA FINDINGS
-----------------
Rows analyzed: {df_eda.shape[0]} (of {df.shape[0]} raw; {zero_dim_rows} rows dropped for impossible 0-dimensions)
Price range: ${df_eda['price'].min():,} - ${df_eda['price'].max():,}
Median price: ${df_eda['price'].median():,.0f}
Mean carat: {df_eda['carat'].mean():.2f}

Correlation with price (strongest to weakest):
{price_corr.round(3).to_string()}

Strongest single influencing factor on price: '{top_factor}' (r = {price_corr[top_factor]:.3f})

Notable pattern: carat (size) alone explains most of the variation in price
(r = {price_corr['carat']:.3f}), far more than cut, color, or clarity grade.
When price is normalized per carat, 'Premium' cut diamonds command the
highest average price per carat (${ppc_by_cut['Premium']:.0f}), while
'Fair' cut diamonds have the lowest (${ppc_by_cut['Fair']:.0f}) -- so cut
quality does matter, it's just a smaller effect than raw carat weight, and
only becomes visible once size is controlled for.
"""
print(findings)
with open(f"{OUT}/summary.txt", "w") as f:
    f.write(findings)

print(f"\nAll charts and stats saved to {OUT}")
